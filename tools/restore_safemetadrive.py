"""Restore checksummed private-release artifacts without Modal or ML dependencies."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_target(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Archive path leaves destination: {name}")
    return path


def restore_package(archive, package, destination):
    if digest(archive) != package["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    expected = {row["path"]: row for row in package["files"]}
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            target = safe_target(destination, member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            row = expected[member.name]
            if member.size != row["bytes"]:
                raise ValueError(f"Size mismatch: {member.name}")
            seen.add(member.name)
            if target.exists():
                if target.is_file() and digest(target) == row["sha256"]:
                    continue
                raise FileExistsError(f"Refusing to replace differing file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            assert source is not None
            fd, temporary = tempfile.mkstemp(prefix=".restore-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(source, output)
                if digest(temporary) != row["sha256"]:
                    raise ValueError(f"File checksum mismatch: {member.name}")
                Path(temporary).replace(target)
            finally:
                Path(temporary).unlink(missing_ok=True)
    if seen != expected.keys():
        raise ValueError(f"Archive is missing {len(expected.keys() - seen)} files")
    return len(seen)


def rebase_paths(destination):
    """Derive machine-local indexes; keep frozen plans/original metadata intact."""
    destination = Path(destination).resolve()
    root = destination / "runs/safemetadrive"
    study = root / "heldout-study-v1"
    plan = study / "frozen-plan.json"
    if plan.exists():
        groups = json.loads(plan.read_text())["groups"]
        local = {
            name: [
                [str(root / path.removeprefix("/results/")) for path in paths] for paths in replicas
            ]
            for name, replicas in groups.items()
        }
        (study / "groups-local.json").write_text(json.dumps(local, indent=2) + "\n")
    original = study / "artifact-map.original.json"
    if original.exists():
        mapping = json.loads(original.read_text())
        for remote, row in mapping["files"].items():
            relative = remote.removeprefix("/results/")
            path = safe_target(root, relative)
            row["path"] = str(path)
            if path.exists():
                if digest(path) != row["sha256"]:
                    raise ValueError(f"Frozen model checksum mismatch: {path}")
                # Preserve documented historical checkpoint commands as aliases.
                alias = safe_target(study / "artifacts", relative)
                if alias.exists():
                    if not alias.is_file() or digest(alias) != row["sha256"]:
                        raise FileExistsError(f"Checkpoint alias differs: {alias}")
                else:
                    alias.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(path, alias)
                    except OSError:
                        shutil.copyfile(path, alias)
        (study / "artifact-map.json").write_text(json.dumps(mapping, indent=2) + "\n")


def expand_datasets(destination):
    """Packed image shards were verified by the outer release manifest first."""
    root = Path(destination) / "runs/safemetadrive"
    count = 0
    for dataset in ("camera-data-v1", "camera-recovery-data-v1"):
        data = root / dataset / "data"
        for archive in sorted((root / dataset / "archives").glob("shard-*.tar")):
            prefix = archive.stem + "/"
            with tarfile.open(archive, "r:") as stream:
                for member in stream:
                    if not member.isfile() or not member.name.startswith(prefix):
                        raise ValueError(f"Unexpected dataset member: {member.name}")
                    target = safe_target(data, member.name)
                    source = stream.extractfile(member)
                    assert source is not None
                    payload = source.read()
                    if target.exists():
                        if not target.is_file() or hashlib.sha256(payload).hexdigest() != digest(
                            target
                        ):
                            raise FileExistsError(f"Dataset file differs: {target}")
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts/safemetadrive/manifest.json"
    )
    parser.add_argument("--packages", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--from-dir", type=Path, help="Use downloaded archives; never access network"
    )
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--expand-datasets", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    packages = {row["id"]: row for row in manifest["packages"]}
    if args.list:
        for key, row in packages.items():
            print(f"{key}: {row['bytes'] / 1e6:.1f} MB — {row['description']}")
        return
    requested = list(packages) if args.all else args.packages or ["records", "models"]
    unknown = set(requested) - packages.keys()
    if unknown:
        parser.error(f"Unknown packages: {sorted(unknown)}")
    chosen = []
    visiting = set()

    def include(key):
        if key in visiting:
            raise ValueError("Cyclic package dependencies")
        if key in chosen:
            return
        visiting.add(key)
        for required in packages[key].get("requires", []):
            include(required)
        visiting.remove(key)
        chosen.append(key)

    for key in requested:
        include(key)
    cache = args.from_dir or args.destination / "runs/safemetadrive-downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for key in chosen:
        package = packages[key]
        archive = cache / package["name"]
        if not archive.exists():
            if args.from_dir:
                raise FileNotFoundError(archive)
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        manifest["release_tag"],
                        "--repo",
                        manifest["repository"],
                        "--pattern",
                        package["name"],
                        "--dir",
                        temporary,
                    ],
                    check=True,
                )
                downloaded = Path(temporary) / package["name"]
                if digest(downloaded) != package["sha256"]:
                    raise ValueError(f"Downloaded checksum mismatch: {package['name']}")
                downloaded.replace(archive)
        count = restore_package(archive, package, args.destination)
        print(f"Restored {key}: {count} files", flush=True)
    rebase_paths(args.destination)
    if args.expand_datasets:
        print(f"Expanded {expand_datasets(args.destination)} dataset files")
    print("Ready. No Modal functions were started.")


if __name__ == "__main__":
    main()
