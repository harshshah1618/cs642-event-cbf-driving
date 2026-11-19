"""Select common baseline checkpoints and inference modes using development reports."""

import argparse
import json
from pathlib import Path

from driving.safemetadrive.selection import select_baselines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="JSON list of local shard-report paths")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in json.loads(Path(args.reports).read_text())]
    result = select_baselines(reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["selection"], indent=2))
