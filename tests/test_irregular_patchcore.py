import torch
from driving.irregular.patchcore import PatchCore, greedy_coreset, nearest_distance


def test_nearest_distance_matches_cdist():
    gen = torch.Generator().manual_seed(0)
    q, bank = torch.randn(300, 16, generator=gen), torch.randn(500, 16, generator=gen)
    expected = torch.cdist(q, bank).min(1).values
    assert torch.allclose(nearest_distance(q, bank, chunk=64), expected, atol=1e-4)


def test_coreset_covers_every_cluster():
    gen = torch.Generator().manual_seed(1)
    centres = torch.tensor([[0.0] * 8, [50.0] * 8, [-50.0] * 8, [0.0] * 4 + [80.0] * 4])
    # Cluster sizes 1000/10/10/10: random sampling would likely miss the small ones.
    sizes = [1000, 10, 10, 10]
    x = torch.cat([c + torch.randn(n, 8, generator=gen) for c, n in zip(centres, sizes, strict=True)])
    labels = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    keep = greedy_coreset(x, 8, projection_dim=8)
    assert set(labels[keep].tolist()) == {0, 1, 2, 3}


def test_patchcore_scores_novel_patch_highest():
    gen = torch.Generator().manual_seed(2)
    bank = torch.randn(400, 32, generator=gen)
    model = PatchCore(bank, "test", (4, 5))
    normal = torch.randn(2, 4, 5, 32, generator=gen) * 0.3
    novel = normal.clone()
    novel[1, 2, 3] += 25.0  # one far-out patch in frame 1
    scores, maps = model.score(torch.cat([normal[:1], novel[1:]]))
    assert scores[1] > scores[0]
    assert maps[1].argmax() == 2 * 5 + 3


def test_save_and_load_round_trip(tmp_path):
    model = PatchCore(torch.randn(10, 4), "dinov2_small", (18, 32))
    model.save(tmp_path / "bank.pt")
    loaded = PatchCore.load(tmp_path / "bank.pt")
    assert torch.equal(loaded.bank, model.bank) and loaded.grid == (18, 32)
