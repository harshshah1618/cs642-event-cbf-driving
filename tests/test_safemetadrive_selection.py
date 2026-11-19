import pytest
from driving.safemetadrive.protocol import environment_config
from driving.safemetadrive.selection import select_baselines


def development_reports():
    reports = []
    for algorithm in ("PPOLag", "CPO"):
        for epoch in (50,100,150,200,250):
            for deterministic in (True,False):
                for seed in range(3):
                    # The best individual mean-action model succeeds twice,
                    # but stochastic epoch100 is stronger across all seeds.
                    successes = (2 if deterministic and seed == 0 else 0) if epoch == 100 else 0
                    if epoch == 100 and not deterministic:
                        successes = 1
                    reports.append(dict(split="validation", config=environment_config("validation"),
                        policy="native_learned", parameters=dict(deterministic=deterministic,
                            checkpoint=f"/results/run/{algorithm}-seed{seed}/models/epoch-{epoch}.pt"),
                        episodes=[dict(seed=100+i, success=i < successes, cost=0, reward=0)
                                  for i in range(50)]))
    return reports


def test_baseline_selection_uses_common_checkpoint_and_mode_across_seeds():
    result = select_baselines(development_reports())
    for row in result["selection"].values():
        assert row["epoch"] == 100
        assert row["deterministic"] is False
        assert row["success_rate"] == pytest.approx(1/50)
        assert len(row["checkpoints"]) == 3


def test_baseline_selection_rejects_test_maps_and_missing_development_maps():
    reports = development_reports()
    reports[0]["split"] = "test"
    with pytest.raises(ValueError, match="reserved development"):
        select_baselines(reports)
    reports[0]["split"] = "validation"
    reports[0]["episodes"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        select_baselines(reports)
