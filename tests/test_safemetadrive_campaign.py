import hashlib
import json

import pytest
from driving.safemetadrive import campaign
from driving.safemetadrive.protocol import environment_config


def test_final_plan_uses_development_selection_and_preserves_frozen_artifacts(tmp_path,monkeypatch):
    reports = {}
    native = []
    for algorithm in ("PPOLag","CPO"):
        for epoch in (50,100,150,200,250):
            for deterministic in (True,False):
                for seed in range(3):
                    checkpoint = tmp_path/"baselines"/f"{algorithm}-seed{seed}"/f"epoch-{epoch}.pt"
                    checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    checkpoint.write_bytes(b"actor")
                    native.append(dict(split="validation",config=environment_config("validation"),
                        policy="native_learned",parameters=dict(checkpoint=str(checkpoint),deterministic=deterministic),
                        episodes=[dict(seed=i,success=epoch == 100 and not deterministic,cost=0,reward=1)
                                  for i in range(100,150)]))
    reports["native-baseline-checkpoints-development-v1"] = native
    for run,step,successes in [("camera-balanced-v2",600,20),("camera-recovery-v5",1000,40)]:
        reports[run+"/development"] = []
        for seed in range(3):
            adapter = tmp_path/run/"snapshots"/f"seed{seed}-through{step}"
            adapter.mkdir(parents=True)
            (adapter/"adapter_model.safetensors").write_bytes(b"adapter")
            (adapter/"snapshot.json").write_text(json.dumps(dict(sha256=hashlib.sha256(b"adapter").hexdigest())))
            reports[run+"/development"].append(dict(split="validation",config=environment_config("validation"),
                parameters=dict(adapter=str(adapter),extra_delay=0,protection=True,lane_action_mask=True),
                episodes=[dict(seed=100+i,success=i < successes,cost=0,reward=1) for i in range(50)]))
    for seed in range(3):
        directory = tmp_path/"skills-1m-v1"/f"skills-seed{seed}"
        directory.mkdir(parents=True)
        for filename in ("ppo.zip","cloned.zip"):
            (directory/filename).write_bytes(b"tracker")
    (tmp_path/"camera-config-parity.json").write_text('{"matched":true}')
    monkeypatch.setattr(campaign,"read_reports",lambda root,run: reports[run])
    plan = campaign.freeze_final(tmp_path,"final")
    assert plan["camera_selection"]["run"] == "camera-recovery-v5"
    assert all(row["epoch"] == 100 and not row["deterministic"]
               for row in plan["baseline_selection"]["selection"].values())
    assert len(plan["cpu_specs"]) == 20
    assert len(plan["camera_specs"]) == 36
    assert len(plan["groups"]["Camera-delay0"]) == 3
    assert all(set(spec[4]) <= set(range(500,550)) for spec in plan["cpu_specs"]+plan["camera_specs"])
    assert not (tmp_path/"final"/"episodes").exists()
    monkeypatch.setattr(campaign,"read_reports",lambda *_: pytest.fail("A frozen plan must not reselect models"))
    assert campaign.freeze_final(tmp_path,"final")["plan_sha256"] == plan["plan_sha256"]
    checkpoint = next(iter(plan["artifact_sha256"]))
    from pathlib import Path
    Path(checkpoint).write_bytes(b"changed")
    with pytest.raises(ValueError,match="Frozen artifact changed"):
        campaign.freeze_final(tmp_path,"final")


def test_camera_family_selection_rejects_test_partition(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"read_reports",lambda *_: [dict(split="test")])
    with pytest.raises(ValueError,match="development maps only"):
        campaign.camera_development(tmp_path,"camera-balanced-v2",600)
