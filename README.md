# CS642 — Event-CBF Driving

Course project sketch: a 3-tier driving stack (slow skill router → fast skills → CBF safety filter) with a pending-call margin while the router is in flight.

This zip is a laptop-ready snapshot of the working code, the course proposal, and the planning notes. Start with the tiny demo below.

## Quick start: `wip-minimal`

Standalone Two-Trap Gauntlet demo. No package install beyond the three deps. CPU is fine.

```bash
cd wip-minimal
pip install -r requirements.txt
python run_minimal.py
```

Usually finishes in a few seconds (under 2 min worst case). A rollout figure is already in `media/wip-minimal-rollout.png`.

## What's in the zip

| Path | What it is |
| --- | --- |
| `wip-minimal/` | Smallest runnable demo — start here |
| `event-cbf-driving/` | Fuller package (envs, router, shield, eval scripts) |
| `two-trap-gauntlet/` | Earlier scripted gauntlet / training sketch |
| `proposal/` | Professor proposal (LaTeX + PDF) |
| `docs/` | Project plan, method/novelty notes, source feasibility |
| `media/` | Demo rollout figure |

## Notes

- Excludes caches, egg-info, and project-store internals.
- Checkpoints (`.pt`) are included so demos can run without retraining.
- `wip-minimal` does not import `event-cbf-driving`; they are separate trees on purpose.
