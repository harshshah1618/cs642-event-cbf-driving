# Event-CBF Driving

A driving simulator project with three parts: a model chooses a maneuver, a
trained driver produces movement commands, and safety checks can override them.
The car keeps driving while a slower decision is pending.

## Start here

- [Two-page milestone report](docs/milestone-report.pdf) ([editable text](docs/milestone-report.md)).
- [What the experiments found](final-study/EXPLAINED.md).
- [Full final report](final-study/final-report.pdf) and [comparison tables](final-study/results/all-conditions.md).
- [Readable PPO code guide](event-cbf-driving/event_cbf_driving/skills/README.md).

The final comparison covered 1,440 controlled simulator drives and 36 camera-model
integration drives. The integrated version finished all 72 short-delay conditions
without crashes, using 98.6% fewer delayed requests than frequent polling.
Those controlled requests used a traffic rule with an imposed wait. Simple rules
also worked well; the tests do not establish that PPO, the language model, or our
particular request timing is essential. The smaller camera-model comparison
produced 22 real replies and does not establish superiority over simple rules.

## Run the code

```bash
cd event-cbf-driving
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

For new single-critic PPO training:

```bash
python -m event_cbf_driving.skills.train_ppo --skill CRUISE_RIGHT --seed 0
python -m event_cbf_driving.skills.train_ppo --skill OVERTAKE_LEFT --seed 1
```

The existing measured drivers remain separate from new training output. See the
[package guide](event-cbf-driving/README.md) for simulation and evaluation, and the
[script index](event-cbf-driving/scripts/README.md) for the retained tools.

## Repository layout

| Folder | Contents |
|---|---|
| `event-cbf-driving/` | Maintained package, tests, configs, checkpoints and scripts. |
| `final-study/` | Recorded final results, report, protocol and audits. |
| `docs/` | Current milestone, its PDF builder, and history/recovery notes. |
| `proposal/` | Original proposal and feedback. |

Older experiment runners, toy projects and duplicate deliveries have been
retired. They remain recoverable from Git and a local archive; see
[history and recovery](docs/history.md). Saved reports describe their original
source versions. New source changes require fresh experiment directories.

The full raw final-study release and base model weights are external to Git.
See [the final study](final-study/README.md) for reproduction requirements.
This is research software; simulation results do not prove real-vehicle safety.
