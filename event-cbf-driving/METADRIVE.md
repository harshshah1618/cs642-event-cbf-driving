# Physical simulation with MetaDrive

MetaDrive/Bullet moves the car and traffic under vehicle physics. The maintained
adapter is `event_cbf_driving/env/metadrive_adapter.py`.

## Setup

```bash
python -m pip install -e ".[dev,metadrive]"
python -m metadrive.pull_asset
make metadrive-check
```

The dependency is pinned in `pyproject.toml` to MetaDrive commit
`fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`. Linux headless operation needs working
OpenGL/EGL libraries; CI installs `libgl1`, `libegl1` and `libglib2.0-0`.

For a small evaluation:

```bash
event-cbf eval --profile metadrive-study --modes proposed --seeds 0 1 2   --no-figures --run-id physical-smoke
```

See [the final study](../final-study/README.md) and
[script index](scripts/README.md) for the integrated comparison and actual camera
model. The trained adapter and pinned prompt are under `checkpoints/vlm/`;
large base weights are external.

## Control and camera path

The learned skill requests road-frame forward and lateral acceleration. The
adapter converts that vector into steering and throttle/brake commands, applies
it to the physical simulator, and measures the resulting velocity change.
Requested and achieved acceleration can differ because actuators saturate and
the vehicle has its own dynamics. Logs retain both values.

Simulation control ticks are 0.02 seconds apart. That rate is not a wall-clock
latency guarantee. In the asynchronous model path, inference runs in a worker
while the controller keeps receiving fresh simulator state and filtering actions.
Camera capture runs on the simulator thread; detached image bytes go to inference.
Late replies are rejected and current lane gaps are checked before switching.

The default driver and safety layer still receive simulator state. The model's
camera input does not make the complete system camera-only. Straight-road lane
checks and the braking backup have assumptions; see [safety](SAFETY.md).

## Inspection tools

- `scripts/record_cameras.py`: capture simulator camera observations.
- `scripts/plot_interactions.py`: plot the recorded action chain.
- `scripts/render_demo.py`: render an episode as a video.

Each tool provides `--help`. Earlier prototypes, stress sweeps and reports are
recoverable from the pre-cleanup snapshot described in [history](../docs/history.md).
