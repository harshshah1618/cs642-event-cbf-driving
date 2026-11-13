"""CARLA adapter skeleton (real CARLA is deliberately not faked).

The CI/evals in this repository use the lightweight highway-compatible
kinematics scenarios.  Real CARLA integration requires *both*:

1. the CARLA Python API (``carla`` package) installed in the same environment,
2. a running CARLA server (``CarlaUE4.sh`` / ``CarlaUE4.exe``) reachable at
   ``host:port``.

:class:`CarlaAdapterStub` documents that boundary and raises an actionable
``NotImplementedError`` instead of pretending to drive.  Integration tests can
use :class:`MockCarlaAdapter`, which wraps any native
:class:`~env_base.DrivingEnv` and forwards the full contract
without requiring a CARLA server.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env_base import validate_action


class CarlaAdapterStub:
    """Skeleton for a future real-CARLA implementation.

    Implement the methods listed in :attr:`REQUIRED_REAL_METHODS` and replace
    :meth:`connect`, :meth:`reset`, :meth:`step`, and :meth:`snapshot` with
    calls into the CARLA Python API.  The adapter must return the same snapshot
    and five-tuple shapes as the native scenarios so ``DrivingStack.run`` works
    unchanged.
    """

    IS_REAL_CARLA = False
    IS_MOCK = False
    SUPPORTS_REAL_CARLA = False
    REQUIRES_SERVER = True
    REQUIRES_CARLA_SERVER = True
    REQUIRES_CARLA_PYTHON_API = True
    CAPABILITIES = {
        "real_carla": False,
        "mock": False,
        "supports_real_carla": False,
        "requires_server": True,
        "requires_carla_server": True,
        "requires_carla_python_api": True,
    }
    capabilities = CAPABILITIES
    REQUIRED_REAL_METHODS = (
        "connect",
        "_spawn_ego",
        "_apply_control",
        "_read_world",
        "_build_snapshot",
    )

    def __init__(self, host: str = "localhost", port: int = 2000, **kwargs: Any):
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError(
            "CarlaAdapterStub.connect() is not implemented. Real CARLA requires "
            "the CARLA Python API and a running CARLA server. Implement "
            "_spawn_ego(), _apply_control(), _read_world(), and _build_snapshot(); "
            "then connect() should create carla.Client(host, port), load the world, "
            "spawn the ego via _spawn_ego(), and set _connected=True. "
            "Use MockCarlaAdapter(native_env) for integration tests that do not "
            "need a CARLA server."
        )

    def _require_connected(self) -> None:
        if not self._connected:
            self.connect()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.connect()

    def step(self, action):
        self.connect()
        validate_action(action)

    def snapshot(self):
        self.connect()

    def close(self) -> None:
        self._connected = False


class MockCarlaAdapter(CarlaAdapterStub):
    """Wrap a native environment behind the CARLA adapter surface.

    This is an integration-test double, not a CARLA simulator: it forwards
    ``reset``/``step``/``snapshot``/``close`` to the wrapped native environment
    and keeps the CARLA capability flags explicit.
    """

    IS_REAL_CARLA = False
    IS_MOCK = True
    SUPPORTS_REAL_CARLA = False
    REQUIRES_SERVER = False
    REQUIRES_CARLA_SERVER = False
    REQUIRES_CARLA_PYTHON_API = False
    CAPABILITIES = {
        "real_carla": False,
        "mock": True,
        "supports_real_carla": False,
        "requires_server": False,
        "requires_carla_server": False,
        "requires_carla_python_api": False,
    }
    capabilities = CAPABILITIES

    def __init__(
        self,
        env: Any = None,
        *args: Any,
        native_env: Any = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if env is None:
            env = native_env
        if env is None:
            raise TypeError("MockCarlaAdapter requires a native env")
        if not all(hasattr(env, method) for method in ("reset", "step", "snapshot")):
            raise TypeError(
                "MockCarlaAdapter requires an environment implementing "
                "reset(), step(), and snapshot()"
            )
        self.env = env
        self.native_env = env

    def connect(self) -> None:
        self._connected = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.connect()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self.connect()
        return self.env.step(validate_action(action))

    def snapshot(self):
        self.connect()
        return self.env.snapshot()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self._connected = False
