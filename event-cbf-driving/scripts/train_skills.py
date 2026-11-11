#!/usr/bin/env python3
"""Standalone wrapper for :mod:`event_cbf_driving.training`.

The implementation lives in the package so the installed ``event-cbf train``
entry point works without the repository's ``scripts/`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_cbf_driving.training import main

if __name__ == "__main__":
    raise SystemExit(main())
