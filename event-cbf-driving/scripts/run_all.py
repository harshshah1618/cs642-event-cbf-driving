#!/usr/bin/env python3
"""Train skills then evaluate the full stack (one-shot)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_skills import main as train_main
from scripts.eval_stack import main as eval_main


def main():
    rc = train_main([])
    if rc:
        return rc
    return eval_main([])


if __name__ == "__main__":
    raise SystemExit(main())
