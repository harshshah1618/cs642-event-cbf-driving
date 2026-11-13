"""Render a recorded episode's three-layer interaction figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from driving.plotting_actions import plot_interactions
from driving.runtime import load_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = load_episode(args.episode)
    plot_interactions(episode.result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
