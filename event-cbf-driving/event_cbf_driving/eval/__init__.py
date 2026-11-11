from .metrics import (
    ablation_table,
    aggregate_rollouts,
    json_safe,
    result_to_episode_record,
    result_to_row,
    summarize,
)
from .reporting import aggregate_by_mode, aggregate_episodes, render_markdown
from .runner import (
    MAX_PROPOSED_ROUTER_CALLS,
    SCHEMA_VERSION,
    EvaluationResult,
    check_regressions,
    load_skill_bank,
    run_episode,
    run_evaluation,
)

try:  # matplotlib is an existing dependency, but imports must not hard-fail.
    from .figures import plot_ablation_bars, plot_gauntlet_rollouts, plot_training_curves
except Exception:  # pragma: no cover - optional plotting dependency

    def _missing_figure(*_args, **_kwargs):
        raise ImportError("matplotlib is required for evaluation figures")

    plot_training_curves = _missing_figure
    plot_gauntlet_rollouts = _missing_figure
    plot_ablation_bars = _missing_figure

__all__ = [
    "MAX_PROPOSED_ROUTER_CALLS",
    "SCHEMA_VERSION",
    "EvaluationResult",
    "ablation_table",
    "aggregate_by_mode",
    "aggregate_episodes",
    "aggregate_rollouts",
    "check_regressions",
    "json_safe",
    "load_skill_bank",
    "plot_ablation_bars",
    "plot_gauntlet_rollouts",
    "plot_training_curves",
    "render_markdown",
    "result_to_episode_record",
    "result_to_row",
    "run_episode",
    "run_evaluation",
    "summarize",
]
