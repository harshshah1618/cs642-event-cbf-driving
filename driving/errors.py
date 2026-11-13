"""Typed exceptions and stable process exit codes.

The CLI maps every exception to a documented exit code so automation can
distinguish a bad config from a safety regression or a router outage.
"""

from __future__ import annotations


class EventCBFError(Exception):
    """Base class for expected, user-actionable failures."""

    exit_code = 1


class ConfigError(EventCBFError):
    """Invalid or inconsistent configuration."""

    exit_code = 2


class ValidationError(EventCBFError):
    """A runtime object failed an invariant check."""

    exit_code = 2


class CheckpointError(EventCBFError):
    """Missing, corrupt, or incompatible checkpoint."""

    exit_code = 3


class RouterError(EventCBFError):
    """Router backend failed or violated its contract."""

    exit_code = 4


class SafetyError(EventCBFError):
    """A safety invariant was violated and the run was stopped."""

    exit_code = 5


class EvaluationFailure(EventCBFError):
    """Evaluation completed but the configured pass bar was not met."""

    exit_code = 6


class ArtifactError(EventCBFError):
    """An output artifact could not be read or written safely."""

    exit_code = 7
