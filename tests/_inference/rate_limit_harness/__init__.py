"""Hermetic, real-clock rate-limit performance harness.

Re-exports the public harness surface so tests can import from the package root.
"""

from tests._inference.rate_limit_harness.harness import (
    RateLimitReport,
    RateLimitScenario,
    SimulatedCompletionsClient,
    SimulatedServerLimiter,
    constant,
    lognormal,
    regime_shift,
    run_scenario,
)

__all__ = [
    "RateLimitReport",
    "RateLimitScenario",
    "SimulatedCompletionsClient",
    "SimulatedServerLimiter",
    "constant",
    "lognormal",
    "regime_shift",
    "run_scenario",
]
