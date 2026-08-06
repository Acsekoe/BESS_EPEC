"""Compatibility import for the isolated Tikhonov/relaxed-KKT formulation."""

from tikhonov_kkt.old.kkt_formulation import (
    COMPLEMENTARITY_FORMULATIONS,
    DEFAULT_COMPLEMENTARITY_EPSILON,
    DEFAULT_COMPLEMENTARITY_SHIFT,
    MODEL_NAME,
    build_single_investor_relaxed_kkt_mpec,
    relaxed_kkt_diagnostics,
)

__all__ = [
    "COMPLEMENTARITY_FORMULATIONS",
    "DEFAULT_COMPLEMENTARITY_EPSILON",
    "DEFAULT_COMPLEMENTARITY_SHIFT",
    "MODEL_NAME",
    "build_single_investor_relaxed_kkt_mpec",
    "relaxed_kkt_diagnostics",
]
