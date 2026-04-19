"""Rating-prediction evaluation wrappers."""

from __future__ import annotations

from .metrics import evaluate_rating_prediction, mae, rmse

__all__ = [
    "mae",
    "rmse",
    "evaluate_rating_prediction",
]
