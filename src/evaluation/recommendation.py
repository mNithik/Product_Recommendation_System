"""Recommendation and Top-K evaluation wrappers."""

from __future__ import annotations

from .metrics import (
    compute_candidate_hit_rate,
    dcg_at_k,
    evaluate_recommendations,
    evaluate_recommendations_per_user,
    f_measure,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f_measure",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_recommendations",
    "evaluate_recommendations_per_user",
    "compute_candidate_hit_rate",
]
