"""Cold-start and sparse-user benchmarking helpers."""

from __future__ import annotations

import pandas as pd

from .metrics import evaluate_recommendations_per_user


def summarize_user_regimes(
    per_user_rows: list[dict],
    *,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
) -> dict[str, float]:
    """
    Summarize recommendation quality for sparse and warm user regimes.

    Returns headline metrics that are easy to compare across models.
    """
    if not per_user_rows:
        return {
            "n_cold_users": 0.0,
            "n_warm_users": 0.0,
            "cold_mean_precision": 0.0,
            "warm_mean_precision": 0.0,
            "cold_mean_recall": 0.0,
            "warm_mean_recall": 0.0,
            "cold_mean_ndcg": 0.0,
            "warm_mean_ndcg": 0.0,
            "cold_warm_precision_gap": 0.0,
            "cold_warm_recall_gap": 0.0,
            "cold_warm_ndcg_gap": 0.0,
        }

    df = pd.DataFrame(per_user_rows)
    cold = df[df["n_train"] <= cold_max_train]
    warm = df[df["n_train"] >= warm_min_train]

    def _mean(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(frame[column].mean())

    cold_precision = _mean(cold, "precision")
    warm_precision = _mean(warm, "precision")
    cold_recall = _mean(cold, "recall")
    warm_recall = _mean(warm, "recall")
    cold_ndcg = _mean(cold, "ndcg")
    warm_ndcg = _mean(warm, "ndcg")

    return {
        "n_cold_users": float(len(cold)),
        "n_warm_users": float(len(warm)),
        "cold_mean_precision": cold_precision,
        "warm_mean_precision": warm_precision,
        "cold_mean_recall": cold_recall,
        "warm_mean_recall": warm_recall,
        "cold_mean_ndcg": cold_ndcg,
        "warm_mean_ndcg": warm_ndcg,
        "cold_warm_precision_gap": warm_precision - cold_precision,
        "cold_warm_recall_gap": warm_recall - cold_recall,
        "cold_warm_ndcg_gap": warm_ndcg - cold_ndcg,
    }


def evaluate_cold_start_benchmark(
    model,
    train_data: list[dict],
    test_data: list[dict],
    *,
    top_n: int = 10,
    batch_size: int = 512,
    min_train_ratings: int = 1,
    max_candidates: int = 10000,
    relevance_threshold=None,
    min_item_ratings: int = 0,
    max_users: int | None = None,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
) -> dict[str, float]:
    """Run the standard recommendation protocol and summarize sparse-user performance."""
    per_user_rows = evaluate_recommendations_per_user(
        model,
        train_data,
        test_data,
        top_n=top_n,
        batch_size=batch_size,
        min_train_ratings=min_train_ratings,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
        min_item_ratings=min_item_ratings,
        max_users=max_users,
    )
    return summarize_user_regimes(
        per_user_rows,
        cold_max_train=cold_max_train,
        warm_min_train=warm_min_train,
    )


def compare_cold_start_benchmarks(
    baseline_metrics: dict[str, float],
    comparison_metrics: dict[str, float],
    *,
    prefix: str = "comparison",
) -> dict[str, float]:
    """
    Compare two cold-start benchmark summaries.

    Useful for reporting whether a hybrid branch helps sparse users relative to a
    baseline ranker.
    """
    return {
        f"{prefix}_cold_precision_delta": float(
            comparison_metrics.get("cold_mean_precision", 0.0) - baseline_metrics.get("cold_mean_precision", 0.0)
        ),
        f"{prefix}_cold_recall_delta": float(
            comparison_metrics.get("cold_mean_recall", 0.0) - baseline_metrics.get("cold_mean_recall", 0.0)
        ),
        f"{prefix}_cold_ndcg_delta": float(
            comparison_metrics.get("cold_mean_ndcg", 0.0) - baseline_metrics.get("cold_mean_ndcg", 0.0)
        ),
        f"{prefix}_warm_ndcg_delta": float(
            comparison_metrics.get("warm_mean_ndcg", 0.0) - baseline_metrics.get("warm_mean_ndcg", 0.0)
        ),
    }
