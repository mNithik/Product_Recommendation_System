"""Tests for user-activity fairness summaries."""

import math

from src.evaluation.fairness import (
    disparity_max_min_ratio,
    disparity_ratios_by_metric,
    disparity_ratio,
    summarize_ndcg_by_train_activity,
)


def test_summarize_buckets_and_ratio():
    rows = [
        {"user": "a", "n_train": 5, "precision": 0.1, "recall": 0.2, "ndcg": 0.05},
        {"user": "b", "n_train": 6, "precision": 0.1, "recall": 0.2, "ndcg": 0.06},
        {"user": "c", "n_train": 50, "precision": 0.2, "recall": 0.3, "ndcg": 0.20},
        {"user": "d", "n_train": 60, "precision": 0.2, "recall": 0.3, "ndcg": 0.10},
    ]
    summary = summarize_ndcg_by_train_activity(rows, n_buckets=2)
    assert not summary.empty
    assert "std_ndcg" in summary.columns
    assert "median_precision" in summary.columns
    r = disparity_ratio(rows, n_buckets=2)
    assert r > 1.0
    ratios = disparity_ratios_by_metric(rows, n_buckets=2)
    assert not math.isnan(ratios["ndcg"])
    assert ratios["ndcg"] > 1.0


def test_disparity_max_min_ratio_empty():
    import pandas as pd

    assert math.isnan(disparity_max_min_ratio(pd.DataFrame(), "mean_ndcg"))
