"""Tests for user-activity fairness summaries."""

import math

from src.evaluation.fairness import (
    disparity_max_min_ratio,
    disparity_ratios_by_metric,
    disparity_ratio,
    run_activity_fairness_audit,
    summarize_cold_start_gap,
    summarize_metrics_by_train_activity,
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


def test_summarize_metrics_by_train_activity_handles_optional_columns():
    rows = [
        {
            "user": "a",
            "n_train": 2,
            "precision": 0.05,
            "recall": 0.10,
            "ndcg": 0.02,
            "hit_rate": 0.0,
            "avg_recommended_popularity": 0.9,
            "catalog_coverage": 0.1,
        },
        {
            "user": "b",
            "n_train": 25,
            "precision": 0.20,
            "recall": 0.30,
            "ndcg": 0.15,
            "hit_rate": 1.0,
            "avg_recommended_popularity": 0.4,
            "catalog_coverage": 0.1,
        },
    ]
    summary = summarize_metrics_by_train_activity(rows, n_buckets=2)
    assert "mean_hit_rate" in summary.columns
    assert "mean_recommended_popularity" in summary.columns
    assert "mean_catalog_coverage" in summary.columns


def test_cold_start_gap_and_fairness_audit_return_expected_fields():
    rows = [
        {"user": "u1", "n_train": 3, "precision": 0.0, "recall": 0.0, "ndcg": 0.0},
        {"user": "u2", "n_train": 5, "precision": 0.1, "recall": 0.2, "ndcg": 0.05},
        {"user": "u3", "n_train": 25, "precision": 0.2, "recall": 0.3, "ndcg": 0.15},
        {"user": "u4", "n_train": 40, "precision": 0.3, "recall": 0.4, "ndcg": 0.20},
    ]
    cold_start = summarize_cold_start_gap(rows, cold_max_train=5, warm_min_train=20)
    assert cold_start["n_cold_users"] == 2.0
    assert cold_start["n_warm_users"] == 2.0
    assert cold_start["cold_warm_ndcg_gap"] > 0.0

    audit = run_activity_fairness_audit(rows, n_buckets=2, cold_max_train=5, warm_min_train=20)
    assert "summary" in audit
    assert "disparity" in audit
    assert "cold_start" in audit
    assert audit["cold_start"]["cold_warm_ndcg_gap"] > 0.0
