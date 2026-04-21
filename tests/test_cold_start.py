"""Tests for cold-start benchmarking helpers."""

from src.evaluation.cold_start import (
    compare_cold_start_benchmarks,
    evaluate_cold_start_benchmark,
    summarize_user_regimes,
)


class _ToyRanker:
    def __init__(self):
        self.user_idx = {"U1": 0, "U2": 1, "U3": 2}

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        exclude = set(exclude_items or [])
        pool = {
            "U1": ["A3", "A4", "A5"],
            "U2": ["A4", "A5", "A6"],
            "U3": ["A5", "A6", "A7"],
        }[user_id]
        return [item for item in pool if item not in exclude][:n]


def test_summarize_user_regimes_returns_expected_fields():
    rows = [
        {"user": "U1", "n_train": 2, "precision": 0.0, "recall": 0.0, "ndcg": 0.0},
        {"user": "U2", "n_train": 4, "precision": 0.1, "recall": 0.2, "ndcg": 0.05},
        {"user": "U3", "n_train": 25, "precision": 0.3, "recall": 0.4, "ndcg": 0.20},
    ]
    metrics = summarize_user_regimes(rows, cold_max_train=5, warm_min_train=20)

    assert metrics["n_cold_users"] == 2.0
    assert metrics["n_warm_users"] == 1.0
    assert metrics["cold_warm_ndcg_gap"] >= 0.0


def test_summarize_user_regimes_gpu_flag_falls_back_safely():
    rows = [
        {"user": "U1", "n_train": 2, "precision": 0.0, "recall": 0.0, "ndcg": 0.0},
        {"user": "U2", "n_train": 4, "precision": 0.1, "recall": 0.2, "ndcg": 0.05},
        {"user": "U3", "n_train": 25, "precision": 0.3, "recall": 0.4, "ndcg": 0.20},
    ]
    metrics = summarize_user_regimes(rows, cold_max_train=5, warm_min_train=20, use_gpu=True)

    assert metrics["n_cold_users"] == 2.0
    assert metrics["n_warm_users"] == 1.0
    assert "cold_mean_ndcg" in metrics


def test_compare_cold_start_benchmarks_returns_deltas():
    baseline = {"cold_mean_ndcg": 0.05, "warm_mean_ndcg": 0.10, "cold_mean_precision": 0.02, "cold_mean_recall": 0.04}
    hybrid = {"cold_mean_ndcg": 0.08, "warm_mean_ndcg": 0.11, "cold_mean_precision": 0.03, "cold_mean_recall": 0.05}
    delta = compare_cold_start_benchmarks(baseline, hybrid, prefix="hybrid_vs_base")

    assert delta["hybrid_vs_base_cold_ndcg_delta"] > 0.0
    assert delta["hybrid_vs_base_warm_ndcg_delta"] > 0.0


def test_evaluate_cold_start_benchmark_runs_end_to_end():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A4", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A5", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A6", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A7", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A8", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A9", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A10", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A11", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A12", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A13", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A14", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A15", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A16", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A17", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A18", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A19", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A20", "overall": 5.0},
    ]
    test_data = [
        {"reviewerID": "U1", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A4", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A5", "overall": 5.0},
    ]

    metrics = evaluate_cold_start_benchmark(
        _ToyRanker(),
        train_data,
        test_data,
        top_n=2,
        min_train_ratings=1,
        relevance_threshold=4.0,
        max_users=10,
        cold_max_train=2,
        warm_min_train=10,
    )

    assert metrics["n_cold_users"] >= 1.0
    assert metrics["n_warm_users"] >= 1.0
