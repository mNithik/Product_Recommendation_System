"""Unit tests for evaluation metrics."""

import math

import pytest

from src.evaluation.metrics import (
    dcg_at_k,
    f_measure,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


class TestRatingMetrics:
    def test_mae_perfect(self):
        assert mae([3, 4, 5], [3, 4, 5]) == 0.0

    def test_mae_known(self):
        assert mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)

    def test_rmse_perfect(self):
        assert rmse([3, 4, 5], [3, 4, 5]) == 0.0

    def test_rmse_known(self):
        assert rmse([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)

    def test_rmse_penalizes_outliers(self):
        """RMSE should be >= MAE for the same predictions."""
        preds = [1, 1, 1, 5]
        actuals = [2, 2, 2, 2]
        assert rmse(preds, actuals) >= mae(preds, actuals)


class TestTopNMetrics:
    def test_precision_at_k_perfect(self):
        recommended = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_precision_at_k_none_relevant(self):
        recommended = ["A", "B", "C"]
        relevant = ["D", "E"]
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_precision_at_k_partial(self):
        recommended = ["A", "B", "C", "D", "E"]
        relevant = ["A", "C"]
        assert precision_at_k(recommended, relevant, k=5) == pytest.approx(0.4)

    def test_recall_at_k_perfect(self):
        recommended = ["A", "B", "C"]
        relevant = ["A", "B"]
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        recommended = ["A", "B"]
        relevant = ["A", "B", "C", "D"]
        assert recall_at_k(recommended, relevant, k=2) == pytest.approx(0.5)

    def test_recall_empty_relevant(self):
        assert recall_at_k(["A", "B"], [], k=2) == 0

    def test_f_measure_balanced(self):
        assert f_measure(0.5, 0.5) == pytest.approx(0.5)

    def test_f_measure_zero(self):
        assert f_measure(0, 0) == 0

    def test_ndcg_at_k_perfect(self):
        """All relevant items ranked first should give NDCG = 1."""
        recommended = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_no_hits(self):
        recommended = ["X", "Y", "Z"]
        relevant = ["A", "B"]
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_ndcg_at_k_order_matters(self):
        """Relevant item at position 1 should beat relevant item at position 3."""
        relevant = ["A"]
        ndcg_first = ndcg_at_k(["A", "B", "C"], relevant, k=3)
        ndcg_last = ndcg_at_k(["B", "C", "A"], relevant, k=3)
        assert ndcg_first > ndcg_last

    def test_dcg_at_k_single_hit(self):
        relevant = {"A"}
        recommended = ["A", "B", "C"]
        expected = 1.0 / math.log2(2)
        assert dcg_at_k(relevant, recommended, k=3) == pytest.approx(expected)
