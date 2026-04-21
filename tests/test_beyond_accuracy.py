"""Tests for beyond-accuracy recommendation metrics."""

from src.evaluation.recommendation import evaluate_beyond_accuracy


class _ToyRanker:
    def __init__(self):
        self.user_idx = {"U1": 0, "U2": 1}

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        exclude = set(exclude_items or [])
        candidates_by_user = {
            "U1": ["A3", "A4", "A5"],
            "U2": ["A4", "A5", "A6"],
        }
        return [item for item in candidates_by_user[user_id] if item not in exclude][:n]


def test_evaluate_beyond_accuracy_returns_expected_metric_keys():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A4", "overall": 4.0},
        {"reviewerID": "U4", "asin": "A4", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A5", "overall": 4.0},
    ]
    test_data = [
        {"reviewerID": "U1", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A4", "overall": 5.0},
    ]

    metrics = evaluate_beyond_accuracy(
        _ToyRanker(),
        train_data,
        test_data,
        top_n=2,
        min_train_ratings=1,
        relevance_threshold=4.0,
        max_users=10,
    )

    assert metrics["CatalogCoverage"] > 0.0
    assert 0.0 <= metrics["Diversity"] <= 1.0
    assert metrics["Novelty"] >= 0.0
    assert 0.0 <= metrics["PopularityConcentration"] <= 1.0
    assert metrics["n_users_beyond_accuracy_eval"] == 2.0
