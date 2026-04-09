"""Integration tests for recommendation output shapes and constraints."""

import pytest

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.utils.data_loader import build_index


@pytest.fixture
def sample_data():
    return [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U1", "asin": "A3", "overall": 3.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A4", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A2", "overall": 2.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A5", "overall": 4.0},
    ]


class TestBuildIndex:
    def test_all_users_indexed(self, sample_data):
        users, items, rev_users, rev_items = build_index(sample_data)
        assert len(users) == 3
        assert len(items) == 5

    def test_reverse_mapping(self, sample_data):
        users, items, rev_users, rev_items = build_index(sample_data)
        for uid, idx in users.items():
            assert rev_users[idx] == uid
        for iid, idx in items.items():
            assert rev_items[idx] == iid

    def test_indices_are_contiguous(self, sample_data):
        users, items, _, _ = build_index(sample_data)
        assert set(users.values()) == set(range(len(users)))
        assert set(items.values()) == set(range(len(items)))


class TestTopNOutputConstraints:
    """Verify that metric functions handle edge cases correctly."""

    def test_precision_never_exceeds_one(self):
        for _ in range(50):
            rec = ["A", "B", "C", "D", "E"]
            rel = ["A", "B", "C", "D", "E", "F", "G"]
            p = precision_at_k(rec, rel, k=5)
            assert 0 <= p <= 1.0

    def test_recall_never_exceeds_one(self):
        rec = ["A", "B", "C"]
        rel = ["A", "B"]
        r = recall_at_k(rec, rel, k=3)
        assert 0 <= r <= 1.0

    def test_ndcg_bounded(self):
        rec = ["A", "B", "C"]
        rel = ["A"]
        n = ndcg_at_k(rec, rel, k=3)
        assert 0 <= n <= 1.0

    def test_empty_recommendation(self):
        assert precision_at_k([], ["A"], k=10) == 0
        assert recall_at_k([], ["A"], k=10) == 0
        assert ndcg_at_k([], ["A"], k=10) == 0
