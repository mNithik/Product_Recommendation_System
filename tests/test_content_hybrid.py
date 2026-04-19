"""Tests for the optional content-aware hybrid wrapper."""

from src.hybrid import ContentHybridConfig, ContentHybridRanker


class _BaseRanker:
    def __init__(self):
        self.user_idx = {"U1": 0}
        self.item_idx = {"A1": 0, "A2": 1, "A3": 2, "A4": 3}
        self.rev_user = {0: "U1"}
        self.rev_item = {0: "A1", 1: "A2", 2: "A3", 3: "A4"}

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        exclude = set(exclude_items or [])
        pool = ["A2", "A3", "A4"]
        return [item for item in pool if item not in exclude][:n]


class _TextIndex:
    def cosine_user_item(self, train_records, user_id: str, item_asin: str) -> float:
        scores = {
            ("U1", "A2"): 0.1,
            ("U1", "A3"): 0.9,
            ("U1", "A4"): 0.4,
            ("UNKNOWN", "A2"): 0.2,
            ("UNKNOWN", "A3"): 0.8,
            ("UNKNOWN", "A4"): 0.3,
        }
        return scores.get((user_id, item_asin), 0.0)


def test_content_hybrid_reorders_pool_for_known_user():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A4", "overall": 4.0},
    ]
    wrapper = ContentHybridRanker(
        _BaseRanker(),
        train_data,
        _TextIndex(),
        config=ContentHybridConfig(enabled=True, alpha=0.0, cold_start_alpha=0.0, pool_size=3),
    )

    recs = wrapper.recommend_top_n("U1", n=3, exclude_items={"A1"})

    assert recs[0] == "A3"


def test_content_hybrid_falls_back_to_popularity_for_unknown_user():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A3", "overall": 4.0},
    ]
    wrapper = ContentHybridRanker(
        _BaseRanker(),
        train_data,
        _TextIndex(),
        config=ContentHybridConfig(enabled=True, alpha=0.7, cold_start_alpha=0.3, pool_size=3),
    )

    recs = wrapper.recommend_top_n("UNKNOWN", n=2)

    assert recs[0] == "A2"
