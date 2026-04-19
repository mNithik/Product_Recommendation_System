"""Tests for explicit ranking and recommendation pipeline stages."""

from src.models.popularity import PopularityBaseline
from src.pipeline import rank_items_for_user, rank_items_for_users, recommend_from_ranking


class _FakeSingleUserModel:
    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        del user_id, exclude_items
        return ["I3", "I1", "I2"][:n]


class _FakeBatchModel:
    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, **kwargs):
        del exclude_sets, kwargs
        out = []
        for u in user_indices:
            out.append([f"I{u}A", f"I{u}B", f"I{u}C"][:n])
        return out


class _FakeLatentBackend:
    def __init__(self):
        self.user_factors = [[1.0, 0.0], [0.0, 1.0]]
        self.item_factors = [[0.9, 0.1], [0.1, 0.8], [0.5, 0.5]]


class _FakeImplicitModel:
    def __init__(self):
        self.user_idx = {"U1": 0, "U2": 1}
        self.item_idx = {"A": 0, "B": 1, "C": 2}
        self.model = _FakeLatentBackend()

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        del exclude_items
        order = {"U1": ["A", "C", "B"], "U2": ["B", "C", "A"]}
        return order[user_id][:n]


def test_recommendation_is_derived_from_ranking_for_single_user():
    model = _FakeSingleUserModel()
    ranking = rank_items_for_user(model, user_id="U1", n_candidates=3)

    assert [row.item_id for row in ranking.items] == ["I3", "I1", "I2"]
    assert [row.rank for row in ranking.items] == [1, 2, 3]

    recommendation = recommend_from_ranking(ranking, top_n=2)
    assert recommendation.recommended_items == ["I3", "I1"]
    assert recommendation.metadata["derived_from_ranking"] is True


def test_batch_ranking_wraps_existing_batch_recommendations():
    model = _FakeBatchModel()
    rankings = rank_items_for_users(
        model,
        user_ids=["U10", "U11"],
        user_indices=[10, 11],
        exclude_sets=[set(), set()],
        n_candidates=2,
    )

    assert len(rankings) == 2
    assert [row.item_id for row in rankings[0].items] == ["I10A", "I10B"]
    assert [row.item_id for row in rankings[1].items] == ["I11A", "I11B"]


def test_popularity_ranking_carries_item_count_scores():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 3.0},
        {"reviewerID": "U3", "asin": "A1", "overall": 2.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 5.0},
    ]
    model = PopularityBaseline()
    model.fit(train_data)

    ranking = rank_items_for_user(model, user_id="U1", n_candidates=1)

    assert ranking.metadata["score_source"] == "item_counts"
    assert ranking.items[0].score is not None
    assert ranking.items[0].item_id == "A3"
    assert ranking.items[0].score == 1.0


def test_implicit_style_ranking_carries_latent_scores():
    model = _FakeImplicitModel()
    ranking = rank_items_for_user(model, user_id="U1", n_candidates=3)

    assert ranking.metadata["score_source"] == "latent_dot"
    assert [row.item_id for row in ranking.items] == ["A", "C", "B"]
    assert ranking.items[0].score == 0.9
