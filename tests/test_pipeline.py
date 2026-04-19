"""Tests for explicit ranking and recommendation pipeline stages."""

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
