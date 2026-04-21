"""Tests for lightweight causal score adjustment."""

from src.explainability import RecommendationExplanation
from src.pipeline import RankedItem, RankingResult
from src.postprocessing import CausalAdjustmentConfig, apply_causal_adjustment


def _explanation(item_id: str, confidence: float) -> RecommendationExplanation:
    return RecommendationExplanation(
        user_id="U1",
        recommended_item=item_id,
        explanation_text="test",
        support_confidence=confidence,
        supporting_history=[],
        supporting_similar_items=[],
        explanation_type="test",
    )


def test_causal_adjustment_toggle_off_returns_original_ranking():
    ranking = RankingResult(
        user_id="U1",
        items=[
            RankedItem(item_id="A", rank=1, score=0.9),
            RankedItem(item_id="B", rank=2, score=0.8),
        ],
    )
    explanations = {"A": _explanation("A", 0.9), "B": _explanation("B", 0.1)}

    adjusted = apply_causal_adjustment(
        ranking,
        explanations,
        item_popularity={"A": 10, "B": 1},
        config=CausalAdjustmentConfig(enabled=False),
    )

    assert adjusted is ranking


def test_causal_adjustment_updates_scores_and_can_change_order():
    ranking = RankingResult(
        user_id="U1",
        items=[
            RankedItem(item_id="A", rank=1, score=0.81),
            RankedItem(item_id="B", rank=2, score=0.80),
        ],
    )
    explanations = {"A": _explanation("A", 0.0), "B": _explanation("B", 1.0)}

    adjusted = apply_causal_adjustment(
        ranking,
        explanations,
        item_popularity={"A": 10, "B": 1},
        config=CausalAdjustmentConfig(enabled=True, support_weight=0.20, popularity_penalty_weight=0.05),
    )

    assert adjusted is not ranking
    assert adjusted.metadata["causal_adjustment_enabled"] is True
    assert adjusted.items[0].item_id == "B"
    assert adjusted.items[0].metadata["adjusted_score"] >= adjusted.items[1].metadata["adjusted_score"]
    assert adjusted.items[0].metadata["used_score_fallback"] is False


def test_causal_adjustment_marks_rank_fallback_when_raw_score_missing():
    ranking = RankingResult(
        user_id="U1",
        items=[
            RankedItem(item_id="A", rank=1, score=None),
            RankedItem(item_id="B", rank=2, score=None),
        ],
    )
    explanations = {"A": _explanation("A", 0.4), "B": _explanation("B", 0.2)}

    adjusted = apply_causal_adjustment(
        ranking,
        explanations,
        item_popularity={"A": 5, "B": 2},
        config=CausalAdjustmentConfig(enabled=True),
    )

    assert adjusted.items[0].metadata["used_score_fallback"] is True
    assert adjusted.items[0].metadata["effective_base_score_source"] == "rank_fallback"
