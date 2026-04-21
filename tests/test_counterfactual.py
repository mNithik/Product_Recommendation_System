"""Tests for approximate counterfactual explanations."""

from src.explainability import (
    ItemSimilarityIndex,
    build_counterfactual_explanation,
    explain_recommendation,
)
from src.pipeline import RankingResult, RankedItem


def _sample_rows():
    return [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U4", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A3", "overall": 4.0},
    ]


def test_counterfactual_has_expected_schema():
    index = ItemSimilarityIndex(_sample_rows())
    explanation = explain_recommendation(index, "U1", "A3")
    ranking = RankingResult(
        user_id="U1",
        items=[
            RankedItem(item_id="A3", rank=1, score=0.91),
            RankedItem(item_id="A4", rank=2, score=0.73),
            RankedItem(item_id="A5", rank=3, score=0.52),
        ],
    )

    counterfactual = build_counterfactual_explanation(explanation, ranking)

    assert counterfactual.user_id == "U1"
    assert counterfactual.recommended_item == "A3"
    assert counterfactual.current_rank == 1
    assert counterfactual.next_item == "A4"
    assert counterfactual.estimated_margin is not None
    assert counterfactual.minimal_change_text
    assert 0.0 <= counterfactual.confidence <= 1.0


def test_counterfactual_actions_are_returned_when_support_exists():
    index = ItemSimilarityIndex(_sample_rows())
    explanation = explain_recommendation(index, "U1", "A3")
    ranking = RankingResult(
        user_id="U1",
        items=[
            RankedItem(item_id="A3", rank=1, score=0.85),
            RankedItem(item_id="A9", rank=2, score=0.80),
        ],
    )

    counterfactual = build_counterfactual_explanation(explanation, ranking)

    assert counterfactual.weakening_actions
    assert counterfactual.weakening_actions[0].action == "remove_or_weaken_positive_interaction"
    assert counterfactual.weakening_actions[0].estimated_impact >= 0.0
