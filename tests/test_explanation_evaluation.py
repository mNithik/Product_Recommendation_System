"""Tests for explanation and counterfactual evaluation summaries."""

from src.evaluation.explanation import evaluate_counterfactuals, evaluate_explanations
from src.explainability import CounterfactualExplanation, RecommendationExplanation


def _explanation(item_id: str, confidence: float, with_history: bool, with_similar: bool):
    return RecommendationExplanation(
        user_id="U1",
        recommended_item=item_id,
        explanation_text="test",
        support_confidence=confidence,
        supporting_history=[object()] if with_history else [],
        supporting_similar_items=[object()] if with_similar else [],
        explanation_type="test",
    )


def _counterfactual(item_id: str, confidence: float, with_actions: bool, with_margin: bool):
    return CounterfactualExplanation(
        user_id="U1",
        recommended_item=item_id,
        current_rank=1,
        current_score=0.9,
        next_item="B",
        next_item_score=0.8,
        estimated_margin=0.1 if with_margin else None,
        weakening_actions=[object()] if with_actions else [],
        minimal_change_text="test",
        confidence=confidence,
    )


def test_evaluate_explanations_reports_expected_coverages():
    metrics = evaluate_explanations(
        [
            _explanation("A", 0.8, with_history=True, with_similar=False),
            _explanation("B", 0.4, with_history=False, with_similar=True),
        ]
    )

    assert metrics["explanation_coverage"] == 1.0
    assert metrics["history_support_coverage"] == 0.5
    assert metrics["similar_item_support_coverage"] == 0.5
    assert 0.0 <= metrics["mean_support_confidence"] <= 1.0


def test_evaluate_counterfactuals_reports_expected_coverages():
    metrics = evaluate_counterfactuals(
        [
            _counterfactual("A", 0.7, with_actions=True, with_margin=True),
            _counterfactual("B", 0.3, with_actions=False, with_margin=False),
        ]
    )

    assert metrics["counterfactual_coverage"] == 1.0
    assert metrics["weakening_action_coverage"] == 0.5
    assert metrics["margin_availability"] == 0.5
    assert 0.0 <= metrics["mean_counterfactual_confidence"] <= 1.0
