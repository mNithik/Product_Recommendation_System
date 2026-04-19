"""Tests for explanation, counterfactual, and causal-adjusted evaluation summaries."""

from src.evaluation.explanation import (
    evaluate_counterfactuals,
    evaluate_explainable_recommendations,
    evaluate_explanations,
)
from src.explainability import CounterfactualExplanation, RecommendationExplanation
from src.postprocessing import CausalAdjustmentConfig


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


class _ToyRanker:
    def __init__(self):
        self.user_idx = {"U1": 0}
        self.scores = {"A3": 0.80, "A4": 0.79, "A5": 0.10}

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None):
        exclude = set(exclude_items or [])
        ranked = [item for item in ("A3", "A4", "A5") if item not in exclude]
        return ranked[:n]

    def predict(self, user_id: str, item_id: str) -> float:
        return self.scores[item_id]


def test_evaluate_explainable_recommendations_returns_structural_and_causal_metrics():
    train_data = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U4", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A4", "overall": 4.0},
    ]
    test_data = [
        {"reviewerID": "U1", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A4", "overall": 4.0},
    ]

    metrics = evaluate_explainable_recommendations(
        _ToyRanker(),
        train_data,
        test_data,
        top_n=2,
        min_train_ratings=1,
        relevance_threshold=4.0,
        explanation_sample_users=10,
        causal_config=CausalAdjustmentConfig(enabled=True, support_weight=0.20, popularity_penalty_weight=0.05),
    )

    assert metrics["explanation_coverage"] == 1.0
    assert metrics["counterfactual_coverage"] == 1.0
    assert metrics["n_explanation_users_eval"] == 1.0
    assert "causal_Precision" in metrics
    assert "causal_Recall" in metrics
    assert "causal_NDCG" in metrics
