"""Evaluation helpers for explanation and counterfactual layers."""

from __future__ import annotations

from collections.abc import Sequence

from src.explainability import CounterfactualExplanation, RecommendationExplanation


def evaluate_explanations(
    explanations: Sequence[RecommendationExplanation],
) -> dict[str, float]:
    """
    Summarize explanation-layer outputs.

    These are structural quality metrics for the post-hoc explanation module,
    not ranking metrics.
    """
    if not explanations:
        return {
            "explanation_coverage": 0.0,
            "history_support_coverage": 0.0,
            "similar_item_support_coverage": 0.0,
            "mean_support_confidence": 0.0,
        }

    n = len(explanations)
    with_history = sum(1 for e in explanations if e.supporting_history)
    with_similar = sum(1 for e in explanations if e.supporting_similar_items)
    mean_conf = sum(float(e.support_confidence) for e in explanations) / n
    return {
        "explanation_coverage": 1.0,
        "history_support_coverage": with_history / n,
        "similar_item_support_coverage": with_similar / n,
        "mean_support_confidence": mean_conf,
    }


def evaluate_counterfactuals(
    counterfactuals: Sequence[CounterfactualExplanation],
) -> dict[str, float]:
    """
    Summarize counterfactual-layer outputs.

    These metrics are about output completeness and margin availability, not
    causal correctness.
    """
    if not counterfactuals:
        return {
            "counterfactual_coverage": 0.0,
            "weakening_action_coverage": 0.0,
            "margin_availability": 0.0,
            "mean_counterfactual_confidence": 0.0,
        }

    n = len(counterfactuals)
    with_actions = sum(1 for cf in counterfactuals if cf.weakening_actions)
    with_margin = sum(1 for cf in counterfactuals if cf.estimated_margin is not None)
    mean_conf = sum(float(cf.confidence) for cf in counterfactuals) / n
    return {
        "counterfactual_coverage": 1.0,
        "weakening_action_coverage": with_actions / n,
        "margin_availability": with_margin / n,
        "mean_counterfactual_confidence": mean_conf,
    }
