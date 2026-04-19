"""Evaluation helpers for explanation and counterfactual layers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from src.explainability import (
    CounterfactualExplanation,
    ItemSimilarityIndex,
    RecommendationExplanation,
    build_counterfactual_explanation,
    explain_recommendation,
)
from src.pipeline import rank_items_for_user, recommend_from_ranking
from src.postprocessing import CausalAdjustmentConfig, apply_causal_adjustment

from .metrics import f_measure, ndcg_at_k, precision_at_k, recall_at_k


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


def evaluate_explainable_recommendations(
    model,
    train_data: list[dict],
    test_data: list[dict],
    *,
    top_n: int = 10,
    min_train_ratings: int = 5,
    max_candidates: int = 10000,
    relevance_threshold: float | None = None,
    min_item_ratings: int = 0,
    explanation_sample_users: int | None = 250,
    causal_config: CausalAdjustmentConfig | None = None,
) -> dict[str, float]:
    """
    Evaluate post-hoc explanation, counterfactual, and optional causal layers.

    This helper is intentionally separate from baseline recommendation metrics.
    It operates on top of the ranking stage and reports:
    - explanation output coverage/quality summaries,
    - counterfactual output coverage/quality summaries,
    - optional causal-adjusted Top-N metrics when enabled.
    """
    test_by_user: dict[str, list[str]] = defaultdict(list)
    for row in test_data:
        if relevance_threshold is None or float(row["overall"]) >= float(relevance_threshold):
            test_by_user[row["reviewerID"]].append(row["asin"])

    train_by_user: dict[str, set[str]] = defaultdict(set)
    item_popularity: dict[str, int] = defaultdict(int)
    for row in train_data:
        user_id = row["reviewerID"]
        item_id = row["asin"]
        train_by_user[user_id].add(item_id)
        item_popularity[item_id] += 1

    users_eval = [
        user_id
        for user_id in test_by_user
        if user_id in model.user_idx
        and test_by_user[user_id]
        and len(train_by_user[user_id]) >= min_train_ratings
    ]
    if explanation_sample_users is not None and explanation_sample_users > 0:
        users_eval = users_eval[:explanation_sample_users]

    if not users_eval:
        metrics = evaluate_explanations([])
        metrics.update(evaluate_counterfactuals([]))
        metrics["n_explanation_users_eval"] = 0
        if (causal_config or CausalAdjustmentConfig()).enabled:
            metrics.update(
                {
                    "causal_Precision": 0.0,
                    "causal_Recall": 0.0,
                    "causal_F-measure": 0.0,
                    "causal_NDCG": 0.0,
                }
            )
        return metrics

    similarity_index = ItemSimilarityIndex(train_data)
    explanations: list[RecommendationExplanation] = []
    counterfactuals: list[CounterfactualExplanation] = []
    causal_precisions: list[float] = []
    causal_recalls: list[float] = []
    causal_ndcgs: list[float] = []
    cfg = causal_config or CausalAdjustmentConfig()
    n_candidates = max(int(top_n) + 1, int(top_n))

    for user_id in users_eval:
        relevant = test_by_user[user_id]
        ranking = rank_items_for_user(
            model,
            user_id=user_id,
            n_candidates=n_candidates,
            exclude_items=train_by_user[user_id],
            max_candidates=max_candidates,
            min_item_ratings=min_item_ratings,
        )
        recommendation = recommend_from_ranking(ranking, top_n=top_n)
        user_explanations = [
            explain_recommendation(similarity_index, user_id, item_id)
            for item_id in recommendation.recommended_items
        ]
        explanations.extend(user_explanations)
        counterfactuals.extend(
            build_counterfactual_explanation(explanation, ranking)
            for explanation in user_explanations
        )

        if cfg.enabled:
            explanations_by_item = {
                explanation.recommended_item: explanation for explanation in user_explanations
            }
            adjusted_ranking = apply_causal_adjustment(
                ranking,
                explanations_by_item,
                item_popularity=dict(item_popularity),
                config=cfg,
            )
            adjusted_recommendation = recommend_from_ranking(adjusted_ranking, top_n=top_n)
            adjusted_items = adjusted_recommendation.recommended_items
            causal_precisions.append(precision_at_k(adjusted_items, relevant, top_n))
            causal_recalls.append(recall_at_k(adjusted_items, relevant, top_n))
            causal_ndcgs.append(ndcg_at_k(adjusted_items, relevant, top_n))

    metrics = evaluate_explanations(explanations)
    metrics.update(evaluate_counterfactuals(counterfactuals))
    metrics["n_explanation_users_eval"] = float(len(users_eval))

    if cfg.enabled:
        avg_precision = float(np.mean(causal_precisions)) if causal_precisions else 0.0
        avg_recall = float(np.mean(causal_recalls)) if causal_recalls else 0.0
        metrics.update(
            {
                "causal_Precision": avg_precision,
                "causal_Recall": avg_recall,
                "causal_F-measure": float(f_measure(avg_precision, avg_recall)),
                "causal_NDCG": float(np.mean(causal_ndcgs)) if causal_ndcgs else 0.0,
            }
        )

    return metrics
