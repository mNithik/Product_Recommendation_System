"""Structured post-hoc explanations for recommended items."""

from __future__ import annotations

from dataclasses import dataclass

from .item_similarity import ItemSimilarityIndex, SimilarItemSupport


@dataclass(frozen=True)
class RecommendationExplanation:
    """Structured explanation object for one recommended item."""

    user_id: str
    recommended_item: str
    explanation_text: str
    support_confidence: float
    supporting_history: list[SimilarItemSupport]
    supporting_similar_items: list[SimilarItemSupport]
    explanation_type: str


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _confidence_from_support(
    history_support: list[SimilarItemSupport],
    similar_item_support: list[SimilarItemSupport],
) -> float:
    if not history_support and not similar_item_support:
        return 0.15

    best_history_overlap = history_support[0].user_overlap if history_support else 0
    best_history_similarity = history_support[0].similarity if history_support else 0.0
    best_item_similarity = similar_item_support[0].similarity if similar_item_support else 0.0

    overlap_component = min(best_history_overlap / 5.0, 1.0)
    score = 0.55 * overlap_component + 0.30 * best_history_similarity + 0.15 * best_item_similarity
    return _clamp_score(score)


def explain_recommendation(
    index: ItemSimilarityIndex,
    user_id: str,
    recommended_item: str,
    *,
    min_history_rating: float = 4.0,
    top_history_support: int = 3,
    top_similar_items: int = 3,
) -> RecommendationExplanation:
    """
    Explain a recommended item using historical support and item similarity.

    This is a practical post-hoc explanation layer designed for transparency and
    reportability, not a reproduction of a heavy explainability paper.
    """
    history_support = index.supporting_history_items(
        user_id,
        recommended_item,
        min_rating=min_history_rating,
        top_k=top_history_support,
    )
    history_items = {row["item"] for row in index.user_history_items(user_id)}
    similar_item_support = index.similar_items(
        recommended_item,
        exclude_items=history_items | {recommended_item},
        top_k=top_similar_items,
    )
    confidence = _confidence_from_support(history_support, similar_item_support)

    if history_support:
        top = history_support[0]
        explanation_text = (
            f"Recommended because your strong interaction with {top.item_id} is supported by "
            f"{top.user_overlap} overlapping users and item similarity {top.similarity:.2f}."
        )
        explanation_type = "history_and_overlap"
    elif index.user_history_items(user_id):
        explanation_text = (
            f"Recommended from your broader interaction profile and local item-neighborhood support "
            f"around {recommended_item}."
        )
        explanation_type = "profile_support"
    else:
        explanation_text = (
            "Recommended mainly from global popularity because there is little or no user history."
        )
        explanation_type = "cold_start_popularity"

    return RecommendationExplanation(
        user_id=user_id,
        recommended_item=recommended_item,
        explanation_text=explanation_text,
        support_confidence=confidence,
        supporting_history=history_support,
        supporting_similar_items=similar_item_support,
        explanation_type=explanation_type,
    )
