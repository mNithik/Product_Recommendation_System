"""Approximate counterfactual explanations for ranked recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline import RankingResult

from .explanation_engine import RecommendationExplanation


@dataclass(frozen=True)
class CounterfactualAction:
    """One minimal weakening action candidate."""

    item_id: str
    action: str
    estimated_impact: float
    reason: str


@dataclass(frozen=True)
class CounterfactualExplanation:
    """Approximate counterfactual explanation for one recommended item."""

    user_id: str
    recommended_item: str
    current_rank: int
    current_score: float | None
    next_item: str | None
    next_item_score: float | None
    estimated_margin: float | None
    weakening_actions: list[CounterfactualAction]
    minimal_change_text: str
    confidence: float


def _safe_margin(current_score: float | None, next_score: float | None) -> float | None:
    if current_score is None or next_score is None:
        return None
    return max(0.0, float(current_score) - float(next_score))


def _support_influence(overlap: int, similarity: float, avg_rating: float) -> float:
    overlap_component = min(overlap / 5.0, 1.0)
    rating_component = min(avg_rating / 5.0, 1.0)
    score = 0.55 * overlap_component + 0.30 * similarity + 0.15 * rating_component
    return max(0.0, min(1.0, score))


def build_counterfactual_explanation(
    explanation: RecommendationExplanation,
    ranking: RankingResult,
) -> CounterfactualExplanation:
    """
    Build a practical CountER-style counterfactual explanation.

    The implementation is intentionally approximate and post-hoc. It does not
    retrain the model or require a full causal graph. Instead, it estimates
    which supporting interactions appear most responsible for holding the item
    near its current rank, and states the smallest weakening condition likely to
    lower it.
    """
    ranked_items = ranking.items
    current_idx = next(
        (idx for idx, row in enumerate(ranked_items) if row.item_id == explanation.recommended_item),
        None,
    )
    if current_idx is None:
        return CounterfactualExplanation(
            user_id=explanation.user_id,
            recommended_item=explanation.recommended_item,
            current_rank=-1,
            current_score=None,
            next_item=None,
            next_item_score=None,
            estimated_margin=None,
            weakening_actions=[],
            minimal_change_text="Counterfactual weakening is unavailable because the item is not in the ranking result.",
            confidence=0.0,
        )

    current_row = ranked_items[current_idx]
    next_row = ranked_items[current_idx + 1] if current_idx + 1 < len(ranked_items) else None
    margin = _safe_margin(current_row.score, next_row.score if next_row else None)

    actions: list[CounterfactualAction] = []
    for support in explanation.supporting_history:
        impact = _support_influence(support.user_overlap, support.similarity, support.avg_rating)
        actions.append(
            CounterfactualAction(
                item_id=support.item_id,
                action="remove_or_weaken_positive_interaction",
                estimated_impact=impact,
                reason=(
                    f"Support item {support.item_id} contributes overlap={support.user_overlap} "
                    f"and similarity={support.similarity:.2f}."
                ),
            )
        )

    actions.sort(key=lambda row: (-row.estimated_impact, row.item_id))

    if actions:
        chosen = actions[0]
        if len(actions) > 1 and margin is not None and chosen.estimated_impact < margin:
            cumulative = 0.0
            chosen_items: list[str] = []
            for row in actions:
                cumulative += row.estimated_impact
                chosen_items.append(row.item_id)
                if cumulative >= margin:
                    break
            if len(chosen_items) == 1:
                minimal_change_text = (
                    f"If the positive interaction with {chosen_items[0]} were removed or weakened, "
                    f"{explanation.recommended_item} would likely drop below its current position."
                )
            else:
                items_text = ", ".join(chosen_items)
                minimal_change_text = (
                    f"If support from {items_text} were jointly weakened, "
                    f"{explanation.recommended_item} would likely lose its current rank advantage."
                )
        else:
            minimal_change_text = (
                f"If the positive interaction with {chosen.item_id} were removed or weakened, "
                f"{explanation.recommended_item} would likely drop in rank or score."
            )
    else:
        minimal_change_text = (
            f"If broader popularity or neighborhood support for {explanation.recommended_item} weakened, "
            "its position would likely fall."
        )

    return CounterfactualExplanation(
        user_id=explanation.user_id,
        recommended_item=explanation.recommended_item,
        current_rank=current_row.rank,
        current_score=current_row.score,
        next_item=next_row.item_id if next_row else None,
        next_item_score=next_row.score if next_row else None,
        estimated_margin=margin,
        weakening_actions=actions[:3],
        minimal_change_text=minimal_change_text,
        confidence=explanation.support_confidence,
    )
