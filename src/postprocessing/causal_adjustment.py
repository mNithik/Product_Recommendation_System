"""Lightweight causal-inspired post-ranking score adjustment."""

from __future__ import annotations

from dataclasses import dataclass

from src.explainability import RecommendationExplanation
from src.pipeline import RankedItem, RankingResult


@dataclass(frozen=True)
class CausalAdjustmentConfig:
    """Configuration for optional post-ranking score adjustment."""

    enabled: bool = False
    support_weight: float = 0.20
    popularity_penalty_weight: float = 0.10


def _base_score(item: RankedItem) -> float:
    """
    Stable base score for adjustment.

    If a true model score exists we use it. Otherwise we use inverse rank as a
    conservative fallback so the layer still behaves deterministically.
    """
    if item.score is not None:
        return float(item.score)
    return float(-item.rank)


def _normalized_popularity(item_id: str, item_popularity: dict[str, int] | None) -> float:
    if not item_popularity:
        return 0.0
    max_count = max(item_popularity.values(), default=0)
    if max_count <= 0:
        return 0.0
    return float(item_popularity.get(item_id, 0)) / float(max_count)


def apply_causal_adjustment(
    ranking: RankingResult,
    explanations_by_item: dict[str, RecommendationExplanation],
    *,
    item_popularity: dict[str, int] | None = None,
    config: CausalAdjustmentConfig | None = None,
) -> RankingResult:
    """
    Apply a lightweight causal-inspired score adjustment to a ranking result.

    The adjustment is post-hoc and toggleable. It preserves the original ranking
    in metadata and adds interpretable components to each adjusted item.
    """
    cfg = config or CausalAdjustmentConfig()
    if not cfg.enabled:
        return ranking

    adjusted_rows: list[RankedItem] = []
    for row in ranking.items:
        explanation = explanations_by_item.get(row.item_id)
        support_confidence = explanation.support_confidence if explanation is not None else 0.0
        popularity = _normalized_popularity(row.item_id, item_popularity)
        base = _base_score(row)
        support_boost = cfg.support_weight * support_confidence
        popularity_penalty = cfg.popularity_penalty_weight * popularity
        adjusted_score = base + support_boost - popularity_penalty
        adjusted_rows.append(
            RankedItem(
                item_id=row.item_id,
                rank=row.rank,
                score=adjusted_score,
                metadata={
                    **row.metadata,
                    "base_score": row.score,
                    "base_rank": row.rank,
                    "support_confidence": support_confidence,
                    "support_boost": support_boost,
                    "popularity_penalty": popularity_penalty,
                    "adjusted_score": adjusted_score,
                },
            )
        )

    adjusted_rows.sort(key=lambda item: (-float(item.score), item.metadata.get("base_rank", item.rank), item.item_id))
    reranked_rows = [
        RankedItem(
            item_id=row.item_id,
            rank=idx,
            score=row.score,
            metadata=row.metadata,
        )
        for idx, row in enumerate(adjusted_rows, start=1)
    ]
    return RankingResult(
        user_id=ranking.user_id,
        items=reranked_rows,
        metadata={
            **ranking.metadata,
            "causal_adjustment_enabled": True,
            "support_weight": cfg.support_weight,
            "popularity_penalty_weight": cfg.popularity_penalty_weight,
            "base_ranking_items": [row.item_id for row in ranking.items],
        },
    )
