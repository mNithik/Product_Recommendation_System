"""Explicit recommendation stage built on top of ranking outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ranking import RankingResult


@dataclass(frozen=True)
class RecommendationResult:
    """Final recommendation output derived from a ranking result."""

    user_id: str
    recommended_items: list[str]
    ranking: RankingResult
    stage: str = "recommendation"
    metadata: dict[str, Any] = field(default_factory=dict)


def recommend_from_ranking(ranking: RankingResult, top_n: int) -> RecommendationResult:
    """Take the Top-N items from an explicit ranking result."""
    recommended_items = [row.item_id for row in ranking.items[:top_n]]
    return RecommendationResult(
        user_id=ranking.user_id,
        recommended_items=recommended_items,
        ranking=ranking,
        metadata={
            "top_n": int(top_n),
            "derived_from_ranking": True,
        },
    )
