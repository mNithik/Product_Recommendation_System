"""Explicit ranking stage wrappers for recommendation models.

This module introduces a thin ranking layer without changing baseline model
behavior. For now, ranking is derived from the model's existing Top-N API so we
can make the pipeline explicit before adding richer candidate scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RankedItem:
    """Single ranked item produced by the ranking stage."""

    item_id: str
    rank: int
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingResult:
    """Ordered ranking output for one user."""

    user_id: str
    items: list[RankedItem]
    stage: str = "ranking"
    metadata: dict[str, Any] = field(default_factory=dict)


def rank_items_for_user(
    model,
    user_id: str,
    n_candidates: int,
    exclude_items=None,
    max_candidates: int = 10000,
    min_item_ratings: int = 0,
) -> RankingResult:
    """
    Build an explicit ranking result for a single user.

    This wrapper preserves current behavior by delegating to the model's
    ``recommend_top_n`` method and wrapping the ordered items in a ranking
    object. Later phases can replace this with true scored candidate ranking
    without changing callers.
    """
    ranked_ids = model.recommend_top_n(
        user_id,
        n=n_candidates,
        exclude_items=exclude_items,
    )
    items = [
        RankedItem(item_id=item_id, rank=rank)
        for rank, item_id in enumerate(ranked_ids, start=1)
    ]
    return RankingResult(
        user_id=user_id,
        items=items,
        metadata={
            "n_candidates": int(n_candidates),
            "max_candidates": int(max_candidates),
            "min_item_ratings": int(min_item_ratings),
            "source_model": type(model).__name__,
        },
    )


def rank_items_for_users(
    model,
    user_ids: list[str],
    user_indices: list[int],
    exclude_sets: list[set[int]],
    n_candidates: int,
    max_candidates: int = 10000,
    min_item_ratings: int = 0,
) -> list[RankingResult]:
    """
    Build explicit ranking results for a batch of users.

    When available, this delegates to ``recommend_top_n_batch`` to preserve the
    existing optimized inference path.
    """
    if hasattr(model, "recommend_top_n_batch"):
        recs_batch = model.recommend_top_n_batch(
            user_indices,
            exclude_sets,
            n=n_candidates,
            max_candidates=max_candidates,
            min_item_ratings=min_item_ratings,
        )
        results: list[RankingResult] = []
        for user_id, ranked_ids in zip(user_ids, recs_batch):
            items = [
                RankedItem(item_id=item_id, rank=rank)
                for rank, item_id in enumerate(ranked_ids, start=1)
            ]
            results.append(
                RankingResult(
                    user_id=user_id,
                    items=items,
                    metadata={
                        "n_candidates": int(n_candidates),
                        "max_candidates": int(max_candidates),
                        "min_item_ratings": int(min_item_ratings),
                        "source_model": type(model).__name__,
                        "batched": True,
                    },
                )
            )
        return results

    return [
        rank_items_for_user(
            model,
            user_id=user_id,
            n_candidates=n_candidates,
            exclude_items=(
                {model.rev_item[i] for i in exclude_sets[idx] if hasattr(model, "rev_item")}
                if idx < len(exclude_sets) else None
            ),
            max_candidates=max_candidates,
            min_item_ratings=min_item_ratings,
        )
        for idx, user_id in enumerate(user_ids)
    ]
