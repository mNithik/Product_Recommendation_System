"""Explicit ranking stage wrappers for recommendation models.

This module introduces a thin ranking layer without changing baseline model
behavior. For now, ranking is derived from the model's existing Top-N API so we
can make the pipeline explicit before adding richer candidate scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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


def _to_python_float(value) -> float | None:
    """Best-effort conversion from NumPy / torch scalar-like values to float."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_from_popularity(model, item_id: str) -> float | None:
    if not hasattr(model, "item_idx") or not hasattr(model, "item_counts"):
        return None
    idx = model.item_idx.get(item_id)
    if idx is None:
        return None
    return _to_python_float(model.item_counts[idx])


def _score_from_latent_factors(model, user_id: str, item_id: str) -> float | None:
    if not all(hasattr(model, attr) for attr in ("user_idx", "item_idx", "U", "V")):
        return None
    u = model.user_idx.get(user_id)
    i = model.item_idx.get(item_id)
    if u is None or i is None:
        return None
    return _to_python_float((model.U[u] * model.V[i]).sum())


def _score_from_implicit_factors(model, user_id: str, item_id: str) -> float | None:
    backend = getattr(model, "model", None)
    if backend is None or not hasattr(backend, "user_factors") or not hasattr(backend, "item_factors"):
        return None
    if not hasattr(model, "user_idx") or not hasattr(model, "item_idx"):
        return None
    u = model.user_idx.get(user_id)
    i = model.item_idx.get(item_id)
    if u is None or i is None:
        return None
    return _to_python_float(np.dot(backend.user_factors[u], backend.item_factors[i]))


def _score_from_predict(model, user_id: str, item_id: str) -> float | None:
    if not hasattr(model, "predict"):
        return None
    return _to_python_float(model.predict(user_id, item_id))


def _score_ranked_item(model, user_id: str, item_id: str) -> tuple[float | None, str | None]:
    """
    Extract a score for an already-ranked item without changing recommendation order.

    Score extraction is model-specific but intentionally lightweight so the
    explicit ranking stage can carry useful metadata before we introduce richer
    candidate scoring adapters.
    """
    if hasattr(model, "item_counts"):
        score = _score_from_popularity(model, item_id)
        if score is not None:
            return score, "item_counts"

    score = _score_from_implicit_factors(model, user_id, item_id)
    if score is not None:
        return score, "latent_dot"

    score = _score_from_latent_factors(model, user_id, item_id)
    if score is not None:
        return score, "latent_dot"

    score = _score_from_predict(model, user_id, item_id)
    if score is not None:
        return score, "predict"

    return None, None


def _make_ranked_items(model, user_id: str, ranked_ids: list[str]) -> tuple[list[RankedItem], str | None]:
    score_source = None
    items: list[RankedItem] = []
    for rank, item_id in enumerate(ranked_ids, start=1):
        score, item_score_source = _score_ranked_item(model, user_id, item_id)
        if score_source is None and item_score_source is not None:
            score_source = item_score_source
        items.append(
            RankedItem(
                item_id=item_id,
                rank=rank,
                score=score,
            )
        )
    return items, score_source


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
    items, score_source = _make_ranked_items(model, user_id, ranked_ids)
    return RankingResult(
        user_id=user_id,
        items=items,
        metadata={
            "n_candidates": int(n_candidates),
            "max_candidates": int(max_candidates),
            "min_item_ratings": int(min_item_ratings),
            "source_model": type(model).__name__,
            "score_source": score_source,
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
            items, score_source = _make_ranked_items(model, user_id, ranked_ids)
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
                        "score_source": score_source,
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
