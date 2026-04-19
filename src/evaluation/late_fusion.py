"""Late fusion: combine collaborative ranking in a pool with text similarity (post-hoc)."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Protocol

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import f_measure, ndcg_at_k, precision_at_k, recall_at_k


class TextSimilarityIndex(Protocol):
    def cosine_user_item(self, train_records: list[dict], user_id: str, item_asin: str) -> float: ...


def cf_position_scores(pool: list[str]) -> dict[str, float]:
    """
    Map pool order to [0, 1], higher = stronger collaborative signal (earlier in ``pool``).
    """
    n = len(pool)
    if n == 0:
        return {}
    if n == 1:
        return {pool[0]: 1.0}
    return {asin: (n - 1 - i) / (n - 1) for i, asin in enumerate(pool)}


def rank_items_late_fusion(
    pool: list[str],
    train_data: list[dict],
    user_id: str,
    text_cosine: Callable[[list[dict], str, str], float],
    alpha: float,
) -> list[tuple[str, float, float, float]]:
    """
    Sort ``pool`` by fused score.

    Returns:
        List of ``(asin, fused, cf_norm, text_sim)`` sorted by ``fused`` descending.
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    cf_by_asin = cf_position_scores(pool)
    rows: list[tuple[str, float, float, float]] = []
    for asin in pool:
        t = float(max(0.0, min(1.0, text_cosine(train_data, user_id, asin))))
        c = float(cf_by_asin.get(asin, 0.0))
        fused = alpha * c + (1.0 - alpha) * t
        rows.append((asin, fused, c, t))
    rows.sort(key=lambda x: -x[1])
    return rows


def evaluate_late_fusion_recommendations(
    model,
    train_data: list[dict],
    test_data: list[dict],
    text_index: TextSimilarityIndex,
    alpha: float,
    top_n: int = 10,
    pool_size: int = 80,
    min_train_ratings: int = 5,
    relevance_threshold: float | None = 4.0,
    max_users: int | None = 500,
) -> dict:
    """
    Same relevance protocol as ``evaluate_recommendations``, but Top-``n`` is built by
    late-fusing collaborative pool order with ``text_index.cosine_user_item``.

    This does **not** change the underlying model; it is an alternate ranking policy for
    offline comparison only.
    """
    test_by_user: dict[str, list[str]] = defaultdict(list)
    for r in test_data:
        if relevance_threshold is None or float(r["overall"]) >= float(relevance_threshold):
            test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user: dict[str, set[str]] = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    user_idx = getattr(model, "user_idx", None)
    if not user_idx:
        return {
            "Precision": 0.0,
            "Recall": 0.0,
            "F-measure": 0.0,
            "NDCG": 0.0,
            "n_users_eval": 0,
            "alpha": float(alpha),
            "pool_size": int(max(int(pool_size), int(top_n))),
        }

    users_eval = [
        u
        for u in test_by_user
        if u in user_idx and test_by_user[u] and len(train_by_user[u]) >= min_train_ratings
    ]
    if max_users is not None and max_users > 0:
        users_eval = users_eval[:max_users]

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    cos = text_index.cosine_user_item
    n_pool = max(int(pool_size), int(top_n))

    for user in tqdm(users_eval, desc="  Late-fusion Top-N", unit=" users"):
        relevant = test_by_user[user]
        exclude = train_by_user[user]
        pool = model.recommend_top_n(user, n=n_pool, exclude_items=exclude)
        if not pool:
            continue
        ranked = rank_items_late_fusion(pool, train_data, user, cos, alpha)
        recommended = [x[0] for x in ranked[:top_n]]
        precisions.append(precision_at_k(recommended, relevant, top_n))
        recalls.append(recall_at_k(recommended, relevant, top_n))
        ndcgs.append(ndcg_at_k(recommended, relevant, top_n))

    avg_p = float(np.mean(precisions)) if precisions else 0.0
    avg_r = float(np.mean(recalls)) if recalls else 0.0
    return {
        "Precision": avg_p,
        "Recall": avg_r,
        "F-measure": float(f_measure(avg_p, avg_r)),
        "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_eval": len(precisions),
        "alpha": float(alpha),
        "pool_size": int(n_pool),
    }
