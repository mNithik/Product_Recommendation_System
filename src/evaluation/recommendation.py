"""Recommendation and Top-K evaluation wrappers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.pipeline import rank_items_for_user, rank_items_for_users, recommend_from_ranking

from .metrics import (
    compute_candidate_hit_rate,
    dcg_at_k,
    evaluate_recommendations,
    evaluate_recommendations_per_user,
    f_measure,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _normalized_item_popularity(train_data: list[dict]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for row in train_data:
        counts[row["asin"]] += 1
    max_count = max(counts.values(), default=1)
    return {item_id: float(count) / float(max_count) for item_id, count in counts.items()}


def _item_user_sets(train_data: list[dict]) -> dict[str, set[str]]:
    users_by_item: dict[str, set[str]] = defaultdict(set)
    for row in train_data:
        users_by_item[row["asin"]].add(row["reviewerID"])
    return users_by_item


def _pairwise_diversity(recommended: list[str], users_by_item: dict[str, set[str]]) -> float:
    if len(recommended) < 2:
        return 0.0
    distances: list[float] = []
    for i, left_item in enumerate(recommended[:-1]):
        left_users = users_by_item.get(left_item, set())
        for right_item in recommended[i + 1:]:
            right_users = users_by_item.get(right_item, set())
            union = left_users | right_users
            similarity = (len(left_users & right_users) / len(union)) if union else 0.0
            distances.append(1.0 - float(similarity))
    return float(np.mean(distances)) if distances else 0.0


def _mean_self_information(recommended: list[str], popularity: dict[str, float]) -> float:
    if not recommended:
        return 0.0
    scores = []
    for item_id in recommended:
        p = max(popularity.get(item_id, 0.0), 1e-12)
        scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0


def _average_popularity(recommended: list[str], popularity: dict[str, float]) -> float:
    if not recommended:
        return 0.0
    return float(np.mean([popularity.get(item_id, 0.0) for item_id in recommended]))


def evaluate_beyond_accuracy(
    model,
    train_data: list[dict],
    test_data: list[dict],
    *,
    top_n: int = 10,
    batch_size: int = 512,
    min_train_ratings: int = 5,
    max_candidates: int = 10000,
    relevance_threshold=None,
    min_item_ratings: int = 0,
    max_users: int | None = None,
) -> dict[str, float]:
    """
    Evaluate beyond-accuracy recommendation quality.

    Reports:
    - catalog coverage
    - intra-list diversity
    - novelty (mean self-information)
    - popularity concentration (higher means more popularity-heavy lists)
    """
    test_by_user = defaultdict(list)
    for row in test_data:
        if relevance_threshold is None or float(row["overall"]) >= float(relevance_threshold):
            test_by_user[row["reviewerID"]].append(row["asin"])

    train_by_user = defaultdict(set)
    for row in train_data:
        train_by_user[row["reviewerID"]].add(row["asin"])

    users_eval = [
        user_id
        for user_id in test_by_user
        if user_id in model.user_idx
        and test_by_user[user_id]
        and len(train_by_user[user_id]) >= min_train_ratings
    ]
    if max_users is not None and max_users > 0:
        users_eval = users_eval[:max_users]

    popularity = _normalized_item_popularity(train_data)
    users_by_item = _item_user_sets(train_data)
    all_recommended_items: set[str] = set()
    diversity_scores: list[float] = []
    novelty_scores: list[float] = []
    popularity_scores: list[float] = []

    if hasattr(model, "recommend_top_n_batch"):
        for i in range(0, len(users_eval), batch_size):
            batch_users = users_eval[i : i + batch_size]
            u_indices = [model.user_idx[user_id] for user_id in batch_users]
            exclude_sets = [
                {model.item_idx[item_id] for item_id in train_by_user[user_id] if item_id in model.item_idx}
                for user_id in batch_users
            ]
            ranking_batch = rank_items_for_users(
                model,
                user_ids=batch_users,
                user_indices=u_indices,
                exclude_sets=exclude_sets,
                n_candidates=top_n,
                max_candidates=max_candidates,
                min_item_ratings=min_item_ratings,
            )
            for ranking in ranking_batch:
                recommended = recommend_from_ranking(ranking, top_n=top_n).recommended_items
                all_recommended_items.update(recommended)
                diversity_scores.append(_pairwise_diversity(recommended, users_by_item))
                novelty_scores.append(_mean_self_information(recommended, popularity))
                popularity_scores.append(_average_popularity(recommended, popularity))
    else:
        for user_id in users_eval:
            ranking = rank_items_for_user(
                model,
                user_id=user_id,
                n_candidates=top_n,
                exclude_items=train_by_user[user_id],
                max_candidates=max_candidates,
                min_item_ratings=min_item_ratings,
            )
            recommended = recommend_from_ranking(ranking, top_n=top_n).recommended_items
            all_recommended_items.update(recommended)
            diversity_scores.append(_pairwise_diversity(recommended, users_by_item))
            novelty_scores.append(_mean_self_information(recommended, popularity))
            popularity_scores.append(_average_popularity(recommended, popularity))

    n_items_total = len({row["asin"] for row in train_data})
    return {
        "CatalogCoverage": (
            float(len(all_recommended_items)) / float(n_items_total) if n_items_total > 0 else 0.0
        ),
        "Diversity": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "Novelty": float(np.mean(novelty_scores)) if novelty_scores else 0.0,
        "PopularityConcentration": float(np.mean(popularity_scores)) if popularity_scores else 0.0,
        "n_users_beyond_accuracy_eval": float(len(users_eval)),
    }


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f_measure",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_recommendations",
    "evaluate_recommendations_per_user",
    "evaluate_beyond_accuracy",
    "compute_candidate_hit_rate",
]
