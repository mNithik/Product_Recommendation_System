"""Evaluation metrics for rating prediction and Top-N recommendation."""

import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def mae(predictions, actuals) -> float:
    return float(np.mean(np.abs(np.array(predictions) - np.array(actuals))))


def rmse(predictions, actuals) -> float:
    return float(np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)))


def precision_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    rec_set = set(recommended[:k])
    return len(rec_set & set(relevant)) / k if k > 0 else 0


def recall_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    if not relevant:
        return 0
    rec_set = set(recommended[:k])
    return len(rec_set & set(relevant)) / len(relevant)


def f_measure(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def dcg_at_k(relevant, recommended, k: int = 10) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    rel_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            dcg += 1.0 / np.log2(i + 2)
    n_rel = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0
    return dcg / idcg


def evaluate_rating_prediction(model, test_data: list[dict],
                               batch_size: int = 32768) -> dict:
    """Evaluate MAE and RMSE for rating prediction."""
    actuals = np.array([float(r["overall"]) for r in test_data], dtype=np.float64)
    if hasattr(model, "predict_batch"):
        preds = []
        fallback_mean = float(getattr(model, "global_mean", 3.0))
        for i in tqdm(range(0, len(test_data), batch_size),
                      desc="  Predicting ratings", unit=" batches"):
            batch = test_data[i: i + batch_size]
            u_ix, i_ix, valid_mask = [], [], []
            for r in batch:
                u = model.user_idx.get(r["reviewerID"])
                it = model.item_idx.get(r["asin"])
                v = u is not None and it is not None
                valid_mask.append(v)
                if v:
                    u_ix.append(u)
                    i_ix.append(it)
            p_iter = iter([])
            if u_ix:
                p_batch = model.predict_batch(u_ix, i_ix)
                p_iter = iter(p_batch)
            for v in valid_mask:
                p = next(p_iter) if v else fallback_mean
                preds.append(float(np.nan_to_num(p, nan=fallback_mean)))
        preds = np.array(preds, dtype=np.float64)
        return {"MAE": mae(preds, actuals), "RMSE": rmse(preds, actuals)}

    preds = []
    for r in tqdm(test_data, desc="  Predicting ratings", unit=" reviews"):
        preds.append(model.predict(r["reviewerID"], r["asin"]))
    return {"MAE": mae(preds, actuals), "RMSE": rmse(preds, actuals)}


def compute_candidate_hit_rate(model, train_data, test_data,
                               max_candidates=10000, min_train_ratings=5,
                               relevance_threshold=None, min_item_ratings=0):
    """Diagnostic: fraction of test items present in candidate pool before scoring."""
    if not hasattr(model, "_get_candidates"):
        return None
    test_by_user = defaultdict(list)
    for r in test_data:
        if relevance_threshold is None or float(r["overall"]) >= float(relevance_threshold):
            test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    users_eval = [
        u for u in test_by_user
        if u in model.user_idx and test_by_user[u]
        and len(train_by_user[u]) >= min_train_ratings
    ]
    hits, total = 0, 0
    for u in users_eval[:500]:
        u_idx = model.user_idx[u]
        exclude = {model.item_idx[iid] for iid in train_by_user[u] if iid in model.item_idx}
        candidates = set(model._get_candidates(u_idx, exclude, max_candidates,
                                               min_item_ratings=min_item_ratings))
        for iid in test_by_user[u]:
            if iid in model.item_idx and model.item_idx[iid] in candidates:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0


def evaluate_recommendations(model, train_data, test_data, top_n=10,
                             batch_size=512, min_train_ratings=5,
                             max_candidates=10000, relevance_threshold=None,
                             min_item_ratings=0) -> dict:
    """Evaluate Top-N: Precision, Recall, F-measure, NDCG."""
    test_by_user = defaultdict(list)
    for r in test_data:
        if relevance_threshold is None or float(r["overall"]) >= float(relevance_threshold):
            test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    users_eval = [
        u for u in test_by_user
        if u in model.user_idx and test_by_user[u]
        and len(train_by_user[u]) >= min_train_ratings
    ]
    precisions, recalls, ndcgs = [], [], []

    if hasattr(model, "recommend_top_n_batch"):
        for i in tqdm(range(0, len(users_eval), batch_size),
                      desc="  Top-N recommendations", unit=" batches"):
            batch_users = users_eval[i: i + batch_size]
            u_indices = [model.user_idx[u] for u in batch_users]
            exclude_sets = [
                {model.item_idx[iid] for iid in train_by_user[u] if iid in model.item_idx}
                for u in batch_users
            ]
            recs_batch = model.recommend_top_n_batch(
                u_indices, exclude_sets, n=top_n,
                max_candidates=max_candidates, min_item_ratings=min_item_ratings,
            )
            for user, relevant, recommended in zip(
                batch_users, [test_by_user[u] for u in batch_users], recs_batch
            ):
                precisions.append(precision_at_k(recommended, relevant, top_n))
                recalls.append(recall_at_k(recommended, relevant, top_n))
                ndcgs.append(ndcg_at_k(recommended, relevant, top_n))
    else:
        for user in tqdm(users_eval, desc="  Top-N recommendations", unit=" users"):
            relevant = test_by_user[user]
            exclude = train_by_user[user]
            recommended = model.recommend_top_n(user, n=top_n, exclude_items=exclude)
            precisions.append(precision_at_k(recommended, relevant, top_n))
            recalls.append(recall_at_k(recommended, relevant, top_n))
            ndcgs.append(ndcg_at_k(recommended, relevant, top_n))

    avg_p = np.mean(precisions) if precisions else 0
    avg_r = np.mean(recalls) if recalls else 0
    return {
        "Precision": float(avg_p),
        "Recall": float(avg_r),
        "F-measure": float(f_measure(avg_p, avg_r)),
        "NDCG": float(np.mean(ndcgs)) if ndcgs else 0,
        "n_users_eval": len(users_eval),
    }


def evaluate_recommendations_per_user(
    model,
    train_data,
    test_data,
    top_n: int = 10,
    batch_size: int = 512,
    min_train_ratings: int = 5,
    max_candidates: int = 10000,
    relevance_threshold=None,
    min_item_ratings: int = 0,
    max_users: int | None = None,
) -> list[dict]:
    """
    Same protocol as evaluate_recommendations, but return one record per user
    (for fairness / disparity analysis by activity).
    """
    test_by_user = defaultdict(list)
    for r in test_data:
        if relevance_threshold is None or float(r["overall"]) >= float(relevance_threshold):
            test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    users_eval = [
        u
        for u in test_by_user
        if u in model.user_idx
        and test_by_user[u]
        and len(train_by_user[u]) >= min_train_ratings
    ]
    if max_users is not None and max_users > 0:
        users_eval = users_eval[:max_users]

    rows: list[dict] = []

    if hasattr(model, "recommend_top_n_batch"):
        for i in range(0, len(users_eval), batch_size):
            batch_users = users_eval[i : i + batch_size]
            u_indices = [model.user_idx[u] for u in batch_users]
            exclude_sets = [
                {model.item_idx[iid] for iid in train_by_user[u] if iid in model.item_idx}
                for u in batch_users
            ]
            recs_batch = model.recommend_top_n_batch(
                u_indices,
                exclude_sets,
                n=top_n,
                max_candidates=max_candidates,
                min_item_ratings=min_item_ratings,
            )
            for user, relevant, recommended in zip(
                batch_users, [test_by_user[u] for u in batch_users], recs_batch
            ):
                p = precision_at_k(recommended, relevant, top_n)
                r = recall_at_k(recommended, relevant, top_n)
                n = ndcg_at_k(recommended, relevant, top_n)
                rows.append(
                    {
                        "user": user,
                        "n_train": len(train_by_user[user]),
                        "precision": float(p),
                        "recall": float(r),
                        "ndcg": float(n),
                    }
                )
    else:
        for user in users_eval:
            relevant = test_by_user[user]
            exclude = train_by_user[user]
            recommended = model.recommend_top_n(user, n=top_n, exclude_items=exclude)
            p = precision_at_k(recommended, relevant, top_n)
            r = recall_at_k(recommended, relevant, top_n)
            n = ndcg_at_k(recommended, relevant, top_n)
            rows.append(
                {
                    "user": user,
                    "n_train": len(train_by_user[user]),
                    "precision": float(p),
                    "recall": float(r),
                    "ndcg": float(n),
                }
            )
    return rows
