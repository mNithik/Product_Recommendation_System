"""
Recommender system: Item-based Collaborative Filtering.
Implements rating prediction and Top-N recommendation with evaluation metrics.
"""

import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm


def load_data(filepath):
    """Load train or test data from JSONL."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  Loading {os.path.basename(filepath)}", unit=" lines"):
            if line.strip():
                data.append(json.loads(line))
    return data


def build_index(data):
    """Build user/item mappings and rating matrix indices."""
    users = {}
    items = {}
    for r in data:
        u, i = r["reviewerID"], r["asin"]
        if u not in users:
            users[u] = len(users)
        if i not in items:
            items[i] = len(items)

    rev_users = {v: k for k, v in users.items()}
    rev_items = {v: k for k, v in items.items()}
    return users, items, rev_users, rev_items


class ItemBasedCF:
    """
    Item-based Collaborative Filtering using cosine similarity.
    Built from scratch using numpy as building blocks.
    """

    def __init__(self, k=50):
        self.k = k  # number of nearest neighbors
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.ratings = None  # sparse: dict (u_idx, i_idx) -> rating
        self.item_means = None
        self.similarity = None
        self.item_raters = None

    def fit(self, data):
        """Fit model on training data."""
        print("  Building user/item indices...")
        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)

        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        print(f"  {n_users} users, {n_items} items")

        self.ratings = defaultdict(float)
        item_sums = defaultdict(float)
        item_counts = defaultdict(int)

        for r in data:
            u = self.user_idx[r["reviewerID"]]
            i = self.item_idx[r["asin"]]
            rating = float(r["overall"])
            self.ratings[(u, i)] = rating
            item_sums[i] += rating
            item_counts[i] += 1

        self.item_means = np.zeros(n_items)
        for i in range(n_items):
            if item_counts[i] > 0:
                self.item_means[i] = item_sums[i] / item_counts[i]
            else:
                self.item_means[i] = 3.0  # default

        # Build user -> items mapping
        self.item_raters = defaultdict(set)
        user_items = defaultdict(list)  # user -> [(item, rating), ...]
        for (u, i), r in self.ratings.items():
            self.item_raters[i].add(u)
            user_items[u].append((i, r))

        # Compute similarity only for item pairs that co-occur (share raters)
        # Adjusted cosine: over common users only
        pair_num = defaultdict(float)  # (i,j) -> sum (r_i - mean_i)*(r_j - mean_j)
        pair_den_i = defaultdict(float)  # (i,j) -> sum (r_i - mean_i)^2
        pair_den_j = defaultdict(float)  # (i,j) -> sum (r_j - mean_j)^2

        print("  Computing item-item similarities...")
        for u, rated in tqdm(user_items.items(), desc="  Similarity", unit=" users"):
            centered = [(i, r - self.item_means[i]) for i, r in rated]
            for (i, ci), (j, cj) in combinations(centered, 2):
                if i > j:
                    i, j = j, i
                    ci, cj = cj, ci
                pair_num[(i, j)] += ci * cj
                pair_den_i[(i, j)] += ci * ci
                pair_den_j[(i, j)] += cj * cj

        print("  Building similarity matrix...")
        self.similarity = defaultdict(dict)
        for (i, j), num in tqdm(pair_num.items(), desc="  Pairs", unit=" pairs"):
            den = np.sqrt(pair_den_i[(i, j)] * pair_den_j[(i, j)]) + 1e-9
            sim = max(0, num / den)
            self.similarity[i][j] = sim
            self.similarity[j][i] = sim

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.item_means.mean() if self.item_means is not None else 3.0

        u = self.user_idx[user_id]
        i = self.item_idx[item_id]

        if (u, i) in self.ratings:
            return self.ratings[(u, i)]

        # Find items user has rated
        user_items = [j for (uid, j), _ in self.ratings.items() if uid == u]
        if not user_items:
            return self.item_means[i]

        # Get k nearest neighbors (items similar to i that user rated)
        neighbors = []
        for j in user_items:
            if j in self.similarity and i in self.similarity[j]:
                sim = self.similarity[j][i]
                r = self.ratings[(u, j)]
                neighbors.append((sim, r - self.item_means[j]))

        if not neighbors:
            return self.item_means[i]

        neighbors.sort(key=lambda x: -x[0])
        top_k = neighbors[: self.k]

        num = sum(s * d for s, d in top_k)
        den = sum(abs(s) for s, _ in top_k)
        if den < 1e-9:
            return self.item_means[i]

        return self.item_means[i] + num / den

    def recommend_top_n(self, user_id, n=10, exclude_items=None):
        """Recommend top-N items for user (excluding already rated)."""
        if user_id not in self.user_idx:
            return []

        u = self.user_idx[user_id]
        exclude = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude.add(self.item_idx[iid])

        # Items user has already rated in train
        rated = {j for (uid, j), _ in self.ratings.items() if uid == u}
        rated |= exclude

        # Score all candidate items
        candidates = []
        for i in range(len(self.item_idx)):
            if i in rated:
                continue
            pred = self._predict_internal(u, i)
            item_id = self.rev_item[i]
            candidates.append((pred, item_id))

        candidates.sort(key=lambda x: -x[0])
        return [item_id for _, item_id in candidates[:n]]

    def _predict_internal(self, u, i):
        """Internal predict using indices."""
        user_items = [j for (uid, j), _ in self.ratings.items() if uid == u]
        if not user_items:
            return self.item_means[i]

        neighbors = []
        for j in user_items:
            if j in self.similarity and i in self.similarity[j]:
                sim = self.similarity[j][i]
                r = self.ratings[(u, j)]
                neighbors.append((sim, r - self.item_means[j]))

        if not neighbors:
            return self.item_means[i]

        neighbors.sort(key=lambda x: -x[0])
        top_k = neighbors[: self.k]
        num = sum(s * d for s, d in top_k)
        den = sum(abs(s) for s, _ in top_k)
        if den < 1e-9:
            return self.item_means[i]
        return self.item_means[i] + num / den


class MatrixFactorizationGPU:
    """
    Matrix Factorization with User/Item Biases, trained with ALS on GPU.
    Predictions: mu + b_u + b_i + U[u] @ V[i]^T.
    Supports limited candidates for Top-N (reachable items only).
    """

    def __init__(self, n_factors=64, reg=0.1, n_epochs=10, device=None):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.U = None
        self.V = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 3.0
        self.user_items = None
        self.item_cooccur = None

    def fit(self, data):
        """Fit MF model with biases using ALS on GPU."""
        import torch

        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        print(f"  {n_users} users, {n_items} items | Device: {self.device}")

        users_arr = np.array([self.user_idx[r["reviewerID"]] for r in data])
        items_arr = np.array([self.item_idx[r["asin"]] for r in data])
        ratings_arr = np.array([float(r["overall"]) for r in data], dtype=np.float32)
        self.global_mean = float(np.mean(ratings_arr))

        user_items = [[] for _ in range(n_users)]
        item_users = [[] for _ in range(n_items)]
        for u, i, r in zip(users_arr, items_arr, ratings_arr):
            user_items[u].append((i, r))
            item_users[i].append((u, r))

        self.user_items = [set(x[0] for x in user_items[u]) for u in range(n_users)]
        self.item_rating_counts = np.array([len(item_users[i]) for i in range(n_items)])

        user_means = np.array([np.mean([x[1] for x in user_items[u]]) if user_items[u] else self.global_mean for u in range(n_users)], dtype=np.float32)
        item_means = np.array([np.mean([x[1] for x in item_users[i]]) if item_users[i] else self.global_mean for i in range(n_items)], dtype=np.float32)
        self.user_bias = torch.tensor(user_means - self.global_mean, device=self.device)
        self.item_bias = torch.tensor(item_means - self.global_mean, device=self.device)

        ratings_centered = ratings_arr - self.global_mean - self.user_bias.cpu().numpy()[users_arr] - self.item_bias.cpu().numpy()[items_arr]
        ratings_centered = ratings_centered.astype(np.float32)

        self.V = torch.randn(n_items, self.n_factors, device=self.device) * 0.01
        self.U = torch.zeros(n_users, self.n_factors, device=self.device)

        print("  Building candidate sets (item co-occurrence)...")
        self.item_cooccur = defaultdict(set)
        for u, rated in enumerate(user_items):
            items_u = [x[0] for x in rated]
            for i, j in combinations(items_u, 2):
                self.item_cooccur[i].add(j)
                self.item_cooccur[j].add(i)

        print("  Training (ALS with biases on GPU)...")
        reg_I = self.reg * torch.eye(self.n_factors, device=self.device)
        for epoch in tqdm(range(self.n_epochs), desc="  Epochs"):
            for u in range(n_users):
                if not user_items[u]:
                    continue
                idx = np.where(users_arr == u)[0]
                items_u = items_arr[idx]
                r_centered = torch.tensor(ratings_centered[idx], dtype=torch.float32, device=self.device)
                V_u = self.V[items_u]
                A = V_u.T @ V_u + reg_I * len(items_u)
                b = V_u.T @ r_centered
                self.U[u] = torch.linalg.solve(A, b)
            for i in range(n_items):
                if not item_users[i]:
                    continue
                idx = np.where(items_arr == i)[0]
                users_i = users_arr[idx]
                r_centered = torch.tensor(ratings_centered[idx], dtype=torch.float32, device=self.device)
                U_i = self.U[users_i]
                A = U_i.T @ U_i + reg_I * len(users_i)
                b = U_i.T @ r_centered
                self.V[i] = torch.linalg.solve(A, b)

        self.U = self.U.to(torch.float32)
        self.V = self.V.to(torch.float32)

    def _score(self, u, i):
        """Score = mu + b_u + b_i + U@V."""
        return self.global_mean + self.user_bias[u].item() + self.item_bias[i].item() + (self.U[u] * self.V[i]).sum().item()

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.global_mean
        u = self.user_idx[user_id]
        i = self.item_idx[item_id]
        pred = self._score(u, i)
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, user_indices, item_indices):
        """Batch predict on GPU. Returns numpy array."""
        import torch

        u_t = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        i_t = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        preds = self.global_mean + self.user_bias[u_t] + self.item_bias[i_t] + (self.U[u_t] * self.V[i_t]).sum(dim=1)
        out = np.clip(preds.cpu().numpy().astype(np.float64), 1.0, 5.0)
        out = np.nan_to_num(out, nan=self.global_mean, posinf=5.0, neginf=1.0)
        return out

    def _get_candidates(self, u, exclude, max_candidates=10000):
        """Get candidate items = union of co-occurring items, prioritized by popularity.
        If co-occurrence < max_candidates: pad with most-rated items.
        If co-occurrence > max_candidates: take top by global rating count.
        """
        rated = self.user_items[u] | exclude
        candidates = set()
        for i in self.user_items[u]:
            candidates.update(self.item_cooccur.get(i, set()))
        candidates -= rated

        if not candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            candidates = [i for i in top_popular if i not in rated][:max_candidates]
            return candidates

        cand_list = list(candidates)
        if len(cand_list) <= max_candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            pad = [i for i in top_popular if i not in rated and i not in candidates][: max_candidates - len(cand_list)]
            cand_list = cand_list + pad
        else:
            cand_list = sorted(cand_list, key=lambda i: -self.item_rating_counts[i])[:max_candidates]
        return cand_list

    def recommend_top_n(self, user_id, n=10, exclude_items=None):
        """Recommend top-N. Limits to candidate items (reachable via co-occurrence) for better precision."""
        if user_id not in self.user_idx:
            return []
        u = self.user_idx[user_id]
        exclude = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude.add(self.item_idx[iid])

        candidates = self._get_candidates(u, exclude)
        if not candidates:
            return []

        import torch
        c_t = torch.tensor(candidates, dtype=torch.long, device=self.device)
        scores = self.global_mean + self.user_bias[u] + self.item_bias[c_t] + (self.U[u] * self.V[c_t]).sum(dim=1)
        scores = scores.cpu().numpy()
        top_pos = np.argsort(-scores)[:n]
        return [self.rev_item[candidates[i]] for i in top_pos]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, max_candidates=10000):
        """Batch recommend with limited candidates per user."""
        import torch

        all_recs = []
        for b, u in enumerate(user_indices):
            candidates = self._get_candidates(u, exclude_sets[b], max_candidates)
            if not candidates:
                all_recs.append([self.rev_item[0]] * n)
                continue
            c_t = torch.tensor(candidates, dtype=torch.long, device=self.device)
            scores = self.global_mean + self.user_bias[u] + self.item_bias[c_t] + (self.U[u] * self.V[c_t]).sum(dim=1)
            scores = scores.cpu().numpy()
            top_pos = np.argsort(-scores)[:n]
            recs = [self.rev_item[candidates[i]] for i in top_pos]
            if len(recs) < n:
                recs += [self.rev_item[0]] * (n - len(recs))
            all_recs.append(recs)
        return all_recs


# --- Evaluation Metrics ---

def mae(predictions, actuals):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(predictions) - np.array(actuals)))


def rmse(predictions, actuals):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))


def precision_at_k(recommended, relevant, k=10):
    """Precision@k = |relevant ∩ recommended| / k."""
    rec_set = set(recommended[:k])
    return len(rec_set & set(relevant)) / k if k > 0 else 0


def recall_at_k(recommended, relevant, k=10):
    """Recall@k = |relevant ∩ recommended| / |relevant|."""
    if not relevant:
        return 0
    rec_set = set(recommended[:k])
    return len(rec_set & set(relevant)) / len(relevant)


def f_measure(precision, recall):
    """F-measure = 2 * P * R / (P + R)."""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def dcg_at_k(relevant, recommended, k=10):
    """Discounted Cumulative Gain at k."""
    rec_list = recommended[:k]
    dcg = 0
    for i, item in enumerate(rec_list):
        if item in relevant:
            rel = 1
            dcg += rel / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized DCG at k. Binary relevance: item in test set = 1."""
    rel_set = set(relevant)
    dcg = 0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG: all relevant items ranked first
    n_rel = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0
    return dcg / idcg


def evaluate_rating_prediction(model, test_data, batch_size=32768):
    """Evaluate rating prediction: MAE and RMSE. Uses batch predict on GPU when available."""
    actuals = np.array([float(r["overall"]) for r in test_data], dtype=np.float64)
    if hasattr(model, "predict_batch"):
        preds = []
        fallback_mean = float(getattr(model, "global_mean", 3.0))
        for i in tqdm(range(0, len(test_data), batch_size), desc="  Predicting ratings", unit=" batches"):
            batch = test_data[i : i + batch_size]
            u_ix, i_ix, valid_mask = [], [], []
            for r in batch:
                u = model.user_idx.get(r["reviewerID"])
                i = model.item_idx.get(r["asin"])
                v = u is not None and i is not None
                valid_mask.append(v)
                if v:
                    u_ix.append(u)
                    i_ix.append(i)
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


def compute_candidate_hit_rate(model, train_data, test_data, max_candidates=10000, min_train_ratings=5):
    """Diagnostic: % of test items present in candidate list before MF scoring.
    Low hit rate = candidate filter is bottleneck."""
    if not hasattr(model, "_get_candidates"):
        return None
    test_by_user = defaultdict(list)
    for r in test_data:
        test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    users_eval = [
        u for u in test_by_user
        if u in model.user_idx and test_by_user[u] and len(train_by_user[u]) >= min_train_ratings
    ]
    hits, total = 0, 0
    for u in users_eval[:500]:
        u_idx = model.user_idx[u]
        exclude = {model.item_idx[iid] for iid in train_by_user[u] if iid in model.item_idx}
        candidates = set(model._get_candidates(u_idx, exclude, max_candidates))
        for iid in test_by_user[u]:
            if iid in model.item_idx and model.item_idx[iid] in candidates:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0


def evaluate_recommendations(model, train_data, test_data, top_n=10, batch_size=512, min_train_ratings=5, max_candidates=10000):
    """Evaluate Top-N recommendations: Precision, Recall, F-measure, NDCG.
    Excludes users with fewer than min_train_ratings in train (cold-start filter)."""
    test_by_user = defaultdict(list)
    for r in test_data:
        test_by_user[r["reviewerID"]].append(r["asin"])
    train_by_user = defaultdict(set)
    for r in train_data:
        train_by_user[r["reviewerID"]].add(r["asin"])

    users_eval = [
        u for u in test_by_user
        if u in model.user_idx and test_by_user[u] and len(train_by_user[u]) >= min_train_ratings
    ]
    precisions, recalls, ndcgs = [], [], []

    if hasattr(model, "recommend_top_n_batch"):
        for i in tqdm(range(0, len(users_eval), batch_size), desc="  Top-N recommendations", unit=" batches"):
            batch_users = users_eval[i : i + batch_size]
            u_indices = [model.user_idx[u] for u in batch_users]
            exclude_sets = [
                {model.item_idx[iid] for iid in train_by_user[u] if iid in model.item_idx}
                for u in batch_users
            ]
            recs_batch = model.recommend_top_n_batch(u_indices, exclude_sets, n=top_n, max_candidates=max_candidates)
            for user, relevant, recommended in zip(batch_users, [test_by_user[u] for u in batch_users], recs_batch):
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
        "Precision": avg_p,
        "Recall": avg_r,
        "F-measure": f_measure(avg_p, avg_r),
        "NDCG": np.mean(ndcgs) if ndcgs else 0,
        "n_users_eval": len(users_eval),
    }
