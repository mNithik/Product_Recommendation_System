"""Matrix Factorization with user/item biases, trained with ALS on GPU."""

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class MatrixFactorizationGPU:
    """
    ALS-based Matrix Factorization with biases.
    Predictions: mu + b_u + b_i + U[u] @ V[i]^T
    Supports limited candidates for Top-N via item co-occurrence.
    """

    def __init__(self, n_factors: int = 64, reg: float = 0.1,
                 n_epochs: int = 10, device: str | None = None):
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

    def fit(self, data: list[dict]):
        import torch

        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items | Device: %s", n_users, n_items, self.device)

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

        user_means = np.array([
            np.mean([x[1] for x in user_items[u]]) if user_items[u] else self.global_mean
            for u in range(n_users)
        ], dtype=np.float32)
        item_means = np.array([
            np.mean([x[1] for x in item_users[i]]) if item_users[i] else self.global_mean
            for i in range(n_items)
        ], dtype=np.float32)
        self.user_bias = torch.tensor(user_means - self.global_mean, device=self.device)
        self.item_bias = torch.tensor(item_means - self.global_mean, device=self.device)

        ratings_centered = ratings_arr - self.global_mean \
            - self.user_bias.cpu().numpy()[users_arr] \
            - self.item_bias.cpu().numpy()[items_arr]
        ratings_centered = ratings_centered.astype(np.float32)

        self.V = torch.randn(n_items, self.n_factors, device=self.device) * 0.01
        self.U = torch.zeros(n_users, self.n_factors, device=self.device)

        logger.info("Building candidate sets (item co-occurrence)...")
        self.item_cooccur = defaultdict(set)
        for u, rated in enumerate(user_items):
            items_u = [x[0] for x in rated]
            for i, j in combinations(items_u, 2):
                self.item_cooccur[i].add(j)
                self.item_cooccur[j].add(i)

        logger.info("Training ALS with biases on %s...", self.device)
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
        return self.global_mean + self.user_bias[u].item() + self.item_bias[i].item() \
            + (self.U[u] * self.V[i]).sum().item()

    def predict(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.global_mean
        u = self.user_idx[user_id]
        i = self.item_idx[item_id]
        return float(np.clip(self._score(u, i), 1.0, 5.0))

    def predict_batch(self, user_indices, item_indices):
        import torch
        u_t = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        i_t = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        preds = self.global_mean + self.user_bias[u_t] + self.item_bias[i_t] \
            + (self.U[u_t] * self.V[i_t]).sum(dim=1)
        out = np.clip(preds.cpu().numpy().astype(np.float64), 1.0, 5.0)
        return np.nan_to_num(out, nan=self.global_mean, posinf=5.0, neginf=1.0)

    def _get_candidates(self, u, exclude, max_candidates=10000, min_item_ratings=0):
        rated = self.user_items[u] | exclude
        candidates = set()
        for i in self.user_items[u]:
            candidates.update(self.item_cooccur.get(i, set()))
        candidates -= rated

        if min_item_ratings > 0:
            candidates = {c for c in candidates if self.item_rating_counts[c] >= min_item_ratings}

        if not candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            return [i for i in top_popular
                    if i not in rated and self.item_rating_counts[i] >= min_item_ratings][:max_candidates]

        cand_list = list(candidates)
        if len(cand_list) <= max_candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            pad = [i for i in top_popular
                   if i not in rated and i not in candidates
                   and self.item_rating_counts[i] >= min_item_ratings][:max_candidates - len(cand_list)]
            cand_list = cand_list + pad
        else:
            cand_list = sorted(cand_list, key=lambda i: -self.item_rating_counts[i])[:max_candidates]
        return cand_list

    def _rank_scores(self, u, c_t):
        """Personalized ranking: dot product with inverse popularity weighting."""
        import torch
        raw = (self.U[u] * self.V[c_t]).sum(dim=1)
        pop = torch.tensor(
            self.item_rating_counts[c_t.cpu().numpy()],
            dtype=torch.float32, device=self.device,
        )
        penalty = torch.log2(2.0 + pop)
        return (raw / penalty).cpu().numpy()

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        if user_id not in self.user_idx:
            return []
        import torch
        u = self.user_idx[user_id]
        exclude = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude.add(self.item_idx[iid])
        candidates = self._get_candidates(u, exclude)
        if not candidates:
            return []
        c_t = torch.tensor(candidates, dtype=torch.long, device=self.device)
        scores = self._rank_scores(u, c_t)
        top_pos = np.argsort(-scores)[:n]
        return [self.rev_item[candidates[i]] for i in top_pos]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10,
                              max_candidates=10000, min_item_ratings=0):
        import torch
        candidate_lists = [
            self._get_candidates(u, exclude_sets[b], max_candidates, min_item_ratings=min_item_ratings)
            for b, u in enumerate(user_indices)
        ]
        max_len = max((len(c) for c in candidate_lists), default=0)
        if max_len == 0:
            return [[self.rev_item[0]] * n for _ in user_indices]

        pad_item = 0
        cand_tensor = torch.full(
            (len(user_indices), max_len),
            fill_value=pad_item,
            dtype=torch.long,
            device=self.device,
        )
        valid_mask = torch.zeros((len(user_indices), max_len), dtype=torch.bool, device=self.device)

        for row, candidates in enumerate(candidate_lists):
            if not candidates:
                continue
            length = len(candidates)
            cand_tensor[row, :length] = torch.tensor(candidates, dtype=torch.long, device=self.device)
            valid_mask[row, :length] = True

        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        user_emb = self.U[user_tensor].unsqueeze(1)
        item_emb = self.V[cand_tensor]
        raw = (user_emb * item_emb).sum(dim=2)
        pop = torch.tensor(
            self.item_rating_counts[cand_tensor.cpu().numpy()],
            dtype=torch.float32,
            device=self.device,
        )
        penalty = torch.log2(2.0 + pop)
        scores = (raw / penalty).masked_fill(~valid_mask, float("-inf"))

        k = min(int(n), int(scores.shape[1]))
        top_positions = torch.topk(scores, k=k, dim=1, largest=True).indices.cpu().tolist()

        all_recs = []
        for row, positions in enumerate(top_positions):
            candidates = candidate_lists[row]
            recs = [self.rev_item[candidates[pos]] for pos in positions if pos < len(candidates)]
            if len(recs) < n:
                recs += [self.rev_item[0]] * (n - len(recs))
            all_recs.append(recs)
        return all_recs
