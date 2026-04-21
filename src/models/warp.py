"""WARP (Weighted Approximate-Rank Pairwise) loss — vectorized PyTorch implementation.

WARP focuses optimization on the top of the ranked list by sampling negatives
until a rank-violating item is found, then weighting the update by an
approximation of the item's rank.
"""

import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class WARPModel:
    """
    WARP-loss MF built from scratch with PyTorch.
    Fully vectorized: batched negative sampling + rank-weight computation on GPU.
    """

    def __init__(self, n_factors: int = 64, n_epochs: int = 20,
                 lr: float = 0.01, reg: float = 1e-4,
                 pos_threshold: float = 4.0, max_trials: int = 20,
                 batch_size: int = 4096, device: str | None = None):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.pos_threshold = float(pos_threshold)
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.U = None
        self.V = None
        self.user_items = None

    def fit(self, data: list[dict]):
        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items | Device: %s", n_users, n_items, self.device)

        user_pos = defaultdict(set)
        user_all = defaultdict(set)
        for r in data:
            u = self.user_idx[r["reviewerID"]]
            i = self.item_idx[r["asin"]]
            user_all[u].add(i)
            if float(r["overall"]) >= self.pos_threshold:
                user_pos[u].add(i)

        self.user_items = dict(user_all)

        pairs = []
        for u, items in user_pos.items():
            for i in items:
                pairs.append((u, i))
        pairs = np.array(pairs, dtype=np.int64)
        n_pairs = len(pairs)

        logger.info("WARP positives (rating >= %.1f): %d pairs from %d users",
                     self.pos_threshold, n_pairs, len(user_pos))

        user_pos_list = dict(user_pos)

        self.U = torch.randn(n_users, self.n_factors, device=self.device) * 0.01
        self.V = torch.randn(n_items, self.n_factors, device=self.device) * 0.01
        self.U.requires_grad_(True)
        self.V.requires_grad_(True)

        optimizer = torch.optim.Adam([self.U, self.V], lr=self.lr, weight_decay=self.reg)
        rng = np.random.default_rng(42)

        rank_weights = torch.zeros(self.max_trials + 1, device=self.device)
        for t in range(1, self.max_trials + 1):
            approx_rank = n_items // t
            rank_weights[t] = sum(1.0 / j for j in range(1, approx_rank + 1))

        logger.info("Training WARP (%d epochs, batch=%d)...", self.n_epochs, self.batch_size)
        for epoch in tqdm(range(self.n_epochs), desc="  WARP Epochs"):
            rng.shuffle(pairs)
            epoch_loss = 0.0

            for start in range(0, n_pairs, self.batch_size):
                batch = pairs[start: start + self.batch_size]
                B = len(batch)
                u_idx = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                i_idx = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)

                u_emb = self.U[u_idx]
                i_emb = self.V[i_idx]
                s_pos = (u_emb * i_emb).sum(dim=1)

                neg_candidates = torch.randint(0, n_items, (B, self.max_trials), device=self.device)
                neg_np = neg_candidates.cpu().numpy()
                batch_u = batch[:, 0]

                first_neg_trial = np.full(B, self.max_trials, dtype=np.int64)
                for row in range(B):
                    pos_set = user_pos_list.get(int(batch_u[row]), set())
                    for t in range(self.max_trials):
                        if int(neg_np[row, t]) not in pos_set:
                            first_neg_trial[row] = t
                            break

                found_neg = first_neg_trial < self.max_trials
                if not found_neg.any():
                    continue

                found_mask = torch.tensor(found_neg, dtype=torch.bool, device=self.device)
                trial_t = torch.tensor(first_neg_trial, dtype=torch.long, device=self.device)
                neg_item_idx = neg_candidates[torch.arange(B, device=self.device), trial_t]

                j_emb = self.V[neg_item_idx]
                s_neg = (u_emb * j_emb).sum(dim=1)

                violators = found_mask & (s_neg > s_pos - 1.0)
                if not violators.any():
                    continue

                trial_nums = trial_t[violators] + 1
                weights = rank_weights[trial_nums]
                margins = torch.clamp(1.0 - s_pos[violators] + s_neg[violators], min=0.0)
                loss = (weights * margins).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self.U = self.U.detach()
        self.V = self.V.detach()
        logger.info("WARP training complete.")

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        if user_id not in self.user_idx:
            return []
        import torch
        u = self.user_idx[user_id]
        scores = (self.U[u] @ self.V.T)

        exclude_idx = self.user_items.get(u, set())
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude_idx = exclude_idx | {self.item_idx[iid]}
        for idx in exclude_idx:
            scores[idx] = float("-inf")

        k = min(int(n), int(scores.numel()))
        top_pos = torch.topk(scores, k=k, largest=True).indices.tolist()
        return [self.rev_item[i] for i in top_pos]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, **kwargs):
        import torch
        if not user_indices:
            return []

        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        scores = self.U[user_tensor] @ self.V.T

        for row, u in enumerate(user_indices):
            exclude_idx = self.user_items.get(u, set()) | exclude_sets[row]
            if exclude_idx:
                idx_tensor = torch.tensor(list(exclude_idx), dtype=torch.long, device=self.device)
                scores[row, idx_tensor] = float("-inf")

        k = min(int(n), int(scores.shape[1]))
        top_indices = torch.topk(scores, k=k, dim=1, largest=True).indices.cpu().tolist()

        all_recs = []
        for row in top_indices:
            recs = [self.rev_item[i] for i in row]
            if len(recs) < n:
                recs += [self.rev_item[0]] * (n - len(recs))
            all_recs.append(recs)
        return all_recs
