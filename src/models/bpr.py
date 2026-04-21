"""BPR (Bayesian Personalized Ranking) Matrix Factorization — custom PyTorch implementation."""

import itertools
import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class BPRMatrixFactorization:
    """
    MF trained with BPR pairwise ranking loss.
    Optimized for ranking: positive items should score higher than negatives.
    Use for Top-N recommendation, NOT explicit rating prediction.
    """

    def __init__(self, n_factors: int = 128, reg: float = 0.01, n_epochs: int = 20,
                 lr: float = 0.01, pos_threshold: float = 4.0,
                 train_max_candidates: int = 5000, device: str | None = None):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.lr = lr
        self.pos_threshold = float(pos_threshold)
        self.train_max_candidates = int(train_max_candidates)
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.U = None
        self.V = None
        self.user_items = None
        self.user_pos_items = None
        self.item_cooccur = None
        self.item_rating_counts = None

    def fit(self, data: list[dict]):
        import torch

        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items | Device: %s", n_users, n_items, self.device)

        users_arr = np.array([self.user_idx[r["reviewerID"]] for r in data], dtype=np.int64)
        items_arr = np.array([self.item_idx[r["asin"]] for r in data], dtype=np.int64)
        ratings_arr = np.array([float(r["overall"]) for r in data], dtype=np.float32)

        user_items = [[] for _ in range(n_users)]
        user_pos = [[] for _ in range(n_users)]
        item_users = [[] for _ in range(n_items)]
        for u, i, r in zip(users_arr, items_arr, ratings_arr):
            user_items[u].append(i)
            item_users[i].append(u)
            if r >= self.pos_threshold:
                user_pos[u].append(i)

        self.user_items = [set(x) for x in user_items]
        self.user_pos_items = [set(x) for x in user_pos]
        user_items_list = [list(s) for s in self.user_items]
        self.item_rating_counts = np.array([len(item_users[i]) for i in range(n_items)])

        logger.info("Building candidate sets (item co-occurrence)...")
        self.item_cooccur = defaultdict(set)
        for u, rated in enumerate(user_items):
            items_u = list(rated)
            for i, j in combinations(items_u, 2):
                self.item_cooccur[i].add(j)
                self.item_cooccur[j].add(i)

        n_pos_pairs = sum(len(s) for s in self.user_pos_items)
        n_pos_users = sum(1 for s in self.user_pos_items if s)
        logger.info("Positives for BPR: rating >= %.1f | %d interactions from %d users",
                     self.pos_threshold, n_pos_pairs, n_pos_users)

        def _random_from_set(s, rng, max_skip=32):
            if not s:
                return None
            steps = int(rng.integers(0, min(max_skip, max(len(s) - 1, 0)) + 1))
            return next(itertools.islice(iter(s), steps, None))

        def _sample_hard_negative(u, rng):
            seed_items = user_items_list[u]
            if not seed_items:
                return int(rng.integers(0, n_items))
            pos = self.user_pos_items[u]
            for _ in range(30):
                seed = seed_items[int(rng.integers(0, len(seed_items)))]
                neigh = self.item_cooccur.get(seed)
                if not neigh:
                    continue
                j = _random_from_set(neigh, rng)
                if j is None:
                    continue
                if j not in pos:
                    return int(j)
            for _ in range(200):
                j = int(rng.integers(0, n_items))
                if j not in pos:
                    return j
            return int(rng.integers(0, n_items))

        self.U = torch.randn(n_users, self.n_factors, device=self.device)
        self.U.requires_grad_(True)
        self.U.data.mul_(0.01)
        self.V = torch.randn(n_items, self.n_factors, device=self.device)
        self.V.requires_grad_(True)
        self.V.data.mul_(0.01)

        pairs = []
        for u in range(n_users):
            for i in self.user_pos_items[u]:
                pairs.append((u, i))
        pairs = np.array(pairs, dtype=np.int64)

        logger.info("Training BPR loss on %s...", self.device)
        n_pairs = len(pairs)
        rng = np.random.default_rng(42)
        optimizer = torch.optim.Adam([self.U, self.V], lr=self.lr, weight_decay=self.reg)

        for epoch in tqdm(range(self.n_epochs), desc="  Epochs"):
            rng.shuffle(pairs)
            batch_size = 4096
            for start in range(0, n_pairs, batch_size):
                optimizer.zero_grad()
                batch = pairs[start: start + batch_size]
                u_batch = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                i_batch = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                j_batch = []
                for u, i in batch:
                    j_batch.append(_sample_hard_negative(int(u), rng))
                j_batch = torch.tensor(j_batch, dtype=torch.long, device=self.device)

                U_b = self.U[u_batch]
                V_i = self.V[i_batch]
                V_j = self.V[j_batch]
                x_ui = (U_b * V_i).sum(dim=1)
                x_uj = (U_b * V_j).sum(dim=1)
                diff = x_ui - x_uj
                loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
                loss.backward()
                optimizer.step()

        self.U = self.U.detach().to(torch.float32)
        self.V = self.V.detach().to(torch.float32)

    def _get_candidates(self, u, exclude, max_candidates=10000):
        rated = self.user_items[u] | exclude
        candidates = set()
        for i in self.user_items[u]:
            candidates.update(self.item_cooccur.get(i, set()))
        candidates -= rated

        if not candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            return [i for i in top_popular if i not in rated][:max_candidates]

        cand_list = list(candidates)
        if len(cand_list) <= max_candidates:
            top_popular = np.argsort(-self.item_rating_counts)
            pad = [i for i in top_popular
                   if i not in rated and i not in candidates][:max_candidates - len(cand_list)]
            cand_list = cand_list + pad
        else:
            cand_list = sorted(cand_list, key=lambda i: -self.item_rating_counts[i])[:max_candidates]
        return cand_list

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
        scores = (self.U[u] * self.V[c_t]).sum(dim=1)
        k = min(int(n), int(scores.numel()))
        if k <= 0:
            return []
        top_pos = torch.topk(scores, k=k, largest=True).indices.tolist()
        recs = [self.rev_item[candidates[i]] for i in top_pos]
        if len(recs) < n:
            recs += [self.rev_item[0]] * (n - len(recs))
        return recs

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, max_candidates=10000, **kwargs):
        import torch
        candidate_lists = [
            self._get_candidates(u, exclude_sets[b], max_candidates)
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
        scores = (user_emb * item_emb).sum(dim=2)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

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
