"""Item-based Collaborative Filtering using adjusted cosine similarity."""

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """
    Item-based CF built from scratch using NumPy.
    Similarity: adjusted cosine over co-rated users.
    Prediction: weighted k-NN deviation from item mean.
    """

    def __init__(self, k: int = 50):
        self.k = k
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.ratings = None
        self.item_means = None
        self.similarity = None
        self.item_raters = None

    def fit(self, data: list[dict]):
        logger.info("Building user/item indices...")
        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)

        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items", n_users, n_items)

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
            self.item_means[i] = item_sums[i] / item_counts[i] if item_counts[i] > 0 else 3.0

        self.item_raters = defaultdict(set)
        user_items = defaultdict(list)
        for (u, i), r in self.ratings.items():
            self.item_raters[i].add(u)
            user_items[u].append((i, r))

        pair_num = defaultdict(float)
        pair_den_i = defaultdict(float)
        pair_den_j = defaultdict(float)

        logger.info("Computing item-item similarities...")
        for u, rated in tqdm(user_items.items(), desc="  Similarity", unit=" users"):
            centered = [(i, r - self.item_means[i]) for i, r in rated]
            for (i, ci), (j, cj) in combinations(centered, 2):
                if i > j:
                    i, j = j, i
                    ci, cj = cj, ci
                pair_num[(i, j)] += ci * cj
                pair_den_i[(i, j)] += ci * ci
                pair_den_j[(i, j)] += cj * cj

        logger.info("Building similarity matrix...")
        self.similarity = defaultdict(dict)
        for (i, j), num in tqdm(pair_num.items(), desc="  Pairs", unit=" pairs"):
            den = np.sqrt(pair_den_i[(i, j)] * pair_den_j[(i, j)]) + 1e-9
            sim = max(0, num / den)
            self.similarity[i][j] = sim
            self.similarity[j][i] = sim

    def predict(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.item_means.mean() if self.item_means is not None else 3.0

        u = self.user_idx[user_id]
        i = self.item_idx[item_id]

        if (u, i) in self.ratings:
            return self.ratings[(u, i)]

        return self._predict_internal(u, i)

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        if user_id not in self.user_idx:
            return []

        u = self.user_idx[user_id]
        exclude = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude.add(self.item_idx[iid])

        rated = {j for (uid, j), _ in self.ratings.items() if uid == u}
        rated |= exclude

        candidates = []
        for i in range(len(self.item_idx)):
            if i in rated:
                continue
            pred = self._predict_internal(u, i)
            candidates.append((pred, self.rev_item[i]))

        candidates.sort(key=lambda x: -x[0])
        return [item_id for _, item_id in candidates[:n]]

    def _predict_internal(self, u: int, i: int) -> float:
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
