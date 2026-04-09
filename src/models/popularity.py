"""Popularity-based baseline recommender."""

import logging
from collections import defaultdict

import numpy as np

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class PopularityBaseline:
    """
    Non-personalized baseline: recommends globally most-popular items.
    Rating prediction: returns the item's average rating (or global mean).
    Top-N: ranks items by number of interactions in training set.
    """

    def __init__(self):
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.global_mean = 3.0
        self.item_means = None
        self.item_counts = None
        self.popularity_ranking = None
        self.user_train_items = None

    def fit(self, data: list[dict]):
        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items", len(self.user_idx), n_items)

        item_sums = defaultdict(float)
        item_counts = defaultdict(int)
        all_ratings = []

        self.user_train_items = defaultdict(set)
        for r in data:
            u = self.user_idx[r["reviewerID"]]
            i = self.item_idx[r["asin"]]
            rating = float(r["overall"])
            item_sums[i] += rating
            item_counts[i] += 1
            all_ratings.append(rating)
            self.user_train_items[u].add(i)

        self.global_mean = float(np.mean(all_ratings))

        self.item_means = np.full(n_items, self.global_mean)
        self.item_counts = np.zeros(n_items, dtype=np.int64)
        for i in range(n_items):
            if item_counts[i] > 0:
                self.item_means[i] = item_sums[i] / item_counts[i]
                self.item_counts[i] = item_counts[i]

        self.popularity_ranking = np.argsort(-self.item_counts)
        logger.info("Popularity baseline fitted. Global mean: %.3f", self.global_mean)

    def predict(self, user_id: str, item_id: str) -> float:
        if item_id in self.item_idx:
            return float(self.item_means[self.item_idx[item_id]])
        return self.global_mean

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        exclude_idx = set()
        if user_id in self.user_idx:
            exclude_idx = self.user_train_items.get(self.user_idx[user_id], set())
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_idx:
                    exclude_idx.add(self.item_idx[iid])

        recs = []
        for i in self.popularity_ranking:
            if i not in exclude_idx:
                recs.append(self.rev_item[i])
                if len(recs) >= n:
                    break
        return recs

    def get_popular_items(self, n: int = 20) -> list[dict]:
        """Return top-N popular items with their stats (for demo display)."""
        results = []
        for i in self.popularity_ranking[:n]:
            results.append({
                "item_id": self.rev_item[i],
                "avg_rating": float(self.item_means[i]),
                "num_ratings": int(self.item_counts[i]),
            })
        return results
