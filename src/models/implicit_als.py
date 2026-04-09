"""Implicit ALS (Hu-Koren-Volinsky) — weighted matrix factorization for implicit feedback."""

import logging

import numpy as np
from scipy.sparse import csr_matrix

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class ImplicitALSRanker:
    """
    Implicit ALS using the `implicit` library.
    Treats interactions as confidence-weighted implicit feedback.
    Rating >= threshold -> positive with confidence proportional to rating.
    """

    def __init__(self, n_factors: int = 64, n_epochs: int = 15,
                 reg: float = 0.01, pos_threshold: float = 4.0):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.pos_threshold = float(pos_threshold)
        self.model = None
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.user_items_csr = None
        self.all_user_items_csr = None

    def fit(self, data: list[dict]):
        from implicit.als import AlternatingLeastSquares

        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items", n_users, n_items)

        rows, cols, vals = [], [], []
        all_rows, all_cols = [], []
        for r in data:
            u = self.user_idx[r["reviewerID"]]
            i = self.item_idx[r["asin"]]
            rating = float(r["overall"])
            all_rows.append(u)
            all_cols.append(i)
            if rating >= self.pos_threshold:
                rows.append(u)
                cols.append(i)
                vals.append(rating)

        logger.info("Positive interactions (rating >= %.1f): %d", self.pos_threshold, len(rows))

        confidence = np.array(vals, dtype=np.float32)
        self.user_items_csr = csr_matrix((confidence, (rows, cols)), shape=(n_users, n_items))

        all_vals = np.ones(len(all_rows), dtype=np.float32)
        self.all_user_items_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_users, n_items))

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.n_epochs,
            regularization=self.reg,
            random_state=42,
        )
        logger.info("Training Implicit ALS (%d epochs, %d factors)...",
                     self.n_epochs, self.n_factors)
        self.model.fit(self.user_items_csr, show_progress=True)

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        if user_id not in self.user_idx:
            return []
        u = self.user_idx[user_id]
        ids, _ = self.model.recommend(
            u, self.all_user_items_csr[u], N=n, filter_already_liked_items=True,
        )
        return [self.rev_item[i] for i in ids]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, **kwargs):
        all_recs = []
        for u in user_indices:
            ids, _ = self.model.recommend(
                u, self.all_user_items_csr[u], N=n, filter_already_liked_items=True,
            )
            recs = [self.rev_item[i] for i in ids]
            if len(recs) < n:
                recs += [self.rev_item[0]] * (n - len(recs))
            all_recs.append(recs)
        return all_recs
