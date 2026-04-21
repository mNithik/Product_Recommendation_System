"""BPR ranker via the `implicit` library — optimized C++/CUDA backend."""

import logging
import platform

import numpy as np
from scipy.sparse import csr_matrix

from src.utils.data_loader import build_index

logger = logging.getLogger(__name__)


class ImplicitBPRRanker:
    """
    Ranking model using implicit library's BPR implementation.
    Treats rating >= threshold as positive (binary).
    Scores all items natively — no manual candidate set needed.
    """

    def __init__(self, n_factors: int = 128, n_epochs: int = 30,
                 lr: float = 0.01, reg: float = 0.01, pos_threshold: float = 4.0,
                 use_gpu: bool = False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.pos_threshold = float(pos_threshold)
        self.use_gpu = bool(use_gpu)
        self.model = None
        self.user_idx = None
        self.item_idx = None
        self.rev_user = None
        self.rev_item = None
        self.user_items_csr = None

    def fit(self, data: list[dict]):
        from implicit.bpr import BayesianPersonalizedRanking

        self.user_idx, self.item_idx, self.rev_user, self.rev_item = build_index(data)
        n_users = len(self.user_idx)
        n_items = len(self.item_idx)
        logger.info("%d users, %d items", n_users, n_items)

        pos_rows, pos_cols = [], []
        all_rows, all_cols = [], []
        for r in data:
            u = self.user_idx[r["reviewerID"]]
            i = self.item_idx[r["asin"]]
            all_rows.append(u)
            all_cols.append(i)
            if float(r["overall"]) >= self.pos_threshold:
                pos_rows.append(u)
                pos_cols.append(i)

        logger.info("Positive interactions (rating >= %.1f): %d", self.pos_threshold, len(pos_rows))

        pos_vals = np.ones(len(pos_rows), dtype=np.float32)
        self.user_items_csr = csr_matrix((pos_vals, (pos_rows, pos_cols)), shape=(n_users, n_items))

        all_vals = np.ones(len(all_rows), dtype=np.float32)
        self.all_user_items_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_users, n_items))

        model_kwargs = dict(
            factors=self.n_factors,
            iterations=self.n_epochs,
            learning_rate=self.lr,
            regularization=self.reg,
            random_state=42,
        )
        requested_gpu = self.use_gpu
        if requested_gpu and platform.system().lower() == "windows":
            logger.warning(
                "Implicit BPR GPU requested on Windows, but implicit 0.7.2 does not build CUDA there; using CPU."
            )
            self.use_gpu = False
        if self.use_gpu:
            model_kwargs["use_gpu"] = True

        try:
            self.model = BayesianPersonalizedRanking(**model_kwargs)
            logger.info(
                "Training BPR (implicit library, %d epochs, %d factors, gpu=%s)...",
                self.n_epochs, self.n_factors, str(self.use_gpu),
            )
            self.model.fit(self.user_items_csr, show_progress=True)
        except Exception as exc:
            if not self.use_gpu:
                raise
            logger.warning("Implicit BPR GPU path failed; retrying on CPU: %s", exc)
            self.model = BayesianPersonalizedRanking(
                factors=self.n_factors,
                iterations=self.n_epochs,
                learning_rate=self.lr,
                regularization=self.reg,
                random_state=42,
            )
            logger.info("Training BPR (implicit library, %d epochs, %d factors, gpu=False)...",
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

    def recommend_top_n_scored(self, user_id: str, n: int = 10, exclude_items=None):
        if user_id not in self.user_idx:
            return []
        u = self.user_idx[user_id]
        ids, scores = self.model.recommend(
            u, self.all_user_items_csr[u], N=n, filter_already_liked_items=True,
        )
        return [(self.rev_item[int(i)], float(s)) for i, s in zip(ids, scores)]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, **kwargs):
        if not user_indices:
            return []

        try:
            user_arr = np.asarray(user_indices, dtype=np.int32)
            user_items = self.all_user_items_csr[user_arr]
            ids, _ = self.model.recommend(
                user_arr,
                user_items,
                N=n,
                filter_already_liked_items=True,
            )

            if isinstance(ids, np.ndarray) and ids.ndim == 1:
                ids = ids.reshape(1, -1)

            all_recs: list[list[str]] = []
            for row in ids:
                recs = [self.rev_item[int(i)] for i in row[:n]]
                if len(recs) < n:
                    recs += [self.rev_item[0]] * (n - len(recs))
                all_recs.append(recs)
            return all_recs
        except Exception as exc:
            logger.warning("Implicit BPR batched recommend failed; falling back to per-user loop: %s", exc)
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

    def recommend_top_n_batch_scored(self, user_indices, exclude_sets, n=10, **kwargs):
        if not user_indices:
            return []

        try:
            user_arr = np.asarray(user_indices, dtype=np.int32)
            user_items = self.all_user_items_csr[user_arr]
            ids, scores = self.model.recommend(
                user_arr,
                user_items,
                N=n,
                filter_already_liked_items=True,
            )

            if isinstance(ids, np.ndarray) and ids.ndim == 1:
                ids = ids.reshape(1, -1)
                scores = np.asarray(scores).reshape(1, -1)

            rows = []
            for row_ids, row_scores in zip(ids, scores):
                recs = [
                    (self.rev_item[int(i)], float(s))
                    for i, s in zip(row_ids[:n], row_scores[:n])
                ]
                rows.append(recs)
            return rows
        except Exception as exc:
            logger.warning("Implicit BPR batched scored recommend failed; falling back to per-user loop: %s", exc)
            rows = []
            for u in user_indices:
                ids, scores = self.model.recommend(
                    u, self.all_user_items_csr[u], N=n, filter_already_liked_items=True,
                )
                rows.append([(self.rev_item[int(i)], float(s)) for i, s in zip(ids, scores)])
            return rows
