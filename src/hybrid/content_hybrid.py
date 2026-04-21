"""Optional content-aware hybrid wrapper for ranking models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.evaluation.late_fusion import cf_position_scores


@dataclass(frozen=True)
class ContentHybridConfig:
    """Configuration for lightweight collaborative + content fusion."""

    enabled: bool = False
    alpha: float = 0.70
    cold_start_alpha: float = 0.35
    pool_size: int = 80
    cold_start_threshold: int = 5


class ContentHybridRanker:
    """
    Optional wrapper that reorders a base collaborative pool using content similarity.

    The base recommender remains unchanged. This wrapper only changes how the final
    Top-N list is derived from a candidate pool, which keeps the hybrid branch
    modular and easy to compare against the baseline.
    """

    def __init__(
        self,
        base_model,
        train_data: list[dict],
        text_index,
        *,
        config: ContentHybridConfig | None = None,
    ):
        self.base_model = base_model
        self.train_data = train_data
        self.text_index = text_index
        self.config = config or ContentHybridConfig()

        self.user_idx = getattr(base_model, "user_idx", {})
        self.item_idx = getattr(base_model, "item_idx", {})
        self.rev_user = getattr(base_model, "rev_user", None)
        self.rev_item = getattr(base_model, "rev_item", None)

        self.user_history: dict[str, set[str]] = defaultdict(set)
        item_counts: dict[str, int] = defaultdict(int)
        for row in train_data:
            user_id = row["reviewerID"]
            item_id = row["asin"]
            self.user_history[user_id].add(item_id)
            item_counts[item_id] += 1
        self.popularity_ranking = [
            item_id for item_id, _ in sorted(item_counts.items(), key=lambda row: (-row[1], row[0]))
        ]

    def _effective_alpha(self, user_id: str) -> float:
        n_history = len(self.user_history.get(user_id, set()))
        if n_history <= int(self.config.cold_start_threshold):
            return float(self.config.cold_start_alpha)
        return float(self.config.alpha)

    def _popularity_fallback(self, user_id: str, n: int, exclude_items=None) -> list[str]:
        exclude = set(exclude_items or [])
        exclude |= self.user_history.get(user_id, set())
        recs: list[str] = []
        for item_id in self.popularity_ranking:
            if item_id in exclude:
                continue
            recs.append(item_id)
            if len(recs) >= n:
                break
        return recs

    def recommend_top_n(self, user_id: str, n: int = 10, exclude_items=None) -> list[str]:
        if not self.config.enabled:
            return self.base_model.recommend_top_n(user_id, n=n, exclude_items=exclude_items)

        if user_id not in self.user_idx:
            return self._popularity_fallback(user_id, n, exclude_items=exclude_items)

        pool_size = max(int(self.config.pool_size), int(n))
        pool = self.base_model.recommend_top_n(user_id, n=pool_size, exclude_items=exclude_items)
        if not pool:
            return self._popularity_fallback(user_id, n, exclude_items=exclude_items)

        alpha = self._effective_alpha(user_id)
        cf_by_asin = cf_position_scores(pool)
        text_scores = self.text_index.cosine_user_items(self.train_data, user_id, pool)
        ranked = []
        for asin, text_sim in zip(pool, text_scores):
            c = float(cf_by_asin.get(asin, 0.0))
            t = float(max(0.0, min(1.0, text_sim)))
            fused = alpha * c + (1.0 - alpha) * t
            ranked.append((asin, fused, c, t))
        ranked.sort(key=lambda row: -row[1])
        return [row[0] for row in ranked[:n]]

    def recommend_top_n_batch(self, user_indices, exclude_sets, n=10, **kwargs):
        user_ids = [self.rev_user[u] for u in user_indices] if self.rev_user else []
        if not user_ids:
            return []
        all_recs = []
        for idx, user_id in enumerate(user_ids):
            exclude_items = None
            if idx < len(exclude_sets) and self.rev_item:
                exclude_items = {
                    self.rev_item[item_idx]
                    for item_idx in exclude_sets[idx]
                    if item_idx in self.rev_item
                }
            all_recs.append(self.recommend_top_n(user_id, n=n, exclude_items=exclude_items))
        return all_recs
