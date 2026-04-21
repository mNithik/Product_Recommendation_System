"""Item-support and similarity utilities for post-hoc recommendation explanations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class SimilarItemSupport:
    """Evidence that one item supports another recommended item."""

    item_id: str
    user_overlap: int
    similarity: float
    avg_rating: float


class ItemSimilarityIndex:
    """
    Lightweight explanation index based on interaction overlap.

    This is deliberately transparent and inexpensive. It is not used for model
    training; it exists only to support post-hoc explanations.
    """

    def __init__(self, train_records: list[dict]):
        self.user_history: dict[str, list[dict]] = defaultdict(list)
        self.item_users: dict[str, set[str]] = defaultdict(set)
        self.item_rating_sum: dict[str, float] = defaultdict(float)
        self.item_rating_count: dict[str, int] = defaultdict(int)

        for row in train_records:
            user_id = row["reviewerID"]
            item_id = row["asin"]
            rating = float(row["overall"])
            self.user_history[user_id].append({"item": item_id, "rating": rating})
            self.item_users[item_id].add(user_id)
            self.item_rating_sum[item_id] += rating
            self.item_rating_count[item_id] += 1

    def average_rating(self, item_id: str) -> float:
        count = self.item_rating_count.get(item_id, 0)
        if count <= 0:
            return 0.0
        return self.item_rating_sum[item_id] / count

    def user_history_items(self, user_id: str) -> list[dict]:
        return list(self.user_history.get(user_id, []))

    def overlap_count(self, left_item: str, right_item: str) -> int:
        return len(self.item_users.get(left_item, set()) & self.item_users.get(right_item, set()))

    def item_similarity(self, left_item: str, right_item: str) -> float:
        left_users = self.item_users.get(left_item, set())
        right_users = self.item_users.get(right_item, set())
        if not left_users or not right_users:
            return 0.0
        union = left_users | right_users
        if not union:
            return 0.0
        return len(left_users & right_users) / len(union)

    def supporting_history_items(
        self,
        user_id: str,
        candidate_item: str,
        *,
        min_rating: float = 4.0,
        top_k: int = 3,
    ) -> list[SimilarItemSupport]:
        supports: list[SimilarItemSupport] = []
        for row in self.user_history.get(user_id, []):
            history_item = row["item"]
            rating = float(row["rating"])
            if rating < min_rating or history_item == candidate_item:
                continue
            overlap = self.overlap_count(history_item, candidate_item)
            if overlap <= 0:
                continue
            supports.append(
                SimilarItemSupport(
                    item_id=history_item,
                    user_overlap=overlap,
                    similarity=self.item_similarity(history_item, candidate_item),
                    avg_rating=rating,
                )
            )

        supports.sort(
            key=lambda row: (-row.user_overlap, -row.similarity, -row.avg_rating, row.item_id)
        )
        return supports[:top_k]

    def similar_items(
        self,
        item_id: str,
        *,
        exclude_items: set[str] | None = None,
        top_k: int = 3,
        min_overlap: int = 1,
    ) -> list[SimilarItemSupport]:
        exclude = exclude_items or set()
        supports: list[SimilarItemSupport] = []
        for other_item in self.item_users.keys():
            if other_item == item_id or other_item in exclude:
                continue
            overlap = self.overlap_count(item_id, other_item)
            if overlap < min_overlap:
                continue
            supports.append(
                SimilarItemSupport(
                    item_id=other_item,
                    user_overlap=overlap,
                    similarity=self.item_similarity(item_id, other_item),
                    avg_rating=self.average_rating(other_item),
                )
            )

        supports.sort(
            key=lambda row: (-row.similarity, -row.user_overlap, -row.avg_rating, row.item_id)
        )
        return supports[:top_k]
