"""Sentence-BERT similarity for review-text explanations (optional dependency)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from src.trustworthiness.text_profiles import review_snippet_from_record

logger = logging.getLogger(__name__)


class SentenceReviewProfileIndex:
    """
    Encode aggregated per-item review text with Sentence-BERT; score user--item
    cosine similarity (L2-normalized embeddings).
    """

    def __init__(
        self,
        train_records: list[dict],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        max_snippet: int = 400,
        max_parts_per_item: int = 12,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Sentence-BERT explanations. "
                "Install with: pip install sentence-transformers"
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = int(self.model.get_embedding_dimension())
        else:
            self._dim = int(self.model.get_sentence_embedding_dimension())

        parts_by_item: dict[str, list[str]] = defaultdict(list)
        for r in train_records:
            asin = r.get("asin")
            if not asin:
                continue
            snip = review_snippet_from_record(r, max_summary_len=200, max_review_len=max_snippet)
            if not snip:
                continue
            parts_by_item[asin].append(snip)

        self._asins: list[str] = []
        docs: list[str] = []
        for asin, parts in parts_by_item.items():
            merged = " ".join(parts[:max_parts_per_item])
            if len(merged) < 30:
                continue
            self._asins.append(asin)
            docs.append(merged)

        if len(docs) < 10:
            raise ValueError(
                "Not enough text to build Sentence-BERT index (need ~10 items with review text, "
                "summary, and/or metadata phrases)."
            )

        logger.info(
            "Encoding %d item text profiles with %s on %s…",
            len(docs),
            model_name,
            device,
        )
        self._item_emb = self.model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._asin_to_row = {a: i for i, a in enumerate(self._asins)}
        logger.info("SentenceReviewProfileIndex ready (%d items, dim=%d)", len(self._asins), self._dim)

    def user_profile_text(self, train_records: list[dict], user_id: str, max_reviews: int = 15) -> str:
        chunks: list[str] = []
        for r in train_records:
            if r.get("reviewerID") != user_id:
                continue
            snip = review_snippet_from_record(r, max_summary_len=200, max_review_len=500)
            if snip:
                chunks.append(snip)
            if len(chunks) >= max_reviews:
                break
        return " ".join(chunks)

    def cosine_user_item(self, train_records: list[dict], user_id: str, item_asin: str) -> float:
        profile = self.user_profile_text(train_records, user_id)
        if len(profile) < 20:
            return 0.0
        row = self._asin_to_row.get(item_asin)
        if row is None:
            return 0.0
        u = self.model.encode(
            [profile],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        v = self._item_emb[row]
        return float(max(0.0, min(1.0, np.dot(u, v))))
