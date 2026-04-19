"""Lightweight review-text profiles for similarity-based explanations (TF-IDF).

This is *not* a full personalized transformer explainer (cf. ACL 2021 work), but it
uses the same modality (review language) to surface content overlap between a user
profile and a candidate item in a transparent, inexpensive way.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _clean(text: str, max_len: int) -> str:
    t = _WS.sub(" ", (text or "").strip())
    return t[:max_len]


def _overall_phrase(overall) -> str:
    if overall is None:
        return ""
    try:
        x = float(overall)
    except (TypeError, ValueError):
        return ""
    if not (1.0 <= x <= 5.0):
        return ""
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))} star rating"
    return f"{x:.1f} star rating"


def _verified_phrase(verified) -> str:
    if verified is True:
        return "verified purchase"
    if verified is False:
        return "unverified review"
    if isinstance(verified, str):
        low = verified.strip().lower()
        if low in ("true", "1", "yes", "y"):
            return "verified purchase"
        if low in ("false", "0", "no", "n"):
            return "unverified review"
    return ""


def _style_phrase(style) -> str:
    if not style or not isinstance(style, dict):
        return ""
    parts: list[str] = []
    for k in sorted(style.keys()):
        raw_k = str(k).strip().rstrip(":").strip()
        v = style.get(k)
        if v is None:
            continue
        raw_v = _WS.sub(" ", str(v).strip())
        if not raw_v:
            continue
        parts.append(f"{raw_k} {raw_v}".strip())
    if not parts:
        return ""
    return " ".join(parts).strip()


def review_snippet_from_record(
    r: dict,
    max_summary_len: int = 200,
    max_review_len: int = 400,
    max_meta_len: int = 120,
) -> str:
    """
    Build a text snippet for similarity/explanation: optional ``overall``, ``verified``,
    and ``style`` (concatenated as short phrases), then ``summary`` and ``reviewText``.
    Returns empty string if nothing usable is present.
    """
    meta_bits: list[str] = []
    op = _overall_phrase(r.get("overall"))
    if op:
        meta_bits.append(op)
    vp = _verified_phrase(r.get("verified"))
    if vp:
        meta_bits.append(vp)
    sp = _style_phrase(r.get("style"))
    if sp:
        meta_bits.append(sp)

    pieces: list[str] = []
    meta = " ".join(meta_bits).strip()
    if meta:
        pieces.append(_clean(meta, max_meta_len))

    s = (r.get("summary") or "").strip()
    t = (r.get("reviewText") or "").strip()
    if s:
        pieces.append(_clean(s, max_summary_len))
    if t:
        pieces.append(_clean(t, max_review_len))
    return " ".join(pieces).strip()


class ReviewTextProfileIndex:
    """
    Aggregate review snippets per item, fit TF-IDF, support user/item cosine scores.
    """

    def __init__(
        self,
        train_records: list[dict],
        max_features: int = 4000,
        min_df: int = 2,
        max_snippet: int = 400,
        max_parts_per_item: int = 12,
    ):
        self._asin_to_row: dict[str, int] = {}
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words="english",
            ngram_range=(1, 2),
        )

        parts_by_item: dict[str, list[str]] = defaultdict(list)
        for r in train_records:
            asin = r.get("asin")
            if not asin:
                continue
            snip = review_snippet_from_record(r, max_summary_len=200, max_review_len=max_snippet)
            if not snip:
                continue
            parts_by_item[asin].append(snip)

        item_ids: list[str] = []
        docs: list[str] = []
        for asin, parts in parts_by_item.items():
            merged = " ".join(parts[:max_parts_per_item])
            if len(merged) < 30:
                continue
            self._asin_to_row[asin] = len(item_ids)
            item_ids.append(asin)
            docs.append(merged)

        if len(docs) < 10:
            raise ValueError(
                "Not enough text to build TF-IDF index (need ~10 items with review text, "
                "summary, and/or metadata phrases)."
            )

        self._item_matrix = self._vectorizer.fit_transform(docs)
        self._item_ids = item_ids
        logger.info("ReviewTextProfileIndex: %d items with text", len(item_ids))

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
        u = self._vectorizer.transform([profile])
        v = self._item_matrix[row]
        sim = cosine_similarity(u, v)[0, 0]
        return float(max(0.0, min(1.0, sim)))
