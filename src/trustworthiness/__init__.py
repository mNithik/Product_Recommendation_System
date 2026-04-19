"""Trustworthiness-oriented helpers (scoped explainability / audit tools)."""

from .text_profiles import ReviewTextProfileIndex

__all__ = ["ReviewTextProfileIndex"]

# SentenceReviewProfileIndex is optional (requires sentence-transformers).
