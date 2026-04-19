"""Optional Sentence-BERT explanation index."""

import pytest

pytest.importorskip("sentence_transformers")

from src.trustworthiness.sentence_embeddings import SentenceReviewProfileIndex


def test_sentence_index_cosine():
    rows = []
    for i in range(10):
        rows.append(
            {
                "reviewerID": "U1",
                "asin": f"B{i:05d}",
                "overall": 5,
                "reviewText": f"craft supplies sewing painting item {i}",
            }
        )
    rows.append(
        {
            "reviewerID": "U2",
            "asin": "B00005",
            "overall": 4,
            "reviewText": "yarn knitting hobby",
        }
    )
    idx = SentenceReviewProfileIndex(rows, batch_size=4)
    s = idx.cosine_user_item(rows, "U1", "B00000")
    assert 0.0 <= s <= 1.0
    assert len(idx._asins) == 10
