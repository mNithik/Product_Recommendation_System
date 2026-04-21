"""Tests for structured recommendation explanations."""

from src.explainability import ItemSimilarityIndex, explain_recommendation


def _sample_rows():
    return [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U4", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U4", "asin": "A3", "overall": 4.0},
    ]


def test_item_similarity_index_finds_supporting_history_items():
    index = ItemSimilarityIndex(_sample_rows())
    supports = index.supporting_history_items("U1", "A3", top_k=2)

    assert supports
    assert supports[0].item_id in {"A1", "A2"}
    assert supports[0].user_overlap >= 1
    assert 0.0 <= supports[0].similarity <= 1.0


def test_explanation_has_structured_output_and_confidence():
    index = ItemSimilarityIndex(_sample_rows())
    explanation = explain_recommendation(index, "U1", "A3")

    assert explanation.user_id == "U1"
    assert explanation.recommended_item == "A3"
    assert explanation.explanation_text
    assert 0.0 <= explanation.support_confidence <= 1.0
    assert explanation.explanation_type in {
        "history_and_overlap",
        "profile_support",
        "cold_start_popularity",
    }
    assert isinstance(explanation.supporting_history, list)
    assert isinstance(explanation.supporting_similar_items, list)


def test_cold_start_style_explanation_is_supported():
    index = ItemSimilarityIndex(_sample_rows())
    explanation = explain_recommendation(index, "UNKNOWN_USER", "A3")

    assert explanation.explanation_type == "cold_start_popularity"
    assert explanation.support_confidence >= 0.0
