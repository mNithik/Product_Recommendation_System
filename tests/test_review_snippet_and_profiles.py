"""Review snippet helper and TF-IDF profile index (metadata + summary + reviewText)."""

from src.trustworthiness.text_profiles import ReviewTextProfileIndex, review_snippet_from_record


def test_review_snippet_combines_summary_and_body():
    r = {"summary": "Great yarn", "reviewText": "Soft and durable for knitting."}
    s = review_snippet_from_record(r)
    assert "Great yarn" in s
    assert "Soft and durable" in s


def test_review_snippet_summary_only():
    r = {"summary": "A" * 50, "reviewText": ""}
    s = review_snippet_from_record(r)
    assert "A" in s


def test_review_snippet_empty_when_missing():
    assert review_snippet_from_record({}) == ""
    assert review_snippet_from_record({"summary": "", "reviewText": ""}) == ""


def test_review_snippet_fuses_overall_verified_style():
    r = {
        "overall": 5,
        "verified": True,
        "style": {"Format:": " Paperback", "Color:": " Blue"},
        "summary": "Great book",
        "reviewText": "Useful reference.",
    }
    s = review_snippet_from_record(r)
    assert "5 star rating" in s
    assert "verified purchase" in s
    assert "Format Paperback" in s
    assert "Color Blue" in s
    assert "Great book" in s
    assert "Useful reference" in s


def test_review_snippet_overall_only_when_no_text():
    s = review_snippet_from_record({"overall": 4, "verified": False})
    assert "4 star rating" in s
    assert "unverified review" in s


def test_tfidf_index_accepts_summary_without_body():
    rows = []
    for i in range(10):
        rows.append(
            {
                "reviewerID": "U1",
                "asin": f"B{i:05d}",
                "overall": 5,
                "summary": f"quality craft item number {i} for hobbies and sewing projects",
                "reviewText": "",
            }
        )
    idx = ReviewTextProfileIndex(rows, min_df=1, max_features=500)
    assert len(idx._item_ids) == 10
    sim = idx.cosine_user_item(rows, "U1", "B00000")
    assert 0.0 <= sim <= 1.0
