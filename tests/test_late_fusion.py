"""Late fusion: CF pool order + text similarity."""

from src.evaluation.late_fusion import cf_position_scores, rank_items_late_fusion


def test_cf_position_scores_extremes():
    pool = ["a", "b", "c"]
    s = cf_position_scores(pool)
    assert s["a"] == 1.0
    assert s["c"] == 0.0
    assert abs(s["b"] - 0.5) < 1e-9


def test_rank_items_late_fusion_alpha_zero_sorts_by_text():
    train: list = []
    pool = ["x", "y", "z"]

    def cos(_tr, _u, item):
        return {"x": 0.1, "y": 0.9, "z": 0.5}[item]

    rows = rank_items_late_fusion(pool, train, "U", cos, alpha=0.0)
    assert [r[0] for r in rows] == ["y", "z", "x"]


def test_rank_items_late_fusion_alpha_one_keeps_cf_order():
    train: list = []
    pool = ["a", "b", "c"]

    def cos(_tr, _u, item):
        return 0.99 if item == "c" else 0.1

    rows = rank_items_late_fusion(pool, train, "U", cos, alpha=1.0)
    assert [r[0] for r in rows] == ["a", "b", "c"]
