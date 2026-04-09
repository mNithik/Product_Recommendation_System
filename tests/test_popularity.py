"""Unit tests for the Popularity baseline model."""

import pytest

from src.models.popularity import PopularityBaseline


@pytest.fixture
def train_data():
    return [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 3.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A3", "overall": 5.0},
        {"reviewerID": "U3", "asin": "A1", "overall": 3.0},
        {"reviewerID": "U3", "asin": "A2", "overall": 4.0},
        {"reviewerID": "U3", "asin": "A3", "overall": 2.0},
        {"reviewerID": "U3", "asin": "A4", "overall": 5.0},
    ]


@pytest.fixture
def fitted_model(train_data):
    model = PopularityBaseline()
    model.fit(train_data)
    return model


class TestPopularityFit:
    def test_global_mean(self, fitted_model):
        assert 1.0 <= fitted_model.global_mean <= 5.0

    def test_all_items_indexed(self, fitted_model):
        assert len(fitted_model.item_idx) == 4

    def test_all_users_indexed(self, fitted_model):
        assert len(fitted_model.user_idx) == 3


class TestPopularityPredict:
    def test_known_item(self, fitted_model):
        pred = fitted_model.predict("U1", "A1")
        assert 1.0 <= pred <= 5.0

    def test_unknown_item_returns_global_mean(self, fitted_model):
        pred = fitted_model.predict("U1", "UNKNOWN")
        assert pred == pytest.approx(fitted_model.global_mean)


class TestPopularityRecommend:
    def test_excludes_seen_items(self, fitted_model):
        recs = fitted_model.recommend_top_n("U1", n=10)
        assert "A1" not in recs
        assert "A2" not in recs

    def test_returns_correct_count(self, fitted_model):
        recs = fitted_model.recommend_top_n("U1", n=2)
        assert len(recs) == 2

    def test_most_popular_first(self, fitted_model):
        recs = fitted_model.recommend_top_n("U1", n=10)
        assert len(recs) > 0

    def test_unknown_user_returns_popular(self, fitted_model):
        recs = fitted_model.recommend_top_n("UNKNOWN_USER", n=3)
        assert len(recs) == 3

    def test_get_popular_items(self, fitted_model):
        popular = fitted_model.get_popular_items(n=2)
        assert len(popular) == 2
        assert popular[0]["num_ratings"] >= popular[1]["num_ratings"]
