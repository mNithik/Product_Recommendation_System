"""Unit tests for data preprocessing."""

import json
import os
import tempfile

import pytest

from src.preprocessing.preprocess import load_reviews, split_per_user


def _make_jsonl(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def sample_reviews():
    return [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0},
        {"reviewerID": "U1", "asin": "A2", "overall": 3.0},
        {"reviewerID": "U1", "asin": "A3", "overall": 4.0},
        {"reviewerID": "U1", "asin": "A4", "overall": 2.0},
        {"reviewerID": "U1", "asin": "A5", "overall": 1.0},
        {"reviewerID": "U2", "asin": "A1", "overall": 4.0},
        {"reviewerID": "U2", "asin": "A2", "overall": 5.0},
        {"reviewerID": "U2", "asin": "A6", "overall": 3.0},
        {"reviewerID": "U2", "asin": "A7", "overall": 2.0},
        {"reviewerID": "U2", "asin": "A8", "overall": 4.0},
    ]


class TestLoadReviews:
    def test_loads_valid_records(self, sample_reviews):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for r in sample_reviews:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            loaded = list(load_reviews(path))
            assert len(loaded) == 10
            assert all(1 <= r[2] <= 5 for r in loaded)
        finally:
            os.unlink(path)

    def test_skips_invalid_rating(self):
        records = [
            {"reviewerID": "U1", "asin": "A1", "overall": 6.0},
            {"reviewerID": "U1", "asin": "A2", "overall": 0.0},
            {"reviewerID": "U1", "asin": "A3", "overall": 3.0},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            loaded = list(load_reviews(path))
            assert len(loaded) == 1
            assert loaded[0][2] == 3.0
        finally:
            os.unlink(path)

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"reviewerID": "U1", "asin": "A1", "overall": 4.0}) + "\n")
            f.write("\n")
            f.write(json.dumps({"reviewerID": "U1", "asin": "A2", "overall": 3.0}) + "\n")
            path = f.name
        try:
            loaded = list(load_reviews(path))
            assert len(loaded) == 2
        finally:
            os.unlink(path)


class TestSplitPerUser:
    def test_split_ratio(self, sample_reviews):
        by_user = {}
        for r in sample_reviews:
            by_user.setdefault(r["reviewerID"], []).append(r)

        train, test = split_per_user(by_user, train_ratio=0.8, random_state=42)
        assert len(train) + len(test) == len(sample_reviews)

    def test_all_users_represented(self, sample_reviews):
        by_user = {}
        for r in sample_reviews:
            by_user.setdefault(r["reviewerID"], []).append(r)

        train, test = split_per_user(by_user, train_ratio=0.8, random_state=42)
        train_users = {r["reviewerID"] for r in train}
        test_users = {r["reviewerID"] for r in test}
        assert train_users | test_users == {"U1", "U2"}

    def test_deterministic(self, sample_reviews):
        by_user = {}
        for r in sample_reviews:
            by_user.setdefault(r["reviewerID"], []).append(r)

        train1, test1 = split_per_user(by_user, random_state=42)
        train2, test2 = split_per_user(by_user, random_state=42)
        assert [r["asin"] for r in train1] == [r["asin"] for r in train2]
