"""Data preprocessing for Amazon Arts, Crafts & Sewing 5-core dataset."""

import json
import logging
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_reviews(filepath: str):
    """Yield (user_id, item_id, rating, full_record) from JSONL."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                user_id = record.get("reviewerID")
                item_id = record.get("asin")
                rating = record.get("overall")
                if user_id and item_id and rating is not None:
                    rating = float(rating)
                    if 1 <= rating <= 5:
                        yield (user_id, item_id, rating, record)
            except json.JSONDecodeError:
                continue


def split_per_user(reviews_by_user: dict, train_ratio: float = 0.8,
                   random_state: int = 42):
    """Per-user random split: train_ratio for train, rest for test."""
    rng = np.random.default_rng(random_state)
    train, test = [], []
    for user_id, reviews in reviews_by_user.items():
        reviews = list(reviews)
        n = len(reviews)
        indices = rng.permutation(n)
        split_idx = int(n * train_ratio)
        for i in indices[:split_idx]:
            train.append(reviews[i])
        for i in indices[split_idx:]:
            test.append(reviews[i])
    return train, test


def run_preprocessing(raw_path: str, train_path: str, test_path: str,
                      train_ratio: float = 0.8, random_state: int = 42):
    """Full pipeline: load raw data -> per-user split -> write JSONL."""
    logger.info("Loading reviews from %s...", raw_path)
    reviews_by_user = defaultdict(list)

    for user_id, item_id, rating, record in tqdm(load_reviews(raw_path), desc="Loading"):
        row = {
            "reviewerID": user_id,
            "asin": item_id,
            "overall": rating,
            "reviewText": record.get("reviewText", ""),
            "summary": record.get("summary", ""),
        }
        if "verified" in record and record["verified"] is not None:
            row["verified"] = record["verified"]
        if record.get("style"):
            row["style"] = record["style"]
        reviews_by_user[user_id].append(row)

    total = sum(len(v) for v in reviews_by_user.values())
    logger.info("Loaded %d reviews from %d users", total, len(reviews_by_user))

    logger.info("Splitting %.0f/%.0f per user...", train_ratio * 100, (1 - train_ratio) * 100)
    train, test = split_per_user(reviews_by_user, train_ratio, random_state)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for item in tqdm(train, desc="Writing train"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for item in tqdm(test, desc="Writing test"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Train: %d reviews | Test: %d reviews", len(train), len(test))
    return train, test
