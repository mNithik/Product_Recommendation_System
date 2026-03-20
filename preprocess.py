"""
Data preprocessing for Amazon Toys and Games 5-core dataset.
Creates 80/20 train-test split per user as specified in the project.
"""

import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def load_reviews(filepath):
    """Load reviews from JSONL file. Yields (user_id, item_id, rating) tuples."""
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
                    # Normalize rating to 1-5 integer if needed
                    rating = float(rating)
                    if 1 <= rating <= 5:
                        yield (user_id, item_id, rating, record)
            except json.JSONDecodeError:
                continue


def split_per_user(reviews_by_user, train_ratio=0.8, random_state=42):
    """Split each user's reviews: train_ratio for train, rest for test."""
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


def run_preprocessing(raw_path, train_path, test_path, train_ratio=0.8, random_state=42):
    """Load raw data, split per user, and save train/test sets."""
    print("Loading reviews...")
    reviews_by_user = defaultdict(list)

    for user_id, item_id, rating, record in tqdm(load_reviews(raw_path), desc="Loading"):
        reviews_by_user[user_id].append({
            "reviewerID": user_id,
            "asin": item_id,
            "overall": rating,
            "reviewText": record.get("reviewText", ""),
            "summary": record.get("summary", ""),
        })

    print(f"Loaded {sum(len(v) for v in reviews_by_user.values())} reviews from {len(reviews_by_user)} users")

    print("Splitting 80/20 per user...")
    train, test = split_per_user(reviews_by_user, train_ratio, random_state)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for item in tqdm(train, desc="Writing train"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for item in tqdm(test, desc="Writing test"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} reviews | Test: {len(test)} reviews")
    return train, test


if __name__ == "__main__":
    from config import RAW_DATA_PATH, TRAIN_PATH, TEST_PATH, TRAIN_RATIO, RANDOM_STATE

    run_preprocessing(
        RAW_DATA_PATH,
        TRAIN_PATH,
        TEST_PATH,
        train_ratio=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )
