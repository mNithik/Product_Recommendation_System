"""Create a smaller train/test dataset by filtering to a user subset."""

import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


def _filter_jsonl_by_users(in_path: str, out_path: str, keep_users: set) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("reviewerID") in keep_users:
                fout.write(line)
                n += 1
    return n


def make_small_dataset(train_in: str, test_in: str, train_out: str, test_out: str,
                       max_users: int = 5000, min_train_ratings: int = 5,
                       min_test_ratings: int = 1) -> dict:
    """Select active users and write filtered JSONL splits."""
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    with open(train_in, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                train_counts[r["reviewerID"]] += 1

    with open(test_in, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                test_counts[r["reviewerID"]] += 1

    eligible = [
        u for u, c in train_counts.items()
        if c >= min_train_ratings and test_counts.get(u, 0) >= min_test_ratings
    ]
    eligible.sort()
    keep_users = set(eligible[:max_users])

    n_train = _filter_jsonl_by_users(train_in, train_out, keep_users)
    n_test = _filter_jsonl_by_users(test_in, test_out, keep_users)
    result = {"users": len(keep_users), "train_rows": n_train, "test_rows": n_test}
    logger.info("Small dataset: %s", result)
    return result
