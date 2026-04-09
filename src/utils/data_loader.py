"""Data loading and index-building utilities."""

import json
import logging
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> list[dict]:
    """Load train or test data from JSONL."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  Loading {os.path.basename(filepath)}", unit=" lines"):
            if line.strip():
                data.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(data), filepath)
    return data


def build_index(data: list[dict]):
    """Build user/item ID-to-index mappings from interaction records."""
    users = {}
    items = {}
    for r in data:
        u, i = r["reviewerID"], r["asin"]
        if u not in users:
            users[u] = len(users)
        if i not in items:
            items[i] = len(items)

    rev_users = {v: k for k, v in users.items()}
    rev_items = {v: k for k, v in items.items()}
    return users, items, rev_users, rev_items
