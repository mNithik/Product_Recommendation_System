"""YAML configuration loader with defaults and CLI override support."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "data": {
        "raw_path": "Arts_Crafts_and_Sewing_5.json/Arts_Crafts_and_Sewing_5.json",
        "train_path": "data/train.json",
        "test_path": "data/test.json",
        "small_mode": False,
        "small_train_path": "data_small/train.json",
        "small_test_path": "data_small/test.json",
        "train_ratio": 0.8,
        "random_state": 42,
    },
    "model": {
        "type": "bpr",
        "n_factors": 64,
        "reg": 0.0001,
        "n_epochs": 50,
        "lr": 0.001,
        "use_gpu": True,
    },
    "evaluation": {
        "top_n": 10,
        "max_candidates": 100000,
        "min_train_ratings": 5,
        "relevance_threshold": 4.0,
        "min_item_ratings": 0,
    },
    "experiment": {
        "name": "default",
        "output_dir": "experiments",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Convert nested dict to nested SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(config_path: str | None = None) -> SimpleNamespace:
    """Load config from YAML, merge with defaults, return as namespace."""
    cfg = DEFAULTS.copy()

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
        logger.info("Loaded config from %s", config_path)
    else:
        logger.info("Using default configuration")

    ns = _dict_to_namespace(cfg)

    if ns.data.small_mode:
        ns.data.train_path = ns.data.small_train_path
        ns.data.test_path = ns.data.small_test_path

    return ns


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Product Recommendation System")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (overrides config)",
    )
    return parser.parse_args()
