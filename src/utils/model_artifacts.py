"""Helpers for saving and loading trained model artifacts."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def artifact_dir(output_dir: str, experiment_name: str) -> Path:
    """Return the model-artifact directory for an experiment."""
    return Path(output_dir) / experiment_name / "models"


def artifact_path(output_dir: str, experiment_name: str, model_name: str) -> Path:
    """Return the checkpoint path for one model."""
    return artifact_dir(output_dir, experiment_name) / f"{_safe_name(model_name)}.pkl"


def save_model_artifact(model, *, output_dir: str, experiment_name: str, model_name: str) -> str:
    """Serialize a fitted model to disk with pickle."""
    path = artifact_path(output_dir, experiment_name, model_name)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model artifact for %s to %s", model_name, path)
    return str(path)


def load_model_artifact(*, output_dir: str, experiment_name: str, model_name: str):
    """Load a fitted model artifact from disk."""
    path = artifact_path(output_dir, experiment_name, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found for {model_name}: {path}")
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    logger.info("Loaded model artifact for %s from %s", model_name, path)
    return model
