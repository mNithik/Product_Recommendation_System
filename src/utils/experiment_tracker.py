"""Lightweight experiment tracker — saves metrics and config to JSON."""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks experiment metadata, config, and metrics; persists to JSON."""

    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        self.record = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "config": {},
            "metrics": {},
            "steps": [],
        }

    def log_config(self, config):
        """Log the full config namespace as a dict."""
        self.record["config"] = self._namespace_to_dict(config)

    def log_metric(self, name: str, value: float, step: str | None = None):
        """Log a single metric."""
        self.record["metrics"][name] = value
        if step:
            self.record["steps"].append({"step": step, "metric": name, "value": value})
        logger.info("  %s = %.4f", name, value)

    def log_metrics(self, metrics: dict, step: str | None = None):
        """Log a dict of metrics."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(name, value, step)

    def save(self):
        """Persist the experiment record to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.record, f, indent=2, default=str)
        logger.info("Experiment saved to %s", path)
        return path

    @staticmethod
    def _namespace_to_dict(obj):
        """Convert SimpleNamespace tree to plain dict."""
        if hasattr(obj, "__dict__"):
            return {k: ExperimentTracker._namespace_to_dict(v) for k, v in vars(obj).items()}
        return obj
