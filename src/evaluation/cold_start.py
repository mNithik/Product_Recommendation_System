"""Cold-start and sparse-user benchmarking helpers."""

from __future__ import annotations

import logging

import pandas as pd

from .metrics import evaluate_recommendations_per_user

logger = logging.getLogger(__name__)


def summarize_user_regimes(
    per_user_rows: list[dict],
    *,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
    use_gpu: bool = False,
    debug: bool = False,
) -> dict[str, float]:
    """
    Summarize recommendation quality for sparse and warm user regimes.

    Returns headline metrics that are easy to compare across models.
    """
    if not per_user_rows:
        return {
            "n_cold_users": 0.0,
            "n_warm_users": 0.0,
            "cold_mean_precision": 0.0,
            "warm_mean_precision": 0.0,
            "cold_mean_recall": 0.0,
            "warm_mean_recall": 0.0,
            "cold_mean_ndcg": 0.0,
            "warm_mean_ndcg": 0.0,
            "cold_warm_precision_gap": 0.0,
            "cold_warm_recall_gap": 0.0,
            "cold_warm_ndcg_gap": 0.0,
        }

    if use_gpu:
        if debug:
            logger.info(
                "[cold-start] trying GPU aggregation (rows=%d, cold<=%d, warm>=%d)",
                len(per_user_rows),
                cold_max_train,
                warm_min_train,
            )
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if debug:
                logger.info("[cold-start] torch device resolved to %s", device)
            n_train = torch.tensor([float(row.get("n_train", 0.0)) for row in per_user_rows], device=device)
            precision = torch.tensor([float(row.get("precision", 0.0)) for row in per_user_rows], device=device)
            recall = torch.tensor([float(row.get("recall", 0.0)) for row in per_user_rows], device=device)
            ndcg = torch.tensor([float(row.get("ndcg", 0.0)) for row in per_user_rows], device=device)

            cold_mask = n_train <= float(cold_max_train)
            warm_mask = n_train >= float(warm_min_train)

            def _masked_mean(values, mask) -> float:
                count = int(mask.sum().item())
                if count == 0:
                    return 0.0
                return float(values[mask].mean().item())

            cold_precision = _masked_mean(precision, cold_mask)
            warm_precision = _masked_mean(precision, warm_mask)
            cold_recall = _masked_mean(recall, cold_mask)
            warm_recall = _masked_mean(recall, warm_mask)
            cold_ndcg = _masked_mean(ndcg, cold_mask)
            warm_ndcg = _masked_mean(ndcg, warm_mask)

            return {
                "n_cold_users": float(cold_mask.sum().item()),
                "n_warm_users": float(warm_mask.sum().item()),
                "cold_mean_precision": cold_precision,
                "warm_mean_precision": warm_precision,
                "cold_mean_recall": cold_recall,
                "warm_mean_recall": warm_recall,
                "cold_mean_ndcg": cold_ndcg,
                "warm_mean_ndcg": warm_ndcg,
                "cold_warm_precision_gap": warm_precision - cold_precision,
                "cold_warm_recall_gap": warm_recall - cold_recall,
                "cold_warm_ndcg_gap": warm_ndcg - cold_ndcg,
            }
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if debug:
                logger.warning("[cold-start] GPU aggregation failed, falling back to CPU: %s", exc)
            pass

    if debug:
        logger.info("[cold-start] using CPU pandas aggregation")
    df = pd.DataFrame(per_user_rows)
    cold = df[df["n_train"] <= cold_max_train]
    warm = df[df["n_train"] >= warm_min_train]

    def _mean(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(frame[column].mean())

    cold_precision = _mean(cold, "precision")
    warm_precision = _mean(warm, "precision")
    cold_recall = _mean(cold, "recall")
    warm_recall = _mean(warm, "recall")
    cold_ndcg = _mean(cold, "ndcg")
    warm_ndcg = _mean(warm, "ndcg")

    return {
        "n_cold_users": float(len(cold)),
        "n_warm_users": float(len(warm)),
        "cold_mean_precision": cold_precision,
        "warm_mean_precision": warm_precision,
        "cold_mean_recall": cold_recall,
        "warm_mean_recall": warm_recall,
        "cold_mean_ndcg": cold_ndcg,
        "warm_mean_ndcg": warm_ndcg,
        "cold_warm_precision_gap": warm_precision - cold_precision,
        "cold_warm_recall_gap": warm_recall - cold_recall,
        "cold_warm_ndcg_gap": warm_ndcg - cold_ndcg,
    }


def evaluate_cold_start_benchmark(
    model,
    train_data: list[dict],
    test_data: list[dict],
    *,
    top_n: int = 10,
    batch_size: int = 512,
    min_train_ratings: int = 1,
    max_candidates: int = 10000,
    relevance_threshold=None,
    min_item_ratings: int = 0,
    max_users: int | None = None,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
    use_gpu: bool = False,
    debug: bool = False,
) -> dict[str, float]:
    """Run the standard recommendation protocol and summarize sparse-user performance."""
    if debug:
        logger.info(
            "[cold-start] evaluation started (top_n=%d, max_users=%s, max_candidates=%d, use_gpu=%s)",
            top_n,
            str(max_users),
            max_candidates,
            str(use_gpu),
        )
    per_user_rows = evaluate_recommendations_per_user(
        model,
        train_data,
        test_data,
        top_n=top_n,
        batch_size=batch_size,
        min_train_ratings=min_train_ratings,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
        min_item_ratings=min_item_ratings,
        max_users=max_users,
    )
    return summarize_user_regimes(
        per_user_rows,
        cold_max_train=cold_max_train,
        warm_min_train=warm_min_train,
        use_gpu=use_gpu,
        debug=debug,
    )


def compare_cold_start_benchmarks(
    baseline_metrics: dict[str, float],
    comparison_metrics: dict[str, float],
    *,
    prefix: str = "comparison",
) -> dict[str, float]:
    """
    Compare two cold-start benchmark summaries.

    Useful for reporting whether a hybrid branch helps sparse users relative to a
    baseline ranker.
    """
    return {
        f"{prefix}_cold_precision_delta": float(
            comparison_metrics.get("cold_mean_precision", 0.0) - baseline_metrics.get("cold_mean_precision", 0.0)
        ),
        f"{prefix}_cold_recall_delta": float(
            comparison_metrics.get("cold_mean_recall", 0.0) - baseline_metrics.get("cold_mean_recall", 0.0)
        ),
        f"{prefix}_cold_ndcg_delta": float(
            comparison_metrics.get("cold_mean_ndcg", 0.0) - baseline_metrics.get("cold_mean_ndcg", 0.0)
        ),
        f"{prefix}_warm_ndcg_delta": float(
            comparison_metrics.get("warm_mean_ndcg", 0.0) - baseline_metrics.get("warm_mean_ndcg", 0.0)
        ),
    }
