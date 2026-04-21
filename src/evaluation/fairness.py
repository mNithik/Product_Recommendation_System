"""User-oriented fairness-style summaries for recommender evaluation.

This module keeps fairness as an audit layer rather than a fairness-aware
re-ranking intervention. The focus is on exposing disparities across user
activity levels and cold-start slices using the same recommendation protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_metrics_by_train_activity(
    per_user_rows: list[dict],
    n_buckets: int = 4,
) -> pd.DataFrame:
    """
    Bucket users by ``n_train`` quantiles and report mean ranking metrics per bucket.

    Args:
        per_user_rows: dicts with keys like ``n_train``, ``precision``, ``recall``, ``ndcg``.
        n_buckets: number of quantile buckets (default 4, roughly quartiles).

    Returns:
        DataFrame with one row per bucket.
    """
    if not per_user_rows:
        return pd.DataFrame()

    df = pd.DataFrame(per_user_rows)
    for optional_col in ("hit_rate", "avg_recommended_popularity", "catalog_coverage"):
        if optional_col not in df.columns:
            df[optional_col] = np.nan

    if df["n_train"].nunique() < 2:
        out = df.groupby("n_train", as_index=False).agg(
            n_users=("user", "count"),
            min_train=("n_train", "min"),
            max_train=("n_train", "max"),
            mean_ndcg=("ndcg", "mean"),
            std_ndcg=("ndcg", "std"),
            median_ndcg=("ndcg", "median"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            median_precision=("precision", "median"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            median_recall=("recall", "median"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_recommended_popularity=("avg_recommended_popularity", "mean"),
            mean_catalog_coverage=("catalog_coverage", "mean"),
        )
        out.rename(columns={"n_train": "bucket"}, inplace=True)
        out.attrs["ndcg_max_min_ratio"] = float("nan")
        return out

    try:
        df["bucket"] = pd.qcut(
            df["n_train"],
            q=min(n_buckets, df["n_train"].nunique()),
            duplicates="drop",
        )
    except ValueError:
        df["bucket"] = pd.cut(df["n_train"], bins=min(n_buckets, df["n_train"].nunique()))

    out = (
        df.groupby("bucket", observed=True)
        .agg(
            n_users=("user", "count"),
            min_train=("n_train", "min"),
            max_train=("n_train", "max"),
            mean_ndcg=("ndcg", "mean"),
            std_ndcg=("ndcg", "std"),
            median_ndcg=("ndcg", "median"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            median_precision=("precision", "median"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            median_recall=("recall", "median"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_recommended_popularity=("avg_recommended_popularity", "mean"),
            mean_catalog_coverage=("catalog_coverage", "mean"),
        )
        .reset_index()
    )

    if len(out) >= 2 and (out["mean_ndcg"] > 0).any():
        hi = float(out["mean_ndcg"].max())
        positive_vals = out["mean_ndcg"][out["mean_ndcg"] > 0]
        lo = float(positive_vals.min()) if len(positive_vals) > 0 else float(out["mean_ndcg"].min())
        out.attrs["ndcg_max_min_ratio"] = hi / lo if lo > 0 else float("nan")
    else:
        out.attrs["ndcg_max_min_ratio"] = float("nan")

    return out


def summarize_ndcg_by_train_activity(
    per_user_rows: list[dict],
    n_buckets: int = 4,
) -> pd.DataFrame:
    """Backward-compatible alias for the original fairness summary helper."""
    return summarize_metrics_by_train_activity(per_user_rows, n_buckets=n_buckets)


def disparity_max_min_ratio(summary: pd.DataFrame, column: str) -> float:
    """
    max(mean bucket metric) / min(mean bucket metric) for positive values only.

    Used for NDCG, Precision, Recall, and related fairness audit columns.
    """
    if summary.empty or column not in summary.columns:
        return float("nan")
    vals = summary[column].values.astype(float)
    pos = vals[vals > 0]
    if len(pos) < 2:
        return float("nan")
    return float(np.max(pos) / np.min(pos))


def disparity_ratio(per_user_rows: list[dict], n_buckets: int = 4) -> float:
    """max(mean NDCG bucket) / min(mean NDCG bucket), ignoring all-zero buckets."""
    summary = summarize_metrics_by_train_activity(per_user_rows, n_buckets=n_buckets)
    return disparity_max_min_ratio(summary, "mean_ndcg")


def disparity_ratios_by_metric(
    per_user_rows: list[dict], n_buckets: int = 4
) -> dict[str, float]:
    """Disparity ratios for mean NDCG, Precision, Recall, and related metrics."""
    summary = summarize_metrics_by_train_activity(per_user_rows, n_buckets=n_buckets)
    return {
        "ndcg": disparity_max_min_ratio(summary, "mean_ndcg"),
        "precision": disparity_max_min_ratio(summary, "mean_precision"),
        "recall": disparity_max_min_ratio(summary, "mean_recall"),
        "hit_rate": disparity_max_min_ratio(summary, "mean_hit_rate"),
        "recommended_popularity": disparity_max_min_ratio(summary, "mean_recommended_popularity"),
    }


def summarize_cold_start_gap(
    per_user_rows: list[dict],
    *,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
) -> dict[str, float]:
    """
    Compare low-history users against warmer users.

    This provides a simple report-friendly fairness slice for sparse-user
    behavior without introducing protected-group assumptions.
    """
    if not per_user_rows:
        return {
            "n_cold_users": 0.0,
            "n_warm_users": 0.0,
            "cold_mean_ndcg": 0.0,
            "warm_mean_ndcg": 0.0,
            "cold_warm_ndcg_gap": 0.0,
            "cold_mean_precision": 0.0,
            "warm_mean_precision": 0.0,
            "cold_warm_precision_gap": 0.0,
            "cold_mean_recall": 0.0,
            "warm_mean_recall": 0.0,
            "cold_warm_recall_gap": 0.0,
        }

    df = pd.DataFrame(per_user_rows)
    cold = df[df["n_train"] <= cold_max_train]
    warm = df[df["n_train"] >= warm_min_train]

    def _mean(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(frame[column].mean())

    cold_ndcg = _mean(cold, "ndcg")
    warm_ndcg = _mean(warm, "ndcg")
    cold_precision = _mean(cold, "precision")
    warm_precision = _mean(warm, "precision")
    cold_recall = _mean(cold, "recall")
    warm_recall = _mean(warm, "recall")

    return {
        "n_cold_users": float(len(cold)),
        "n_warm_users": float(len(warm)),
        "cold_mean_ndcg": cold_ndcg,
        "warm_mean_ndcg": warm_ndcg,
        "cold_warm_ndcg_gap": warm_ndcg - cold_ndcg,
        "cold_mean_precision": cold_precision,
        "warm_mean_precision": warm_precision,
        "cold_warm_precision_gap": warm_precision - cold_precision,
        "cold_mean_recall": cold_recall,
        "warm_mean_recall": warm_recall,
        "cold_warm_recall_gap": warm_recall - cold_recall,
    }


def run_activity_fairness_audit(
    per_user_rows: list[dict],
    *,
    n_buckets: int = 4,
    cold_max_train: int = 5,
    warm_min_train: int = 20,
) -> dict[str, object]:
    """Return summary tables plus headline fairness metrics."""
    summary = summarize_metrics_by_train_activity(per_user_rows, n_buckets=n_buckets)
    disparity = disparity_ratios_by_metric(per_user_rows, n_buckets=n_buckets)
    cold_start = summarize_cold_start_gap(
        per_user_rows,
        cold_max_train=cold_max_train,
        warm_min_train=warm_min_train,
    )
    return {
        "summary": summary,
        "disparity": disparity,
        "cold_start": cold_start,
    }
