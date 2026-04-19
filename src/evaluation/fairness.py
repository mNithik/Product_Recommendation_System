"""User-oriented fairness-style summaries (activity groups), inspired by Li et al.

This is a lightweight *audit* of disparity across users binned by training-set
activity (number of train ratings), not a full re-ranking fairness intervention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_ndcg_by_train_activity(
    per_user_rows: list[dict],
    n_buckets: int = 4,
) -> pd.DataFrame:
    """
    Bucket users by ``n_train`` quantiles and report mean ranking metrics per bucket.

    Args:
        per_user_rows: dicts with keys ``n_train``, ``precision``, ``recall``, ``ndcg``.
        n_buckets: number of quantile buckets (default 4 ≈ quartiles).

    Returns:
        DataFrame with one row per bucket.
    """
    if not per_user_rows:
        return pd.DataFrame()

    df = pd.DataFrame(per_user_rows)
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
        )
        .reset_index()
    )

    if len(out) >= 2 and (out["mean_ndcg"] > 0).any():
        hi = float(out["mean_ndcg"].max())
        lo = float(out["mean_ndcg"][out["mean_ndcg"] > 0].min() or out["mean_ndcg"].min())
        ratio = hi / lo if lo > 0 else float("nan")
        out.attrs["ndcg_max_min_ratio"] = ratio
    else:
        out.attrs["ndcg_max_min_ratio"] = float("nan")

    return out


def disparity_max_min_ratio(summary: pd.DataFrame, column: str) -> float:
    """
    max(mean bucket metric) / min(mean bucket metric) for positive values only.

    Used for NDCG, Precision, Recall disparity across activity buckets.
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
    summary = summarize_ndcg_by_train_activity(per_user_rows, n_buckets=n_buckets)
    return disparity_max_min_ratio(summary, "mean_ndcg")


def disparity_ratios_by_metric(
    per_user_rows: list[dict], n_buckets: int = 4
) -> dict[str, float]:
    """Disparity ratios for mean NDCG, Precision, and Recall across activity buckets."""
    summary = summarize_ndcg_by_train_activity(per_user_rows, n_buckets=n_buckets)
    return {
        "ndcg": disparity_max_min_ratio(summary, "mean_ndcg"),
        "precision": disparity_max_min_ratio(summary, "mean_precision"),
        "recall": disparity_max_min_ratio(summary, "mean_recall"),
    }
