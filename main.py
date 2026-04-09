"""
Main pipeline for the Product Recommendation System.

Runs ALL models head-to-head:
  Rating prediction:  Popularity, MF (ALS)
  Top-N ranking:      Popularity, BPR (custom), BPR (implicit), Implicit ALS, WARP (custom)

Usage:
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --experiment my_run
"""

import logging
import os
import sys
import time

from src.utils.config_loader import load_config, parse_args
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.data_loader import load_data
from src.preprocessing import run_preprocessing, make_small_dataset
from src.evaluation import evaluate_rating_prediction, evaluate_recommendations
from src.models import PopularityBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_results_table(all_results: list[dict]):
    header = f"{'Model':<35} {'MAE':>8} {'RMSE':>8} {'P@10':>8} {'R@10':>8} {'F1':>8} {'NDCG@10':>8} {'Time':>8}"
    sep = "=" * len(header)
    logger.info("\n" + sep)
    logger.info("RESULTS COMPARISON")
    logger.info(sep)
    logger.info(header)
    logger.info("-" * len(header))
    for r in all_results:
        logger.info(
            f"{r['model']:<35} "
            f"{r.get('MAE', '—'):>8} "
            f"{r.get('RMSE', '—'):>8} "
            f"{r.get('Precision', '—'):>8} "
            f"{r.get('Recall', '—'):>8} "
            f"{r.get('F-measure', '—'):>8} "
            f"{r.get('NDCG', '—'):>8} "
            f"{r.get('time', '—'):>8}"
        )
    logger.info(sep)


def _run_rating_eval(model, name, test_data, tracker, all_results):
    """Evaluate a model on rating prediction and append to results."""
    t0 = time.time()
    metrics = evaluate_rating_prediction(model, test_data)
    elapsed = time.time() - t0
    logger.info("  %s — MAE: %.4f  RMSE: %.4f  (%.1fs)", name, metrics["MAE"], metrics["RMSE"], elapsed)
    tracker.log_metrics({f"{name}_MAE": metrics["MAE"], f"{name}_RMSE": metrics["RMSE"]}, step="rating")

    row = next((r for r in all_results if r["model"] == name), None)
    if row:
        row["MAE"] = _fmt(metrics["MAE"])
        row["RMSE"] = _fmt(metrics["RMSE"])
    else:
        all_results.append({"model": name, "MAE": _fmt(metrics["MAE"]), "RMSE": _fmt(metrics["RMSE"])})
    return metrics


def _run_ranking_eval(model, name, train_data, test_data, ev, tracker, all_results, fit_time):
    """Evaluate a model on Top-N ranking and append to results."""
    t0 = time.time()
    metrics = evaluate_recommendations(
        model, train_data, test_data,
        top_n=ev.top_n,
        min_train_ratings=ev.min_train_ratings,
        max_candidates=getattr(ev, "max_candidates", 100000),
        relevance_threshold=ev.relevance_threshold,
        min_item_ratings=getattr(ev, "min_item_ratings", 0),
    )
    eval_time = time.time() - t0
    total_time = fit_time + eval_time

    logger.info("  %s — P@%d: %.4f  R@%d: %.4f  F1: %.4f  NDCG: %.4f  (%d users, %.0fs)",
                name, ev.top_n, metrics["Precision"], ev.top_n, metrics["Recall"],
                metrics["F-measure"], metrics["NDCG"], metrics.get("n_users_eval", 0), total_time)
    tracker.log_metrics({f"{name}_{k}": v for k, v in metrics.items()}, step="ranking")

    row = next((r for r in all_results if r["model"] == name), None)
    if row:
        row.update({
            "Precision": _fmt(metrics["Precision"]),
            "Recall": _fmt(metrics["Recall"]),
            "F-measure": _fmt(metrics["F-measure"]),
            "NDCG": _fmt(metrics["NDCG"]),
            "time": f"{total_time:.0f}s",
        })
    else:
        all_results.append({
            "model": name,
            "Precision": _fmt(metrics["Precision"]),
            "Recall": _fmt(metrics["Recall"]),
            "F-measure": _fmt(metrics["F-measure"]),
            "NDCG": _fmt(metrics["NDCG"]),
            "time": f"{total_time:.0f}s",
        })
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.experiment:
        cfg.experiment.name = args.experiment

    tracker = ExperimentTracker(cfg.experiment.name, cfg.experiment.output_dir)
    tracker.log_config(cfg)
    all_results = []
    ev = cfg.evaluation

    # =========================================================================
    # Step 1: Preprocessing
    # =========================================================================
    train_path = cfg.data.train_path
    test_path = cfg.data.test_path

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.info("=== Step 1: Data Preprocessing ===")
        if not os.path.exists(cfg.data.raw_path):
            logger.error("Raw data not found at %s", cfg.data.raw_path)
            return
        if cfg.data.small_mode:
            full_train, full_test = "data/train.json", "data/test.json"
            if not os.path.exists(full_train) or not os.path.exists(full_test):
                run_preprocessing(cfg.data.raw_path, full_train, full_test,
                                  cfg.data.train_ratio, cfg.data.random_state)
            make_small_dataset(full_train, full_test, train_path, test_path,
                               max_users=5000,
                               min_train_ratings=ev.min_train_ratings)
        else:
            run_preprocessing(cfg.data.raw_path, train_path, test_path,
                              cfg.data.train_ratio, cfg.data.random_state)
    else:
        logger.info("Train/test data found — skipping preprocessing.")

    logger.info("=== Loading Data ===")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    logger.info("Train: %d | Test: %d", len(train_data), len(test_data))
    tracker.log_metric("train_size", len(train_data), step="data")
    tracker.log_metric("test_size", len(test_data), step="data")

    # =========================================================================
    # Model 1: Popularity Baseline (rating + ranking)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 1: Popularity Baseline")
    logger.info("=" * 60)
    pop = PopularityBaseline()
    t0 = time.time()
    pop.fit(train_data)
    pop_fit = time.time() - t0
    _run_rating_eval(pop, "Popularity Baseline", test_data, tracker, all_results)
    _run_ranking_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker, all_results, pop_fit)

    # =========================================================================
    # Model 2: Matrix Factorization — ALS (rating prediction)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 2: Matrix Factorization (ALS, GPU)")
    logger.info("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            from src.models import MatrixFactorizationGPU
            mf_epochs = getattr(cfg.model, "mf_epochs", 5)
            mf = MatrixFactorizationGPU(
                n_factors=cfg.model.n_factors,
                reg=0.1,
                n_epochs=mf_epochs,
            )
            t0 = time.time()
            mf.fit(train_data)
            mf_fit = time.time() - t0
            _run_rating_eval(mf, "Matrix Factorization (ALS)", test_data, tracker, all_results)
            all_results[-1 if all_results[-1]["model"] == "Matrix Factorization (ALS)" else 0]["time"] = f"{mf_fit:.0f}s"
        else:
            logger.warning("CUDA not available, skipping GPU MF")
    except ImportError:
        logger.warning("PyTorch not available, skipping MF")

    # =========================================================================
    # Model 3: BPR — custom PyTorch (ranking)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 3: BPR (custom PyTorch)")
    logger.info("=" * 60)
    from src.models import BPRMatrixFactorization
    bpr_custom = BPRMatrixFactorization(
        n_factors=cfg.model.n_factors,
        n_epochs=cfg.model.n_epochs,
        lr=cfg.model.lr,
        reg=cfg.model.reg,
        pos_threshold=ev.relevance_threshold,
    )
    t0 = time.time()
    bpr_custom.fit(train_data)
    bpr_fit = time.time() - t0
    _run_ranking_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker, all_results, bpr_fit)

    # =========================================================================
    # Model 4: BPR — implicit library (ranking)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 4: BPR (implicit library)")
    logger.info("=" * 60)
    from src.models import ImplicitBPRRanker
    bpr_impl = ImplicitBPRRanker(
        n_factors=cfg.model.n_factors,
        n_epochs=cfg.model.n_epochs,
        lr=cfg.model.lr,
        reg=cfg.model.reg,
        pos_threshold=ev.relevance_threshold,
    )
    t0 = time.time()
    bpr_impl.fit(train_data)
    bpr_impl_fit = time.time() - t0
    _run_ranking_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker, all_results, bpr_impl_fit)

    # =========================================================================
    # Model 5: Implicit ALS (ranking)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 5: Implicit ALS (Hu-Koren-Volinsky)")
    logger.info("=" * 60)
    from src.models import ImplicitALSRanker
    ials = ImplicitALSRanker(
        n_factors=cfg.model.n_factors,
        n_epochs=15,
        reg=cfg.model.reg,
        pos_threshold=ev.relevance_threshold,
    )
    t0 = time.time()
    ials.fit(train_data)
    ials_fit = time.time() - t0
    _run_ranking_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker, all_results, ials_fit)

    # =========================================================================
    # Model 6: WARP — custom PyTorch (ranking)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 6: WARP (custom PyTorch)")
    logger.info("=" * 60)
    from src.models import WARPModel
    warp = WARPModel(
        n_factors=cfg.model.n_factors,
        n_epochs=cfg.model.n_epochs,
        lr=cfg.model.lr,
        reg=cfg.model.reg,
        pos_threshold=ev.relevance_threshold,
    )
    t0 = time.time()
    warp.fit(train_data)
    warp_fit = time.time() - t0
    _run_ranking_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker, all_results, warp_fit)

    # =========================================================================
    # Final Results
    # =========================================================================
    _print_results_table(all_results)

    tracker.record["results_table"] = all_results
    tracker.save()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
