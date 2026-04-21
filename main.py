"""Main training and evaluation entry point."""

import logging
import os
import sys
import time

from src.utils.config_loader import load_config, parse_args
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.data_loader import load_data
from src.utils.model_artifacts import load_model_artifact, save_model_artifact
from src.preprocessing import run_preprocessing, make_small_dataset
from src.hybrid import ContentHybridConfig, ContentHybridRanker
from src.evaluation import (
    compare_cold_start_benchmarks,
    evaluate_cold_start_benchmark,
    evaluate_beyond_accuracy,
    evaluate_recommendations_per_user,
    evaluate_explainable_recommendations,
    evaluate_rating_prediction,
    evaluate_recommendations,
    run_activity_fairness_audit,
)
from src.postprocessing import CausalAdjustmentConfig
from src.models.popularity import PopularityBaseline
from src.trustworthiness import ReviewTextProfileIndex

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


def _resolve_torch_device(*, use_gpu: bool) -> str:
    """Resolve torch device for custom PyTorch models without changing baseline flow."""
    if not use_gpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, OSError):
        return "cpu"


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


def _run_explainability_eval(model, name, train_data, test_data, ev, tracker):
    """Evaluate post-hoc explanation and optional causal-adjusted ranking metrics."""
    t0 = time.time()
    metrics = evaluate_explainable_recommendations(
        model,
        train_data,
        test_data,
        top_n=ev.top_n,
        min_train_ratings=ev.min_train_ratings,
        max_candidates=getattr(ev, "max_candidates", 100000),
        relevance_threshold=ev.relevance_threshold,
        min_item_ratings=getattr(ev, "min_item_ratings", 0),
        explanation_sample_users=getattr(ev, "explanation_sample_users", 250),
        causal_config=CausalAdjustmentConfig(
            enabled=getattr(ev, "causal_adjustment_enabled", True),
            support_weight=float(getattr(ev, "causal_support_weight", 0.20)),
            popularity_penalty_weight=float(getattr(ev, "causal_popularity_penalty_weight", 0.10)),
        ),
    )
    elapsed = time.time() - t0

    logger.info(
        "  %s explainability - coverage: %.4f  cf coverage: %.4f  causal NDCG: %.4f  (%d users, %.1fs)",
        name,
        metrics["explanation_coverage"],
        metrics["counterfactual_coverage"],
        metrics.get("causal_NDCG", 0.0),
        int(metrics.get("n_explanation_users_eval", 0)),
        elapsed,
    )
    tracker.log_metrics({f"{name}_{k}": v for k, v in metrics.items()}, step="explainability")
    return metrics


def _run_fairness_eval(model, name, train_data, test_data, ev, tracker):
    """Evaluate activity-bucket fairness slices for ranking models."""
    t0 = time.time()
    per_user_rows = evaluate_recommendations_per_user(
        model,
        train_data,
        test_data,
        top_n=ev.top_n,
        batch_size=256,
        min_train_ratings=ev.min_train_ratings,
        max_candidates=getattr(ev, "max_candidates", 100000),
        relevance_threshold=ev.relevance_threshold,
        min_item_ratings=getattr(ev, "min_item_ratings", 0),
        max_users=getattr(ev, "fairness_max_users", 1500),
    )
    audit = run_activity_fairness_audit(
        per_user_rows,
        n_buckets=getattr(ev, "fairness_n_buckets", 4),
        cold_max_train=getattr(ev, "fairness_cold_max_train", ev.min_train_ratings),
        warm_min_train=getattr(ev, "fairness_warm_min_train", max(ev.min_train_ratings * 4, 20)),
    )
    elapsed = time.time() - t0

    disparity = audit["disparity"]
    cold_start = audit["cold_start"]
    metrics = {
        "fairness_n_users_eval": float(len(per_user_rows)),
        "fairness_ndcg_disparity": float(disparity.get("ndcg", float("nan"))),
        "fairness_precision_disparity": float(disparity.get("precision", float("nan"))),
        "fairness_recall_disparity": float(disparity.get("recall", float("nan"))),
        "fairness_hit_rate_disparity": float(disparity.get("hit_rate", float("nan"))),
        "fairness_recommended_popularity_disparity": float(
            disparity.get("recommended_popularity", float("nan"))
        ),
        "fairness_cold_mean_ndcg": float(cold_start.get("cold_mean_ndcg", 0.0)),
        "fairness_warm_mean_ndcg": float(cold_start.get("warm_mean_ndcg", 0.0)),
        "fairness_cold_warm_ndcg_gap": float(cold_start.get("cold_warm_ndcg_gap", 0.0)),
    }

    logger.info(
        "  %s fairness - NDCG disparity: %.4f  cold/warm NDCG gap: %.4f  (%d users, %.1fs)",
        name,
        metrics["fairness_ndcg_disparity"],
        metrics["fairness_cold_warm_ndcg_gap"],
        int(metrics["fairness_n_users_eval"]),
        elapsed,
    )
    tracker.log_metrics({f"{name}_{k}": v for k, v in metrics.items()}, step="fairness")
    return metrics


def _run_beyond_accuracy_eval(model, name, train_data, test_data, ev, tracker):
    """Evaluate coverage, diversity, novelty, and popularity concentration."""
    t0 = time.time()
    metrics = evaluate_beyond_accuracy(
        model,
        train_data,
        test_data,
        top_n=ev.top_n,
        batch_size=256,
        min_train_ratings=ev.min_train_ratings,
        max_candidates=getattr(ev, "max_candidates", 100000),
        relevance_threshold=ev.relevance_threshold,
        min_item_ratings=getattr(ev, "min_item_ratings", 0),
        max_users=getattr(ev, "beyond_accuracy_max_users", 1500),
    )
    elapsed = time.time() - t0
    logger.info(
        "  %s beyond-accuracy - coverage: %.4f  diversity: %.4f  novelty: %.4f  (%.1fs)",
        name,
        metrics["CatalogCoverage"],
        metrics["Diversity"],
        metrics["Novelty"],
        elapsed,
    )
    tracker.log_metrics({f"{name}_{k}": v for k, v in metrics.items()}, step="beyond_accuracy")
    return metrics


def _run_cold_start_eval(model, name, train_data, test_data, ev, tracker):
    """Evaluate sparse-user and warm-user recommendation quality explicitly."""
    t0 = time.time()
    metrics = evaluate_cold_start_benchmark(
        model,
        train_data,
        test_data,
        top_n=ev.top_n,
        batch_size=256,
        min_train_ratings=1,
        max_candidates=getattr(ev, "max_candidates", 100000),
        relevance_threshold=ev.relevance_threshold,
        min_item_ratings=getattr(ev, "min_item_ratings", 0),
        max_users=getattr(ev, "cold_start_max_users", 1500),
        cold_max_train=getattr(ev, "cold_start_threshold", 5),
        warm_min_train=getattr(ev, "warm_user_threshold", 20),
        use_gpu=bool(getattr(ev, "cold_start_use_gpu", False)),
        debug=bool(getattr(ev, "cold_start_debug", False)),
    )
    elapsed = time.time() - t0
    logger.info(
        "  %s cold-start - cold NDCG: %.4f  warm NDCG: %.4f  gap: %.4f  (%d cold users, %.1fs)",
        name,
        metrics["cold_mean_ndcg"],
        metrics["warm_mean_ndcg"],
        metrics["cold_warm_ndcg_gap"],
        int(metrics["n_cold_users"]),
        elapsed,
    )
    tracker.log_metrics({f"{name}_{k}": v for k, v in metrics.items()}, step="cold_start")
    return metrics


def _run_hybrid_eval(base_model, name, train_data, test_data, ev, tracker, all_results):
    """Evaluate a content-aware hybrid wrapper on top of a fitted ranking model."""
    t0 = time.time()
    hybrid_model = _build_content_hybrid_model(base_model, train_data, ev)
    wrap_time = time.time() - t0
    hybrid_name = f"{name} + Content Hybrid"

    logger.info("  %s - built TF-IDF hybrid wrapper (%.1fs)", hybrid_name, wrap_time)
    base_cold_start_metrics = _run_cold_start_eval(base_model, name, train_data, test_data, ev, tracker)
    _run_ranking_eval(hybrid_model, hybrid_name, train_data, test_data, ev, tracker, all_results, wrap_time)
    _run_explainability_eval(hybrid_model, hybrid_name, train_data, test_data, ev, tracker)
    _run_fairness_eval(hybrid_model, hybrid_name, train_data, test_data, ev, tracker)
    _run_beyond_accuracy_eval(hybrid_model, hybrid_name, train_data, test_data, ev, tracker)
    hybrid_cold_start_metrics = _run_cold_start_eval(hybrid_model, hybrid_name, train_data, test_data, ev, tracker)
    deltas = compare_cold_start_benchmarks(
        base_cold_start_metrics,
        hybrid_cold_start_metrics,
        prefix="hybrid_vs_base",
    )
    tracker.log_metrics({f"{hybrid_name}_{k}": v for k, v in deltas.items()}, step="cold_start")


def _fit_or_load_model(
    model,
    name: str,
    train_data: list[dict],
    *,
    output_dir: str,
    experiment_name: str,
    eval_only: bool,
    resume_artifacts: bool,
    save_models: bool,
):
    """Fit a model or load a saved artifact, returning (model, fit_time)."""
    if eval_only:
        loaded_model = load_model_artifact(
            output_dir=output_dir,
            experiment_name=experiment_name,
            model_name=name,
        )
        return loaded_model, 0.0

    if resume_artifacts:
        try:
            loaded_model = load_model_artifact(
                output_dir=output_dir,
                experiment_name=experiment_name,
                model_name=name,
            )
            logger.info("  %s — artifact found, skipping retraining.", name)
            return loaded_model, 0.0
        except FileNotFoundError:
            logger.info("  %s — no artifact found, training model.", name)

    t0 = time.time()
    model.fit(train_data)
    fit_time = time.time() - t0
    if save_models:
        save_model_artifact(
            model,
            output_dir=output_dir,
            experiment_name=experiment_name,
            model_name=name,
        )
    return model, fit_time


def _load_saved_model(name: str, *, output_dir: str, experiment_name: str):
    """Load a saved model artifact without constructing a fresh model first."""
    return load_model_artifact(
        output_dir=output_dir,
        experiment_name=experiment_name,
        model_name=name,
    )


def _build_content_hybrid_model(base_model, train_data, ev):
    """Wrap a fitted ranking model with optional TF-IDF content fusion."""
    text_index = ReviewTextProfileIndex(
        train_data,
        max_features=int(getattr(ev, "hybrid_max_features", 3500)),
        min_df=int(getattr(ev, "hybrid_min_df", 3)),
    )
    return ContentHybridRanker(
        base_model,
        train_data,
        text_index,
        config=ContentHybridConfig(
            enabled=True,
            alpha=float(getattr(ev, "hybrid_alpha", 0.70)),
            cold_start_alpha=float(getattr(ev, "hybrid_cold_start_alpha", 0.35)),
            pool_size=int(getattr(ev, "hybrid_pool_size", max(ev.top_n * 4, 80))),
            cold_start_threshold=int(getattr(ev, "hybrid_cold_start_threshold", ev.min_train_ratings)),
        ),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.experiment:
        cfg.experiment.name = args.experiment

    tracker = ExperimentTracker(cfg.experiment.name, cfg.experiment.output_dir)
    tracker.log_config(cfg)
    all_results = []
    ev = cfg.evaluation
    save_models = not args.no_save_models
    resume_artifacts = bool(args.resume_artifacts and not args.eval_only)

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

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 1: Popularity Baseline")
    logger.info("=" * 60)
    pop = PopularityBaseline()
    pop, pop_fit = _fit_or_load_model(
        pop,
        "Popularity Baseline",
        train_data,
        output_dir=cfg.experiment.output_dir,
        experiment_name=cfg.experiment.name,
        eval_only=args.eval_only,
        resume_artifacts=resume_artifacts,
        save_models=save_models,
    )
    _run_rating_eval(pop, "Popularity Baseline", test_data, tracker, all_results)
    _run_ranking_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker, all_results, pop_fit)
    _run_explainability_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker)
    _run_fairness_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker)
    _run_beyond_accuracy_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker)
    _run_cold_start_eval(pop, "Popularity Baseline", train_data, test_data, ev, tracker)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 2: Matrix Factorization (ALS, GPU)")
    logger.info("=" * 60)
    try:
        if args.eval_only:
            logger.info("  [MF] eval-only mode: loading saved artifact")
            mf = _load_saved_model(
                "Matrix Factorization (ALS)",
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
            )
            mf_fit = 0.0
            logger.info("  [MF] artifact loaded, running rating evaluation")
            _run_rating_eval(mf, "Matrix Factorization (ALS)", test_data, tracker, all_results)
            all_results[-1 if all_results[-1]["model"] == "Matrix Factorization (ALS)" else 0]["time"] = f"{mf_fit:.0f}s"
        else:
            logger.info("  [MF] resolving torch device")
            torch_device = _resolve_torch_device(use_gpu=bool(getattr(cfg.model, "use_gpu", True)))
            logger.info("  [MF] selected device: %s", torch_device)
            if torch_device != "cuda":
                logger.warning("CUDA not available or disabled; MF will run on CPU for this profile.")
            logger.info("  [MF] importing MatrixFactorizationGPU")
            from src.models import MatrixFactorizationGPU
            logger.info("  [MF] import complete")
            mf_epochs = getattr(cfg.model, "mf_epochs", 5)
            logger.info("  [MF] constructing model (n_factors=%s, epochs=%s)", cfg.model.n_factors, mf_epochs)
            mf = MatrixFactorizationGPU(
                n_factors=cfg.model.n_factors,
                reg=0.1,
                n_epochs=mf_epochs,
                device=torch_device,
            )
            logger.info("  [MF] model constructed, starting fit/load")
            mf, mf_fit = _fit_or_load_model(
                mf,
                "Matrix Factorization (ALS)",
                train_data,
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
                eval_only=args.eval_only,
                resume_artifacts=resume_artifacts,
                save_models=save_models,
            )
            logger.info("  [MF] fit/load finished in %.1fs, running rating evaluation", mf_fit)
            _run_rating_eval(mf, "Matrix Factorization (ALS)", test_data, tracker, all_results)
            all_results[-1 if all_results[-1]["model"] == "Matrix Factorization (ALS)" else 0]["time"] = f"{mf_fit:.0f}s"
    except (ImportError, OSError) as exc:
        logger.warning("Skipping MF due to dependency/runtime issue: %s", exc)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 3: BPR (custom PyTorch)")
    logger.info("=" * 60)
    try:
        if args.eval_only:
            bpr_custom = _load_saved_model(
                "BPR (custom PyTorch)",
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
            )
            bpr_fit = 0.0
        else:
            from src.models import BPRMatrixFactorization
            bpr_custom = BPRMatrixFactorization(
                n_factors=cfg.model.n_factors,
                n_epochs=cfg.model.n_epochs,
                lr=cfg.model.lr,
                reg=cfg.model.reg,
                pos_threshold=ev.relevance_threshold,
                device=_resolve_torch_device(use_gpu=bool(getattr(cfg.model, "use_gpu", True))),
            )
            bpr_custom, bpr_fit = _fit_or_load_model(
                bpr_custom,
                "BPR (custom PyTorch)",
                train_data,
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
                eval_only=args.eval_only,
                resume_artifacts=resume_artifacts,
                save_models=save_models,
            )
        _run_ranking_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker, all_results, bpr_fit)
        _run_explainability_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_fairness_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_beyond_accuracy_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_cold_start_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker)
        try:
            _run_hybrid_eval(bpr_custom, "BPR (custom PyTorch)", train_data, test_data, ev, tracker, all_results)
        except ValueError as exc:
            logger.warning("Skipping BPR (custom PyTorch) hybrid due to text-profile issue: %s", exc)
    except (ImportError, OSError, FileNotFoundError) as exc:
        logger.warning("Skipping BPR (custom PyTorch) due to dependency/runtime issue: %s", exc)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 4: BPR (implicit library)")
    logger.info("=" * 60)
    try:
        if args.eval_only:
            bpr_impl = _load_saved_model(
                "BPR (implicit library)",
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
            )
            bpr_impl_fit = 0.0
        else:
            from src.models import ImplicitBPRRanker
            bpr_impl = ImplicitBPRRanker(
                n_factors=cfg.model.n_factors,
                n_epochs=cfg.model.n_epochs,
                lr=cfg.model.lr,
                reg=cfg.model.reg,
                pos_threshold=ev.relevance_threshold,
                use_gpu=bool(getattr(cfg.model, "use_gpu", True)),
            )
            bpr_impl, bpr_impl_fit = _fit_or_load_model(
                bpr_impl,
                "BPR (implicit library)",
                train_data,
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
                eval_only=args.eval_only,
                resume_artifacts=resume_artifacts,
                save_models=save_models,
            )
        _run_ranking_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker, all_results, bpr_impl_fit)
        _run_explainability_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker)
        _run_fairness_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker)
        _run_beyond_accuracy_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker)
        _run_cold_start_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker)
        try:
            _run_hybrid_eval(bpr_impl, "BPR (implicit library)", train_data, test_data, ev, tracker, all_results)
        except ValueError as exc:
            logger.warning("Skipping BPR (implicit library) hybrid due to text-profile issue: %s", exc)
    except (ImportError, OSError, FileNotFoundError) as exc:
        logger.warning("Skipping BPR (implicit library) due to dependency/runtime issue: %s", exc)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 5: Implicit ALS (Hu-Koren-Volinsky)")
    logger.info("=" * 60)
    try:
        if args.eval_only:
            ials = _load_saved_model(
                "Implicit ALS",
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
            )
            ials_fit = 0.0
        else:
            from src.models import ImplicitALSRanker
            ials = ImplicitALSRanker(
                n_factors=cfg.model.n_factors,
                n_epochs=15,
                reg=cfg.model.reg,
                pos_threshold=ev.relevance_threshold,
                use_gpu=bool(getattr(cfg.model, "use_gpu", True)),
            )
            ials, ials_fit = _fit_or_load_model(
                ials,
                "Implicit ALS",
                train_data,
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
                eval_only=args.eval_only,
                resume_artifacts=resume_artifacts,
                save_models=save_models,
            )
        _run_ranking_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker, all_results, ials_fit)
        _run_explainability_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker)
        _run_fairness_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker)
        _run_beyond_accuracy_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker)
        _run_cold_start_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker)
        try:
            _run_hybrid_eval(ials, "Implicit ALS", train_data, test_data, ev, tracker, all_results)
        except ValueError as exc:
            logger.warning("Skipping Implicit ALS hybrid due to text-profile issue: %s", exc)
    except (ImportError, OSError, FileNotFoundError) as exc:
        logger.warning("Skipping Implicit ALS due to dependency/runtime issue: %s", exc)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL 6: WARP (custom PyTorch)")
    logger.info("=" * 60)
    try:
        if args.eval_only:
            warp = _load_saved_model(
                "WARP (custom PyTorch)",
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
            )
            warp_fit = 0.0
        else:
            from src.models import WARPModel
            warp = WARPModel(
                n_factors=cfg.model.n_factors,
                n_epochs=cfg.model.n_epochs,
                lr=cfg.model.lr,
                reg=cfg.model.reg,
                pos_threshold=ev.relevance_threshold,
                device=_resolve_torch_device(use_gpu=bool(getattr(cfg.model, "use_gpu", True))),
            )
            warp, warp_fit = _fit_or_load_model(
                warp,
                "WARP (custom PyTorch)",
                train_data,
                output_dir=cfg.experiment.output_dir,
                experiment_name=cfg.experiment.name,
                eval_only=args.eval_only,
                resume_artifacts=resume_artifacts,
                save_models=save_models,
            )
        _run_ranking_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker, all_results, warp_fit)
        _run_explainability_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_fairness_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_beyond_accuracy_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker)
        _run_cold_start_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker)
        try:
            _run_hybrid_eval(warp, "WARP (custom PyTorch)", train_data, test_data, ev, tracker, all_results)
        except ValueError as exc:
            logger.warning("Skipping WARP (custom PyTorch) hybrid due to text-profile issue: %s", exc)
    except (ImportError, OSError, FileNotFoundError) as exc:
        logger.warning("Skipping WARP (custom PyTorch) due to dependency/runtime issue: %s", exc)

    _print_results_table(all_results)

    tracker.record["results_table"] = all_results
    tracker.save()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
