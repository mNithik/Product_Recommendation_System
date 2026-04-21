from .rating import mae, rmse, evaluate_rating_prediction
from .recommendation import (
    precision_at_k,
    recall_at_k,
    f_measure,
    dcg_at_k,
    evaluate_beyond_accuracy,
    ndcg_at_k,
    evaluate_recommendations,
    evaluate_recommendations_per_user,
    compute_candidate_hit_rate,
)
from .explanation import (
    evaluate_counterfactuals,
    evaluate_explainable_recommendations,
    evaluate_explanations,
)
from .late_fusion import (
    cf_position_scores,
    rank_items_late_fusion,
    evaluate_late_fusion_recommendations,
)
from .fairness import (
    disparity_ratio,
    disparity_max_min_ratio,
    disparity_ratios_by_metric,
    run_activity_fairness_audit,
    summarize_cold_start_gap,
    summarize_metrics_by_train_activity,
    summarize_ndcg_by_train_activity,
)
from .cold_start import (
    compare_cold_start_benchmarks,
    evaluate_cold_start_benchmark,
    summarize_user_regimes,
)

__all__ = [
    "mae",
    "rmse",
    "precision_at_k",
    "recall_at_k",
    "f_measure",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_beyond_accuracy",
    "evaluate_rating_prediction",
    "evaluate_recommendations",
    "evaluate_recommendations_per_user",
    "compute_candidate_hit_rate",
    "evaluate_explanations",
    "evaluate_counterfactuals",
    "evaluate_explainable_recommendations",
    "cf_position_scores",
    "rank_items_late_fusion",
    "evaluate_late_fusion_recommendations",
    "disparity_ratio",
    "disparity_max_min_ratio",
    "disparity_ratios_by_metric",
    "run_activity_fairness_audit",
    "summarize_cold_start_gap",
    "summarize_metrics_by_train_activity",
    "summarize_ndcg_by_train_activity",
    "compare_cold_start_benchmarks",
    "evaluate_cold_start_benchmark",
    "summarize_user_regimes",
]
