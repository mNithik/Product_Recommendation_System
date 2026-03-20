"""
Main pipeline for the CS550 Recommender System project.
Runs preprocessing (if needed), rating prediction, and Top-N recommendation evaluation.
Uses GPU-accelerated Matrix Factorization when USE_GPU=True and CUDA is available.
"""

import os

from config import MAX_CANDIDATES, MIN_TRAIN_RATINGS, RAW_DATA_PATH, TEST_PATH, TOP_N, TRAIN_PATH, USE_GPU
from preprocess import run_preprocessing
from recommender import (
    ItemBasedCF,
    MatrixFactorizationGPU,
    compute_candidate_hit_rate,
    evaluate_rating_prediction,
    evaluate_recommendations,
    load_data,
)


def _get_model():
    if USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                return MatrixFactorizationGPU(n_factors=128, reg=0.1, n_epochs=10), "Matrix Factorization (GPU)"
        except ImportError:
            pass
    return ItemBasedCF(k=50), "Item-based CF (CPU)"


def main():
    # Step 1: Preprocess if train/test don't exist
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("=== Step 1: Data Preprocessing ===")
        if not os.path.exists(RAW_DATA_PATH):
            print(f"Raw data not found at {RAW_DATA_PATH}. Please ensure the JSON file exists.")
            return
        run_preprocessing(RAW_DATA_PATH, TRAIN_PATH, TEST_PATH)
    else:
        print("Train/test data found. Skipping preprocessing.")

    print("\n=== Loading Data ===")
    train_data = load_data(TRAIN_PATH)
    test_data = load_data(TEST_PATH)
    print(f"Train: {len(train_data):,} | Test: {len(test_data):,}")

    # Step 2: Fit model and evaluate rating prediction
    model, model_name = _get_model()
    print(f"\n=== Step 2: Rating Prediction ({model_name}) ===")
    print("Fitting model...")
    model.fit(train_data)

    print("Evaluating rating prediction...")
    rating_metrics = evaluate_rating_prediction(model, test_data)
    print(f"  MAE:  {rating_metrics['MAE']:.4f}")
    print(f"  RMSE: {rating_metrics['RMSE']:.4f}")

    # Step 3: Top-N recommendation evaluation
    print(f"\n=== Step 3: Top-{TOP_N} Recommendation ===")
    if hasattr(model, "_get_candidates"):
        hit_rate = compute_candidate_hit_rate(model, train_data, test_data, max_candidates=MAX_CANDIDATES, min_train_ratings=MIN_TRAIN_RATINGS)
        if hit_rate is not None:
            print(f"  Candidate hit rate (test items in candidate pool): {hit_rate:.2%}")
    rec_metrics = evaluate_recommendations(model, train_data, test_data, top_n=TOP_N, min_train_ratings=MIN_TRAIN_RATINGS, max_candidates=MAX_CANDIDATES)
    if "n_users_eval" in rec_metrics:
        print(f"  Users evaluated (5+ train ratings): {rec_metrics['n_users_eval']:,}")
    print(f"  Precision:  {rec_metrics['Precision']:.4f}")
    print(f"  Recall:     {rec_metrics['Recall']:.4f}")
    print(f"  F-measure:  {rec_metrics['F-measure']:.4f}")
    print(f"  NDCG:       {rec_metrics['NDCG']:.4f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
