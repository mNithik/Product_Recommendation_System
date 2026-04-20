# Recommender Systems Project

A recommender-systems project built on the **Amazon Arts, Crafts & Sewing 5-core** dataset. The repository preserves the original baseline models, cleanly separates rating prediction from Top-N recommendation, and adds modular post-hoc explainability, approximate counterfactual reasoning, fairness and beyond-accuracy evaluation, cold-start benchmarking, and an optional content-aware hybrid branch.

**56K users | 23K items | 494K interactions | 99.96% sparse | explicit + ranking baselines | explainability | interactive demo**

## Project Goals

This repository is organized around six distinct layers:

1. **Baseline rating prediction**
   Uses explicit-feedback models for MAE/RMSE evaluation.
2. **Baseline Top-N ranking and recommendation**
   Uses ranking-oriented models for Precision/Recall/F1/NDCG evaluation.
3. **Counterfactual explainability extension**
   Adds practical post-hoc explanations and weakening conditions for recommended items.
4. **Optional score adjustment**
   Adds a toggleable post-ranking score adjustment layer without replacing the base recommender.
5. **Optional content-aware hybrid branch**
   Adds a TF-IDF review-text fallback / re-ranking wrapper for cold-start and sparse-user comparison.
6. **Evaluation beyond average accuracy**
   Adds fairness, coverage, diversity, novelty, and cold-start benchmark reporting.

The project is meant to support both coursework and clear project reporting, so the code emphasizes modularity, explicit stage boundaries, reproducibility, and readable experiments over heavy end-to-end rewrites.

## Dataset

| Statistic | Value |
|---|---|
| **Source** | [Amazon Reviews - Arts, Crafts & Sewing 5-core](https://nijianmo.github.io/amazon/index.html) |
| **Users** | 56,210 |
| **Items** | 22,917 |
| **Interactions** | 494,485 |
| **Rating scale** | 1-5 |
| **Sparsity** | ~99.96% |

The 5-core subset ensures every user and item has at least 5 interactions.

## Project Pipeline

```text
Raw JSONL reviews
  -> preprocessing and per-user train/test split
  -> rating prediction branch
     -> Popularity, Matrix Factorization (ALS-style)
     -> MAE, RMSE
  -> ranking branch
     -> Popularity, BPR, WARP, Implicit ALS, Implicit BPR
     -> explicit ranking result
     -> final Top-N recommendation
     -> Precision@K, Recall@K, F1, NDCG@K
  -> post-hoc explanation layer
     -> supporting history items
     -> similar supporting items
     -> support confidence
  -> counterfactual layer
     -> minimal weakening condition
     -> weakening candidates with estimated impact
  -> optional score adjustment
     -> support boost
     -> popularity penalty
     -> adjusted ranking
  -> optional content-aware hybrid branch
     -> collaborative pool
     -> TF-IDF review-text similarity
     -> hybrid reranking
```

## Model-Task Separation

The repository enforces a cleaner separation between explicit rating prediction and Top-K recommendation.

### Rating Prediction

Used only for explicit rating prediction:

- `PopularityBaseline`
- `MatrixFactorizationGPU`

Evaluated with:

- `MAE`
- `RMSE`

### Ranking / Top-N Recommendation

Used only for Top-K ranking and recommendation:

- `PopularityBaseline`
- `BPRMatrixFactorization`
- `ImplicitBPRRanker`
- `ImplicitALSRanker`
- `WARPModel`

Evaluated with:

- `Precision@K`
- `Recall@K`
- `F1`
- `NDCG@K`

### Post-Hoc Extensions

Applied after ranking:

- structured explanation engine
- approximate counterfactual explanations
- optional score adjustment
- optional content-aware hybrid wrapper

These modules do **not** replace the underlying baseline recommender.

## Current Architecture

The codebase is organized around explicit stages.

### Core packages

- `src/preprocessing/`
  Data loading, splitting, and dataset generation.
- `src/models/`
  Baseline explicit and ranking models.
- `src/pipeline/`
  Explicit `ranking -> recommendation` stages.
- `src/explainability/`
  Item support, structured explanations, and counterfactual weakening.
- `src/postprocessing/`
  Optional score adjustment.
- `src/hybrid/`
  Optional collaborative + content hybrid wrapper.
- `src/evaluation/`
  Separate evaluation wrappers for rating, recommendation, explanation, fairness, and late fusion.
- `app/`
  Streamlit demo.

### Key files

- `main.py`
  Runs the offline baseline experiment pipeline.
- `app/demo.py`
  Interactive demo with ranking, recommendation, explanation, counterfactual, and causal-adjustment views.
- `ARCHITECTURE_AUDIT_AND_REFACTOR_PLAN.md`
  Architecture ownership and staged refactor plan.

## Explainability And Counterfactual Layer

The main extension is a practical post-hoc counterfactual explanation layer.

For each recommended item, the system can provide:

- supporting past user interactions
- supporting similar items
- a support/confidence score
- a counterfactual weakening statement
- weakening candidates with estimated impact

This is intentionally approximate. It is designed to be transparent and practical rather than a full reproduction of a heavy explainability architecture.

## Optional Causal-Inspired Adjustment

The repository also includes a lightweight post-ranking score adjustment layer.

When enabled, the layer:

- starts from the base ranking score
- adds a support-confidence boost
- subtracts a popularity penalty
- re-sorts items into an adjusted ranking

This is **not** a full causal graph or treatment-effect model. It is an interpretable post-hoc reweighting layer intended for controlled comparison against the unadjusted baseline.

## Optional Content-Aware Hybrid Branch

The project now also includes an optional lightweight hybrid branch built on top of the existing rankers.

When enabled, the hybrid branch:

- keeps the base collaborative model intact
- uses the collaborative model to generate a candidate pool
- computes TF-IDF review-text similarity between the user profile and candidate items
- reorders the collaborative pool with a configurable hybrid weight
- uses a lower collaborative weight for sparse-history users

This is designed as a practical cold-start / sparse-user extension, not as a replacement for the baseline ranking models.

## Evaluation Structure

Evaluation is now split by project concern:

- `src/evaluation/rating.py`
  Rating-prediction evaluation
- `src/evaluation/recommendation.py`
  Top-K recommendation evaluation
- `src/evaluation/explanation.py`
  Explanation and counterfactual coverage summaries
- `src/evaluation/fairness.py`
  Activity-bucket disparity audit
- `src/evaluation/cold_start.py`
  Sparse-user and warm-user benchmark summaries
- `src/evaluation/late_fusion.py`
  Optional text-based late-fusion reranking experiment
- `src/evaluation/recommendation.py`
  Also includes beyond-accuracy metrics such as coverage, diversity, novelty, and popularity concentration

## Current Project Status

The repository now supports the following report-ready evaluation views:

- rating prediction metrics for explicit-feedback baselines
- ranking metrics for Top-N recommenders
- explanation and counterfactual output coverage
- fairness gaps across user-activity groups
- beyond-accuracy metrics such as catalog coverage, diversity, novelty, and popularity concentration
- cold-start and sparse-user benchmarking
- hybrid-vs-base comparison for content-aware reranking

## Streamlit Demo

Launch the demo with:

```bash
streamlit run app/demo.py
```

The demo exposes the staged pipeline clearly:

- user history
- base ranked candidates
- optional content-aware hybrid reranking
- optional causal-adjusted ranking
- final Top-N recommendations
- explanation text
- support confidence
- supporting history and similar items
- counterfactual weakening conditions
- optional TF-IDF content-aware hybrid fallback
- fairness snapshot
- coverage, diversity, and novelty snapshot

## Quick Start

```bash
git clone https://github.com/mNithik/Product_Recommendation_System.git
cd Product_Recommendation_System
pip install -r requirements.txt

# Place Arts_Crafts_and_Sewing_5.json in:
# Arts_Crafts_and_Sewing_5.json/Arts_Crafts_and_Sewing_5.json

python main.py --config configs/default.yaml
streamlit run app/demo.py
pytest tests/ -v
```

## Configuration

Main experiment settings live in `configs/default.yaml`.

Example:

```yaml
model:
  type: "bpr"
  n_factors: 64
  n_epochs: 20
  lr: 0.001
  reg: 0.0001
  mf_epochs: 5

evaluation:
  top_n: 10
  relevance_threshold: 4.0
  min_train_ratings: 5
```

The Streamlit demo also exposes interactive controls for:

- recommendation model selection
- causal-adjustment toggle and weights
- content-aware hybrid toggle and weights

## Testing

The repository includes tests for:

- preprocessing
- recommendation metrics
- ranking/recommendation pipeline behavior
- popularity baseline behavior
- explanation output structure
- counterfactual output structure
- causal-adjustment toggle behavior
- explanation/counterfactual evaluation summaries
- late fusion
- fairness summaries
- beyond-accuracy recommendation metrics
- cold-start benchmark summaries
- content-aware hybrid wrapper
- review-text profile utilities

## Limitations

The project is intentionally practical rather than fully production-grade. The main current limitations are:

- optional dependencies such as `implicit` and some `torch` configurations may vary by environment
- counterfactual explanations are approximate and post-hoc rather than model-intrinsic
- the causal layer is a lightweight score adjustment, not a full causal graph model
- the content-aware hybrid branch uses TF-IDF review text rather than a heavier neural content encoder
- offline experiments can be time-consuming on large models and full data

## Future Work

Implemented:

- explicit baseline rating prediction
- explicit baseline Top-N ranking
- modular ranking/recommendation pipeline
- structured explanation engine
- approximate counterfactual weakening layer
- optional score adjustment layer
- optional content-aware hybrid branch
- fairness, beyond-accuracy, and cold-start evaluation layers
- cleaner evaluation split
- Streamlit demo reflecting the staged pipeline

Not implemented yet or intentionally deferred:

- a full causal collaborative filtering model
- a PETER-style Transformer branch
- a production retrieval stack
- full neural explainability architectures
- deployment-oriented model serving and caching

Those remain valid future-work directions, but the current repository already supports a coherent project report around baseline recommendation, explainability, counterfactual analysis, and optional causal-style post-processing.
