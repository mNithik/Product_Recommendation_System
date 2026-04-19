# Product Recommendation System

A research-oriented recommender-systems project built on the **Amazon Arts, Crafts & Sewing 5-core** dataset. The repository preserves the original baseline models, cleanly separates rating prediction from Top-N recommendation, and adds modular post-hoc explainability, approximate counterfactual reasoning, and an optional lightweight causal-adjustment layer.

**56K users | 23K items | 494K interactions | 99.96% sparse | explicit + ranking baselines | explainability | interactive demo**

## Project Goals

This repository is organized around four distinct layers:

1. **Baseline rating prediction**
   Uses explicit-feedback models for MAE/RMSE evaluation.
2. **Baseline Top-N ranking and recommendation**
   Uses ranking-oriented models for Precision/Recall/F1/NDCG evaluation.
3. **Counterfactual explainability extension**
   Adds practical post-hoc explanations and weakening conditions for recommended items.
4. **Optional causal-inspired adjustment**
   Adds a toggleable post-ranking score adjustment layer without replacing the base recommender.

The project is meant to support both coursework and research-style reporting, so the code emphasizes modularity, explicit stage boundaries, reproducibility, and readable experiments over heavy end-to-end rewrites.

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

## Research Pipeline

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
  -> optional causal-inspired adjustment
     -> support boost
     -> popularity penalty
     -> adjusted ranking
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
- optional causal-inspired score adjustment

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
  Optional causal-inspired score adjustment.
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

The main research extension is a practical CountER-inspired post-hoc explanation layer.

For each recommended item, the system can provide:

- supporting past user interactions
- supporting similar items
- a support/confidence score
- a counterfactual weakening statement
- weakening candidates with estimated impact

This is intentionally approximate. It is designed to be transparent and practical rather than a full reproduction of a heavy explainability architecture.

## Optional Causal-Inspired Adjustment

The repository also includes a lightweight causal-inspired post-ranking adjustment layer.

When enabled, the layer:

- starts from the base ranking score
- adds a support-confidence boost
- subtracts a popularity penalty
- re-sorts items into an adjusted ranking

This is **not** a full causal graph or treatment-effect model. It is an interpretable post-hoc reweighting layer intended for controlled comparison against the unadjusted baseline.

## Evaluation Structure

Evaluation is now split by research concern:

- `src/evaluation/rating.py`
  Rating-prediction evaluation
- `src/evaluation/recommendation.py`
  Top-K recommendation evaluation
- `src/evaluation/explanation.py`
  Explanation and counterfactual coverage summaries
- `src/evaluation/fairness.py`
  Activity-bucket disparity audit
- `src/evaluation/late_fusion.py`
  Optional text-based late-fusion reranking experiment

## Streamlit Demo

Launch the demo with:

```bash
streamlit run app/demo.py
```

The demo exposes the staged pipeline clearly:

- user history
- base ranked candidates
- optional causal-adjusted ranking
- final Top-N recommendations
- explanation text
- support confidence
- supporting history and similar items
- counterfactual weakening conditions
- optional TF-IDF / Sentence-BERT context

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
- dataset size selection
- explanation display
- TF-IDF / Sentence-BERT context
- causal-adjustment toggle and weights

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
- review-text profile utilities

## Current Scope And Future Work

Implemented:

- explicit baseline rating prediction
- explicit baseline Top-N ranking
- modular ranking/recommendation pipeline
- structured explanation engine
- approximate counterfactual weakening layer
- optional causal-inspired adjustment layer
- cleaner evaluation split
- Streamlit demo reflecting the staged pipeline

Not implemented yet or intentionally deferred:

- a full causal collaborative filtering model
- a PETER-style Transformer branch
- a production retrieval stack
- full neural explainability architectures

Those remain valid future-work directions, but the current repository already supports a coherent research report around baseline recommendation, explainability, counterfactual analysis, and optional causal-style post-processing.
