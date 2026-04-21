# Recommender System for Sparse E-Commerce Ratings

This repository implements a recommender system on the **Amazon Arts, Crafts & Sewing 5-core** dataset. It preserves the original working baselines, separates **rating prediction** from **Top-N ranking and recommendation**, and adds modular analysis layers for **explainability**, **counterfactual reasoning**, **fairness**, **beyond-accuracy evaluation**, **cold-start benchmarking**, and **content-aware hybrid reranking**.

**56K users | 23K items | 494K interactions | 99.96% sparsity | explicit-feedback branch | Top-N ranking branch | explainability | Streamlit demo**

## What This Project Includes

- explicit rating prediction with `PopularityBaseline` and `MatrixFactorizationGPU`
- Top-N ranking with popularity, custom PyTorch `BPR`, custom PyTorch `WARP`, `implicit` BPR, and `implicit` ALS
- a clean `ranking -> recommendation` pipeline instead of mixing model scoring and UI output
- post-hoc structured explanations with support evidence and weakening conditions
- beyond-accuracy reporting: coverage, diversity, novelty, popularity concentration
- subgroup analysis: fairness summaries and cold-vs-warm user benchmarking
- an optional TF-IDF content-aware hybrid reranker
- a Streamlit demo for browsing users, ranked candidates, recommendations, and explanations

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

The code emphasizes modularity, explicit stage boundaries, reproducibility, and readable experiments over heavy end-to-end rewrites.

## Best Current Results

These numbers come from the documented full-data run stored under `experiments/`.

### Rating prediction

| Model | MAE | RMSE |
|---|---:|---:|
| Popularity baseline | 0.6234 | 0.9340 |
| Matrix factorization (ALS-style, PyTorch) | **0.4637** | **0.8194** |

### Top-10 ranking

| Model | P@10 | R@10 | F1 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity baseline | 0.0023 | 0.0087 | 0.0036 | 0.0086 |
| BPR (custom PyTorch) | 0.0019 | 0.0072 | 0.0030 | 0.0079 |
| BPR (`implicit`) | 0.0009 | 0.0033 | 0.0015 | 0.0021 |
| WARP (custom PyTorch) | 0.0154 | 0.0618 | 0.0247 | 0.0478 |
| Implicit ALS | **0.0190** | **0.0747** | **0.0303** | **0.0674** |

### Beyond-accuracy highlights

- **Best coverage:** `WARP + content hybrid` (`0.0419`)
- **Best novelty:** `BPR (custom PyTorch)` (`6.0751`)
- **Strongest overall Top-N ranker:** `Implicit ALS`
- **Largest cold-user gain from hybrid reranking:** `BPR + content hybrid`

## What The Metrics Mean

- `MAE`, `RMSE`: lower is better; these measure rating-prediction error
- `Precision@10`: higher is better; more of the displayed Top-10 items are relevant
- `Recall@10`: higher is better; more of the user’s relevant held-out items are recovered
- `F1`: higher is better; balances precision and recall
- `NDCG@10`: higher is better; rewards relevant items appearing higher in the ranked list
- `Coverage`: higher is better; more of the catalog appears at least once
- `Diversity`: higher is better; recommendation lists are less repetitive
- `Novelty`: higher is better; recommendations are less dominated by obvious popular items
- `PopularityConcentration`: lower is better; recommendations are less collapsed onto a narrow head of items
- `Cold/Warm gap`: closer to zero means more balanced behavior across sparse-history and rich-history users

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

## Current Capabilities

The repository currently supports:

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

### Deploy on Streamlit Community Cloud

1. Push your branch to GitHub.
2. In Streamlit Community Cloud, click **Create app**.
3. Select repository + branch and set entrypoint to `app/demo.py`.
4. Deploy.

The demo automatically uses `data/train.json` and `data/test.json` when available, and falls back to `data_small/` for lightweight cloud startup.
For cloud runs, you can also set environment variables `TRAIN_JSON_URL` and `TEST_JSON_URL`.
If full data files are missing, the app will try downloading them first, then fall back to `data_small/` if download fails.

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

## Repository Layout

- `main.py`: main offline training/evaluation pipeline
- `app/demo.py`: Streamlit demo
- `configs/default.yaml`: default experiment settings
- `configs/gpu_profile.yaml`: WSL/GPU-oriented run profile
- `src/models/`: baseline rating and ranking models
- `src/pipeline/`: explicit ranking and recommendation stages
- `src/explainability/`: explanation engine, item support, counterfactual reasoning
- `src/postprocessing/`: optional score-adjustment layer
- `src/hybrid/`: TF-IDF content-aware reranking wrapper
- `src/evaluation/`: rating, recommendation, fairness, explanation, and cold-start evaluation
- `tests/`: regression and evaluation tests
- `report/`: project report sources

## Quick Start

```bash
git clone https://github.com/mNithik/Product_Recommendation_System.git
cd Product_Recommendation_System
pip install -r requirements.txt

# Place Arts_Crafts_and_Sewing_5.json in:
# Arts_Crafts_and_Sewing_5.json/Arts_Crafts_and_Sewing_5.json
```

Then run the default experiment:

```bash
python main.py --config configs/default.yaml --experiment baseline_run
```

## Common Run Commands

### CPU / default run

```bash
python main.py --config configs/default.yaml --experiment baseline_run
```

### Evaluation-only run from saved artifacts

```bash
python main.py --config configs/default.yaml --experiment baseline_run --eval-only
```

### Resume a partially completed run

```bash
python main.py --config configs/gpu_profile.yaml --experiment wsl_gpu_run --resume-artifacts
```

### WSL GPU run

From Ubuntu/WSL:

```bash
cd /mnt/c/Users/nithi/OneDrive/Documents/SPRING2026/DATAMINING/PROJECT
source .venv-wsl-gpu/bin/activate
python main.py --config configs/gpu_profile.yaml --experiment wsl_gpu_run
```

### Streamlit demo

```bash
streamlit run app/demo.py
```

## WSL GPU Setup

For `implicit` CUDA support on Windows, the recommended path is **WSL2 + Ubuntu**, not native Windows Python.

Project helper scripts:

- `scripts/setup_implicit_gpu_wsl.sh`
- `setup_implicit_gpu_wsl.ps1`
- `run_gpu_profile_wsl.ps1`
- `scripts/verify_implicit_gpu.py`

Typical WSL flow:

```bash
cd /mnt/c/Users/nithi/OneDrive/Documents/SPRING2026/DATAMINING/PROJECT
bash scripts/setup_implicit_gpu_wsl.sh
source .venv-wsl-gpu/bin/activate
python scripts/verify_implicit_gpu.py
python main.py --config configs/gpu_profile.yaml --experiment wsl_gpu_run
```

If verification prints `HAS_CUDA: True`, the `implicit` GPU path is available in WSL.

## Tests

Run the test suite with:

```bash
pytest
```

Targeted examples:

```bash
pytest tests/test_cold_start.py
pytest tests/test_content_hybrid.py
pytest tests/test_counterfactual.py
```

## Reproducibility Notes

- Train/test splitting is user-level with a fixed random seed by default
- Ranking models are evaluated separately from the explicit rating branch
- Hybrid reranking is optional and does not replace the underlying collaborative model
- Model artifacts are stored under `experiments/<run_name>/models/`
- JSON summaries are written under `experiments/`

## Demo / Cloud Note

For local runs, the demo uses `data/train.json` and `data/test.json` when available.

For Streamlit Community Cloud:

- the app can run from `data_small/` for lightweight startup
- full data is better kept for local/WSL experiments because free cloud memory is limited

## Project Write-Up

The write-up source is:

- `report/CS550_project_report.tex`

It documents:

- rating prediction results
- Top-N ranking results
- beyond-accuracy metrics
- cold-vs-warm subgroup analysis
- hybrid-vs-base comparisons
- explainability and counterfactual extensions

## License / Attribution

This project uses the Amazon Reviews dataset and builds on open-source libraries including PyTorch, scikit-learn, Streamlit, and `implicit`.
