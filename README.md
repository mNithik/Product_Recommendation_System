# Product Recommendation System

A reproducible recommendation engine built on the **Amazon Arts, Crafts & Sewing 5-core** dataset. Implements six models — from a popularity baseline to GPU-accelerated matrix factorization, BPR, and WARP ranking — with full evaluation and a Streamlit demo for interactive exploration.

---

## Problem

Users interact with a vast catalog of products, but only rate a tiny fraction. The resulting user-item matrix is extremely sparse, making it difficult to predict preferences. This project addresses that challenge by:

- Learning latent user and item representations from observed ratings
- Predicting explicit ratings for unseen user-item pairs (MAE, RMSE)
- Generating personalized **Top-10** recommendation lists ranked by relevance (Precision, Recall, F1, NDCG)
- Comparing **rating-optimized** vs **ranking-optimized** training objectives

## Dataset

| Statistic | Value |
|---|---|
| **Source** | [Amazon Reviews — Arts, Crafts & Sewing 5-core](https://nijianmo.github.io/amazon/index.html) |
| **Users** | 56,210 |
| **Items** | 22,917 |
| **Interactions** | 494,485 |
| **Rating scale** | 1 – 5 |
| **Sparsity** | ~99.96 % |

> *5-core* means every user and item has at least 5 reviews.

## Pipeline

```
Raw JSONL data (494K reviews)
    │
    ▼
Preprocessing (per-user 80/20 random split)
    │
    ├──► train.json (375,028)
    └──► test.json  (119,457)
           │
           ▼
    ┌─────────────────────────────────┐
    │  Rating Prediction              │
    │  Popularity, MF (ALS on GPU)    │──► MAE, RMSE
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  Top-10 Ranking                 │
    │  Popularity, BPR (custom),      │
    │  BPR (implicit), Implicit ALS,  │──► Precision@10, Recall@10, F1, NDCG@10
    │  WARP (custom)                  │
    └─────────────────────────────────┘
           │
           ▼
    Experiment JSON + Results Table
```

## Models Implemented

| Model | Type | Built From Scratch | Use |
|---|---|---|---|
| **Popularity Baseline** | Non-personalized, most-popular items | Yes | Baseline |
| **Matrix Factorization (ALS)** | Latent factor model with biases, GPU-accelerated | Yes (PyTorch) | Rating prediction |
| **BPR (custom)** | Pairwise ranking with hard negative sampling | Yes (PyTorch) | Top-N ranking |
| **BPR (implicit library)** | Optimized C++/CUDA BPR backend | No (library) | Comparison |
| **Implicit ALS** | Weighted implicit feedback MF | No (library) | Top-N ranking |
| **WARP (custom)** | Approximate-rank pairwise loss, top-of-list optimized | Yes (PyTorch) | Top-N ranking |

## Results

### Rating Prediction

| Model | MAE ↓ | RMSE ↓ |
|---|---|---|
| Popularity Baseline | 0.6234 | 0.9340 |
| **Matrix Factorization (ALS)** | **0.4632** | **0.8193** |

### Top-10 Recommendation

| Model | P@10 ↑ | R@10 ↑ | F1 ↑ | NDCG@10 ↑ | Time |
|---|---|---|---|---|---|
| Popularity Baseline | 0.0023 | 0.0087 | 0.0036 | 0.0086 | 1s |
| BPR (custom PyTorch) | 0.0019 | 0.0072 | 0.0030 | 0.0079 | 273s |
| BPR (implicit library) | 0.0007 | 0.0027 | 0.0011 | 0.0015 | 8s |
| **WARP (custom PyTorch)** | **0.0148** | **0.0590** | **0.0237** | **0.0462** | 65s |
| **Implicit ALS** | **0.0188** | **0.0735** | **0.0300** | **0.0667** | 11s |

> **Key finding:** Models trained with rating-prediction objectives (MF/ALS for RMSE) perform poorly on ranking tasks. Ranking-optimized models (WARP, Implicit ALS) achieve 6–8x better Precision@10 and NDCG@10, confirming that the training objective must match the evaluation metric.

## Demo

Interactive Streamlit app with four tabs:

- **Recommend for User** — select a user and model, see Top-N recommendations with explainability
- **Model Comparison** — full results table with bar charts comparing all 6 models
- **Popular Items** — browse the most-rated items
- **Dataset Explorer** — rating distributions, interactions per user, sample data

```bash
streamlit run app/demo.py
```

### Features
- Model selection (Implicit ALS / WARP / BPR custom / BPR implicit / Popularity) in sidebar
- Adjustable number of recommendations
- **Explainability**: each recommendation shows *why* it was suggested
- Cold-start fallback to popularity for new users
- Dataset statistics dashboard

## Project Structure

```
recommender-system/
├── app/
│   └── demo.py             # Streamlit demo
├── data/                    # Generated train/test splits (gitignored)
├── src/
│   ├── preprocessing/       # Data loading, per-user splitting, subsampling
│   ├── models/
│   │   ├── popularity.py    # Popularity baseline
│   │   ├── item_cf.py       # Item-based CF (from scratch)
│   │   ├── matrix_factorization.py  # ALS on GPU (from scratch)
│   │   ├── bpr.py           # BPR with hard negatives (from scratch)
│   │   ├── warp.py          # WARP loss (from scratch)
│   │   ├── implicit_bpr.py  # BPR via implicit library
│   │   └── implicit_als.py  # ALS via implicit library
│   ├── evaluation/          # MAE, RMSE, Precision, Recall, F1, NDCG
│   └── utils/               # Config loader, data loader, experiment tracker
├── configs/                 # YAML experiment configs
├── tests/                   # pytest unit & integration tests (40 tests)
├── experiments/             # Auto-saved experiment JSONs
├── requirements.txt
├── README.md
└── main.py                  # Runs all 6 models head-to-head
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/mNithik/Product_Recommendation_System.git
cd Product_Recommendation_System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset
#    Download Arts_Crafts_and_Sewing_5.json into Arts_Crafts_and_Sewing_5.json/

# 4. Run the full pipeline (all 6 models, ~20 min)
python main.py --config configs/default.yaml

# 5. Launch demo
streamlit run app/demo.py

# 6. Run tests
pytest tests/ -v
```

## Configuration

All hyperparameters are controlled via YAML:

```yaml
model:
  type: "bpr"               # bpr | bpr_implicit | item_cf | matrix_factorization
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

Each run auto-saves config + all metrics to `experiments/<name>_<timestamp>.json`.

## Future Improvements

- **Hybrid recommendations** — combine CF with item metadata / review text embeddings (TF-IDF, sentence transformers)
- **LightGCN** — graph convolution over the user-item bipartite graph for stronger ranking
- **Cold-start handling** — content-based fallback using review text for new items
- **ANN retrieval** — FAISS/ScaNN for sub-linear candidate generation at scale
- **Online serving** — FastAPI endpoint with cached embeddings for real-time inference
