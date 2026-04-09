# Product Recommendation System

A reproducible recommendation engine built on the **Amazon Arts, Crafts & Sewing 5-core** dataset. Implements six models — from a popularity baseline to GPU-accelerated matrix factorization, BPR, and WARP ranking — with full evaluation and a Streamlit demo for interactive exploration.

**56K users | 23K items | 494K interactions | 99.96% sparse | 6 models | full train/test evaluation | interactive demo**

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

| Model | MAE ↓ | RMSE ↓ | vs Baseline |
|---|---|---|---|
| Popularity Baseline | 0.6234 | 0.9340 | — |
| **Matrix Factorization (ALS)** | **0.4632** | **0.8193** | **−26% MAE, −12% RMSE** |

> MF with learned user/item biases on GPU closes the gap substantially. On a 1–5 scale the model's predictions are off by less than half a star on average.

### Top-10 Recommendation

| Model | P@10 ↑ | R@10 ↑ | F1 ↑ | NDCG@10 ↑ | vs Baseline (NDCG) | Time |
|---|---|---|---|---|---|---|
| Popularity Baseline | 0.0023 | 0.0087 | 0.0036 | 0.0086 | — | 1s |
| BPR (custom PyTorch) | 0.0019 | 0.0072 | 0.0030 | 0.0079 | −8% | 273s |
| BPR (implicit library) | 0.0007 | 0.0027 | 0.0011 | 0.0015 | −83% | 8s |
| **WARP (custom PyTorch)** | **0.0148** | **0.0590** | **0.0237** | **0.0462** | **+437%** | 65s |
| **Implicit ALS** | **0.0188** | **0.0735** | **0.0300** | **0.0667** | **+676%** | 11s |

### What These Numbers Mean

These absolute values look small, but **context matters**:

- **99.96% sparsity** — each user has rated only ~9 of 22,917 items. Predicting the handful of relevant items from tens of thousands of candidates is inherently hard.
- **Popularity is a strong baseline** for sparse datasets — it outperforms both BPR variants, which is a well-documented phenomenon when interaction data is thin.
- **Implicit ALS** achieves **+676% NDCG** over the popularity baseline, meaning it ranks relevant items nearly 8× higher in the top-10 list.
- **WARP** (built from scratch in PyTorch) achieves **+437% NDCG** — the best among all from-scratch models and competitive with production-grade library code.
- **Rating ≠ Ranking** — MF achieves excellent RMSE but generates poor top-10 lists. This confirms the widely studied mismatch between rating-prediction and ranking objectives (Cremonesi et al., 2010).

> **Key takeaway:** The training objective must match the evaluation task. Optimizing for RMSE does not produce good recommendation lists — pairwise ranking losses (BPR, WARP) are essential for top-N recommendation.

## Interactive Demo

> **Try it yourself** — the Streamlit app lets you pick any user, choose a model, and see personalized recommendations with explanations of *why* each item was suggested.

```bash
streamlit run app/demo.py
```

| Tab | What It Shows |
|---|---|
| **Recommend for User** | Select a user + model → Top-N recommendations with per-item explainability |
| **Model Comparison** | Side-by-side results table + bar charts for all 6 models |
| **Popular Items** | Browse the 30 most-rated items in the catalog |
| **Dataset Explorer** | Rating distributions, interactions per user, sample data |

### Highlights

- **Explainability** — each recommendation shows the reasoning: overlapping user communities, taste-profile match, or popularity fallback
- **Cold-start handling** — new/unseen users automatically fall back to the popularity baseline
- **Model switching** — swap between Implicit ALS, WARP, BPR, and Popularity in the sidebar and compare recommendations live
- **Dataset toggle** — switch between the full 494K-interaction dataset and a 5K-user subsample for faster iteration

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
