# Product Recommendation System

Offline recommender system for **Amazon Toys & Games** review data (5-core). Built for **CS550**: per-user train/test split, rating prediction, and Top-N recommendation with standard evaluation metrics.

## Features

- **Preprocessing**: Loads review JSONL, applies an **80/20 split per user** (seeded), writes `data/train.json` and `data/test.json`.
- **Models** (see `config.py`):
  - **Matrix Factorization (GPU)** — PyTorch ALS-style training with user/item biases when `USE_GPU = True` and CUDA is available.
  - **Item-based collaborative filtering (CPU)** — Adjusted cosine similarity, *k* nearest neighbors; used when GPU MF is disabled or CUDA is unavailable.
- **Metrics**: **MAE**, **RMSE** (rating prediction); **Precision@N**, **Recall@N**, **F-measure**, **NDCG@N** (Top-N). MF mode also reports a **candidate hit rate** diagnostic.

## Requirements

- Python 3.9+ recommended
- Dependencies: `pip install -r requirements.txt`
- **GPU (optional)**: CUDA-capable GPU + PyTorch with CUDA for the MF path; otherwise the pipeline falls back to item-based CF on CPU.

## Dataset

The raw dataset is **not** included in the repository (large; see `.gitignore`, on the order of hundreds of MB).

1. Obtain the **Amazon Toys & Games 5-core** reviews file (JSON lines), e.g. from [Julian McAuley’s Amazon review datasets](https://jmcauley.ucsd.edu/data/amazon/).
2. Create this layout at the project root:

   ```text
   Product Recommendation System/
   └── Toys_and_Games_5.json/
       └── Toys_and_Games_5.json   # one JSON object per line
   ```

   Each line should include at least `reviewerID`, `asin`, and `overall` (1–5 rating). Optional fields `reviewText` and `summary` are preserved in the train/test files.

## Quick start

```bash
cd "Product Recommendation System"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

- First run: if `data/train.json` and `data/test.json` are missing, `main.py` runs preprocessing on `Toys_and_Games_5.json/Toys_and_Games_5.json`, then trains and evaluates.
- Later runs: if train/test already exist, preprocessing is skipped.

To **rebuild** the split only:

```bash
python preprocess.py
```

## Configuration

Edit `config.py`:

| Setting | Role |
|--------|------|
| `RAW_DATA_PATH` / `DATA_DIR` | Location of raw JSONL |
| `TRAIN_PATH`, `TEST_PATH` | Output split files |
| `TRAIN_RATIO`, `RANDOM_STATE` | Split fraction and seed |
| `TOP_N` | List length for ranking metrics |
| `MAX_CANDIDATES` | Cap on candidate items for MF Top-N |
| `MIN_TRAIN_RATINGS` | Minimum train ratings per user to include in Top-N evaluation |
| `USE_GPU` | Prefer MF on GPU when `True` |

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | End-to-end pipeline: preprocess (if needed), fit, evaluate |
| `preprocess.py` | Standalone preprocessing |
| `recommender.py` | Models, data loading, metrics |
| `config.py` | Paths and hyperparameters |
| `requirements.txt` | Python dependencies |

## License / course use

See course materials (`CS550-Project-description.pdf`) for submission and attribution requirements. Dataset terms apply per the original Amazon data release.
