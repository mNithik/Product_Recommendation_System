"""Configuration for the recommender system project."""

# Paths
DATA_DIR = "Toys_and_Games_5.json"
RAW_DATA_PATH = f"{DATA_DIR}/Toys_and_Games_5.json"
TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"

# Split
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# Recommendation
TOP_N = 10
MAX_CANDIDATES = 10000
MIN_TRAIN_RATINGS = 5

# GPU: use Matrix Factorization on GPU when True (faster); else Item-based CF on CPU
USE_GPU = True
