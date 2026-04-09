"""
Streamlit demo for the Product Recommendation System.

Run:
    streamlit run app/demo.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Product Recommender — Arts, Crafts & Sewing",
    page_icon="🎨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        color: white;
    }
    .rec-card h4 { margin: 0 0 8px 0; color: white; }
    .rec-card p { margin: 2px 0; opacity: 0.9; }
    .metric-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-box h2 { margin: 0; color: #667eea; }
    .metric-box p { margin: 4px 0 0; color: #555; }
    .explain-box {
        background: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
    }
    .winner-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_train_test(train_path: str, test_path: str):
    train = load_data(train_path)
    test = load_data(test_path)
    return train, test


@st.cache_data(show_spinner="Building indices...")
def build_user_item_maps(train_data_json: str):
    train_data = json.loads(train_data_json)
    user_items = defaultdict(list)
    item_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "users": []})

    for r in train_data:
        uid = r["reviewerID"]
        iid = r["asin"]
        rating = float(r["overall"])
        user_items[uid].append({"item": iid, "rating": rating})
        item_stats[iid]["sum"] += rating
        item_stats[iid]["count"] += 1
        item_stats[iid]["users"].append(uid)

    for iid in item_stats:
        s = item_stats[iid]
        s["avg_rating"] = s["sum"] / s["count"]

    return dict(user_items), dict(item_stats)


@st.cache_resource(show_spinner="Fitting recommendation model...")
def fit_model(train_path: str, model_type: str):
    train_data = load_data(train_path)

    if model_type == "Popularity Baseline":
        from src.models import PopularityBaseline
        model = PopularityBaseline()
        model.fit(train_data)
        return model

    if model_type == "Implicit ALS (best)":
        from src.models import ImplicitALSRanker
        model = ImplicitALSRanker(n_factors=64, n_epochs=15, reg=0.0001, pos_threshold=4.0)
        model.fit(train_data)
        return model

    if model_type == "WARP (custom PyTorch)":
        from src.models import WARPModel
        model = WARPModel(n_factors=64, n_epochs=20, lr=0.001, reg=0.0001, pos_threshold=4.0)
        model.fit(train_data)
        return model

    if model_type == "BPR (custom PyTorch)":
        from src.models import BPRMatrixFactorization
        model = BPRMatrixFactorization(n_factors=64, n_epochs=20, lr=0.001, reg=0.0001, pos_threshold=4.0)
        model.fit(train_data)
        return model

    if model_type == "BPR (implicit library)":
        from src.models import ImplicitBPRRanker
        model = ImplicitBPRRanker(n_factors=64, n_epochs=20, lr=0.01, reg=0.01, pos_threshold=4.0)
        model.fit(train_data)
        return model

    from src.models import PopularityBaseline
    model = PopularityBaseline()
    model.fit(train_data)
    return model


@st.cache_data
def load_experiment_results():
    """Load the latest experiment JSON for the results tab."""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        return None
    files = sorted(
        [f for f in os.listdir(exp_dir) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return None
    with open(os.path.join(exp_dir, files[0]), "r") as f:
        return json.load(f)


def find_explanation(user_id: str, recommended_item: str, user_items: dict,
                     item_stats: dict) -> str:
    user_history = user_items.get(user_id, [])
    if not user_history:
        return "Recommended because this item is popular across all users."

    rec_users = set(item_stats.get(recommended_item, {}).get("users", []))
    overlap_items = []
    for entry in user_history:
        iid = entry["item"]
        if entry["rating"] >= 4.0:
            iid_users = set(item_stats.get(iid, {}).get("users", []))
            common = len(rec_users & iid_users)
            if common > 0:
                overlap_items.append((iid, entry["rating"], common))

    overlap_items.sort(key=lambda x: (-x[2], -x[1]))

    if overlap_items:
        top = overlap_items[0]
        return (f"Recommended because you rated **{top[0]}** "
                f"({top[1]:.0f}/5) and {top[2]} users who liked that also liked this item.")

    high_rated = [e for e in user_history if e["rating"] >= 4.0]
    if high_rated:
        return (f"Recommended based on your taste profile — "
                f"you've rated {len(high_rated)} items highly.")

    return "Recommended because this item is popular among similar users."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

TRAIN_PATHS = {
    "Full dataset": "data/train.json",
    "Small dataset": "data_small/train.json",
}
TEST_PATHS = {
    "Full dataset": "data/test.json",
    "Small dataset": "data_small/test.json",
}

available = {k: v for k, v in TRAIN_PATHS.items() if os.path.exists(v)}
if not available:
    st.error("No dataset found. Run `python main.py` first to generate train/test splits.")
    st.stop()

dataset_choice = st.sidebar.selectbox("Dataset", list(available.keys()))
train_path = available[dataset_choice]
test_path = TEST_PATHS[dataset_choice]

MODEL_OPTIONS = [
    "Implicit ALS (best)",
    "WARP (custom PyTorch)",
    "BPR (custom PyTorch)",
    "BPR (implicit library)",
    "Popularity Baseline",
]
model_choice = st.sidebar.selectbox("Recommendation Model", MODEL_OPTIONS)

top_n = st.sidebar.slider("Number of recommendations", 5, 50, 10)
show_explanations = st.sidebar.checkbox("Show explanations", value=True)

st.sidebar.divider()
st.sidebar.caption("Built for CS550 — Massive Data Mining")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train_data, test_data = load_train_test(train_path, test_path)
train_json = json.dumps(train_data)
user_items, item_stats = build_user_item_maps(train_json)

all_users = sorted(user_items.keys())
all_items = sorted(item_stats.keys())

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🎨 Product Recommendation System")
st.caption("Amazon Arts, Crafts & Sewing — Collaborative Filtering Demo")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-box"><h2>{len(all_users):,}</h2><p>Users</p></div>',
                unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><h2>{len(all_items):,}</h2><p>Items</p></div>',
                unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><h2>{len(train_data):,}</h2><p>Train interactions</p></div>',
                unsafe_allow_html=True)
with col4:
    sparsity = 1 - len(train_data) / (len(all_users) * len(all_items))
    st.markdown(f'<div class="metric-box"><h2>{sparsity:.2%}</h2><p>Sparsity</p></div>',
                unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommend for User",
    "📊 Model Comparison",
    "🔥 Popular Items",
    "📈 Dataset Explorer",
])

# ---- Tab 1: User Recommendations ----
with tab1:
    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("Select a user")
        user_id = st.selectbox(
            "User ID",
            all_users,
            index=0,
            help="Pick a user from the training set",
        )

        history = user_items.get(user_id, [])
        st.metric("Items rated", len(history))
        if history:
            avg = np.mean([e["rating"] for e in history])
            st.metric("Avg rating", f"{avg:.2f}")

        st.subheader("Rating history")
        sorted_hist = sorted(history, key=lambda x: -x["rating"])[:15]
        for entry in sorted_hist:
            stars = "⭐" * int(entry["rating"])
            st.text(f"{entry['item']}  {stars}")

    with col_results:
        st.subheader(f"Top-{top_n} Recommendations")
        st.caption(f"Model: **{model_choice}**")

        with st.spinner(f"Running {model_choice}..."):
            model = fit_model(train_path, model_choice)
        exclude = {e["item"] for e in history}
        recs = model.recommend_top_n(user_id, n=top_n, exclude_items=exclude)

        if not recs:
            st.warning("No recommendations available for this user.")
        else:
            for rank, item_id in enumerate(recs, 1):
                stats = item_stats.get(item_id, {})
                avg_r = stats.get("avg_rating", 0)
                n_ratings = stats.get("count", 0)

                st.markdown(f"""
                <div class="rec-card">
                    <h4>#{rank} — {item_id}</h4>
                    <p>Avg rating: {'⭐' * round(avg_r)} ({avg_r:.2f}) &nbsp;|&nbsp; {n_ratings} ratings</p>
                </div>
                """, unsafe_allow_html=True)

                if show_explanations:
                    explanation = find_explanation(user_id, item_id, user_items, item_stats)
                    st.markdown(f'<div class="explain-box">💡 {explanation}</div>',
                                unsafe_allow_html=True)

# ---- Tab 2: Model Comparison ----
with tab2:
    st.subheader("Model Comparison — Full Evaluation Results")

    st.markdown("""
    All models were evaluated on the same train/test split (80/20 per user).
    Rating metrics (MAE, RMSE) measure prediction accuracy.
    Ranking metrics (P@10, R@10, F1, NDCG@10) measure recommendation quality.
    """)

    results_data = {
        "Model": [
            "Popularity Baseline",
            "Matrix Factorization (ALS)",
            "BPR (custom PyTorch)",
            "BPR (implicit library)",
            "Implicit ALS",
            "WARP (custom PyTorch)",
        ],
        "MAE ↓": [0.6234, 0.4632, "—", "—", "—", "—"],
        "RMSE ↓": [0.9340, 0.8193, "—", "—", "—", "—"],
        "P@10 ↑": [0.0023, "—", 0.0019, 0.0007, 0.0188, 0.0148],
        "R@10 ↑": [0.0087, "—", 0.0072, 0.0027, 0.0735, 0.0590],
        "F1 ↑": [0.0036, "—", 0.0030, 0.0011, 0.0300, 0.0237],
        "NDCG@10 ↑": [0.0086, "—", 0.0079, 0.0015, 0.0667, 0.0462],
        "Time": ["1s", "313s", "273s", "8s", "11s", "65s"],
    }
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Key Findings")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("#### Rating Prediction")
        st.markdown("""
        - **MF (ALS)** achieves the best MAE (0.46) and RMSE (0.82)
        - Learns latent user/item factors with biases on GPU
        - Significantly better than the popularity mean baseline
        """)
    with col_f2:
        st.markdown("#### Top-10 Ranking")
        st.markdown("""
        - **Implicit ALS** is the best ranker — P@10: 0.0188, NDCG: 0.0667
        - **WARP** (built from scratch) is second — P@10: 0.0148, NDCG: 0.0462
        - Both are **6-8x better** than the popularity baseline
        - Ranking-optimized objectives massively outperform rating-prediction objectives
        """)

    st.divider()

    st.subheader("Ranking Metrics Comparison")
    ranking_models = ["Popularity", "BPR (custom)", "BPR (implicit)", "WARP (custom)", "Implicit ALS"]
    p10_vals = [0.0023, 0.0019, 0.0007, 0.0148, 0.0188]
    ndcg_vals = [0.0086, 0.0079, 0.0015, 0.0462, 0.0667]

    chart_df = pd.DataFrame({
        "Model": ranking_models * 2,
        "Value": p10_vals + ndcg_vals,
        "Metric": ["Precision@10"] * 5 + ["NDCG@10"] * 5,
    })

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Precision@10**")
        p_df = pd.DataFrame({"Model": ranking_models, "Precision@10": p10_vals}).set_index("Model")
        st.bar_chart(p_df)
    with col_c2:
        st.markdown("**NDCG@10**")
        n_df = pd.DataFrame({"Model": ranking_models, "NDCG@10": ndcg_vals}).set_index("Model")
        st.bar_chart(n_df)

    exp = load_experiment_results()
    if exp:
        st.divider()
        st.caption(f"Results from experiment: `{exp.get('run_id', 'unknown')}`")

# ---- Tab 3: Popular Items ----
with tab3:
    st.subheader("Most Popular Items (by number of ratings)")

    sorted_items = sorted(item_stats.items(), key=lambda x: -x[1]["count"])[:30]
    for rank, (iid, stats) in enumerate(sorted_items, 1):
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            st.write(f"**#{rank}** — `{iid}`")
        with col_b:
            st.write(f"⭐ {stats['avg_rating']:.2f}")
        with col_c:
            st.write(f"📝 {stats['count']} ratings")

# ---- Tab 4: Dataset Explorer ----
with tab4:
    st.subheader("Dataset Statistics")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Rating distribution (train)**")
        ratings = [float(r["overall"]) for r in train_data]
        df = pd.DataFrame({"Rating": ratings})
        st.bar_chart(df["Rating"].value_counts().sort_index())

    with col_b:
        st.write("**Items per user (train)**")
        items_per_user = [len(v) for v in user_items.values()]
        df2 = pd.DataFrame({"Items rated": items_per_user})
        st.bar_chart(df2["Items rated"].value_counts().sort_index().head(50))

    st.divider()
    st.write("**Sample interactions**")
    sample = train_data[:100]
    st.dataframe(
        pd.DataFrame(sample)[["reviewerID", "asin", "overall"]],
        use_container_width=True,
    )
