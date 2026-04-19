"""
Streamlit demo for the Product Recommendation System.

Run:
    streamlit run app/demo.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.late_fusion import evaluate_late_fusion_recommendations, rank_items_late_fusion
from src.trustworthiness.text_profiles import ReviewTextProfileIndex
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


@st.cache_data(show_spinner="Indexing review text (TF-IDF)…")
def build_review_text_index(train_path: str):
    """TF-IDF index over optional star rating, verified, style, summary, and reviewText; None if too little text."""
    train = load_data(train_path)
    try:
        return ReviewTextProfileIndex(train, max_features=3500, min_df=3)
    except ValueError:
        return None


@st.cache_resource(show_spinner="Building Sentence-BERT index (first run can take a few minutes)…")
def build_sentence_embedding_index(train_path: str):
    """MiniLM embeddings over aggregated metadata + summary + reviewText per item; GPU if available."""
    from src.trustworthiness.sentence_embeddings import SentenceReviewProfileIndex

    train = load_data(train_path)
    return SentenceReviewProfileIndex(train)


@st.cache_data(show_spinner="Fairness audit (per-user metrics)…")
def run_user_activity_fairness_audit(
    train_path: str,
    test_path: str,
    model_type: str,
    max_users: int,
    relevance_threshold: float,
    n_buckets: int = 4,
):
    """Per-user Top-10 metrics, then bucket aggregates by training-set activity."""
    from src.evaluation.fairness import disparity_ratios_by_metric, summarize_ndcg_by_train_activity
    from src.evaluation.metrics import evaluate_recommendations_per_user

    train, test = load_train_test(train_path, test_path)
    model = fit_model(train_path, model_type)
    rows = evaluate_recommendations_per_user(
        model,
        train,
        test,
        top_n=10,
        batch_size=256,
        min_train_ratings=5,
        max_candidates=100000,
        relevance_threshold=relevance_threshold,
        min_item_ratings=0,
        max_users=max_users,
    )
    summary = summarize_ndcg_by_train_activity(rows, n_buckets=n_buckets)
    ratios = disparity_ratios_by_metric(rows, n_buckets=n_buckets)
    return rows, summary, ratios


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
text_sim_mode = st.sidebar.radio(
    "Review-text similarity (optional)",
    ("Off", "TF-IDF", "Sentence-BERT (MiniLM)"),
    index=0,
    help=(
        "Does not change recommendations. The selected recommender uses ratings only; "
        "TF-IDF / Sentence-BERT add post-hoc context from each review's star rating, verified flag, "
        "product style fields (when present), summary title, and review body (explainability)."
    ),
)

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommend for User",
    "📊 Model Comparison",
    "🔥 Popular Items",
    "📈 Dataset Explorer",
    "🛡️ Fairness audit",
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
            fit_start = time.perf_counter()
            model = fit_model(train_path, model_choice)
            fit_elapsed = time.perf_counter() - fit_start
        exclude = {e["item"] for e in history}
        rec_start = time.perf_counter()
        recs = model.recommend_top_n(user_id, n=top_n, exclude_items=exclude)
        rec_elapsed = time.perf_counter() - rec_start
        total_elapsed = fit_elapsed + rec_elapsed

        t1, t2, t3 = st.columns(3)
        t1.metric("Model fit/load time", f"{fit_elapsed:.2f}s")
        t2.metric("Recommendation time", f"{rec_elapsed:.2f}s")
        t3.metric("Total time", f"{total_elapsed:.2f}s")

        if text_sim_mode != "Off":
            st.info(
                "**Recommendations** come only from **"
                + model_choice
                + "** (rating / implicit-feedback patterns). "
                "**Text similarity** (TF-IDF or Sentence-BERT) is **post-hoc** "
                "— it scores overlap between the user's past reviews and each item's reviews "
                "(stars, verified, style, summary, review body when present). The ranked cards above stay **pure CF**; "
                "an optional **late fusion** table below blends CF pool order with text similarity for comparison only."
            )

        text_idx = None
        sent_idx = None
        if text_sim_mode == "TF-IDF":
            text_idx = build_review_text_index(train_path)
            if text_idx is None:
                st.caption("Not enough text (metadata, summary, and/or review body) in this split to build a TF-IDF index.")
        elif text_sim_mode == "Sentence-BERT (MiniLM)":
            try:
                sent_idx = build_sentence_embedding_index(train_path)
            except ImportError as e:
                st.warning(str(e))
            except Exception as e:
                st.warning(f"Sentence-BERT index failed: {e}")

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
                    if text_idx is not None:
                        sim = text_idx.cosine_user_item(train_data, user_id, item_id)
                        st.markdown(
                            f'<div class="explain-box">📎 <b>TF-IDF similarity:</b> {sim:.3f}</div>',
                            unsafe_allow_html=True,
                        )
                    if sent_idx is not None:
                        sim = sent_idx.cosine_user_item(train_data, user_id, item_id)
                        st.markdown(
                            f'<div class="explain-box">📎 <b>Sentence-BERT similarity:</b> {sim:.3f}</div>',
                            unsafe_allow_html=True,
                        )

            text_index_fusion = sent_idx if sent_idx is not None else text_idx
            if text_index_fusion is not None:
                st.subheader("Late fusion ranking (separate from CF list above)")
                st.caption(
                    "Builds a larger candidate pool from the **same model**, then re-sorts by "
                    "α×(collaborative position in pool) + (1−α)×(text similarity). "
                    "Does not change training or the primary Top-N cards."
                )
                fc1, fc2 = st.columns(2)
                with fc1:
                    fusion_alpha = st.slider(
                        "α (CF weight)",
                        0.0,
                        1.0,
                        0.65,
                        0.05,
                        key="fusion_alpha",
                        help="1 = keep model pool order; 0 = sort by text only (within pool).",
                    )
                with fc2:
                    fusion_pool = st.slider(
                        "Candidate pool size",
                        max(top_n, 15),
                        min(200, max(50, top_n * 5)),
                        min(100, max(top_n * 4, 40)),
                        5,
                        key="fusion_pool",
                        help="Number of CF-ranked items to pull before re-ranking with text.",
                    )
                pool_n = max(int(fusion_pool), int(top_n))
                pool = model.recommend_top_n(user_id, n=pool_n, exclude_items=exclude)
                fused_rows = rank_items_late_fusion(
                    pool,
                    train_data,
                    user_id,
                    text_index_fusion.cosine_user_item,
                    fusion_alpha,
                )[:top_n]
                pool_rank = {asin: i + 1 for i, asin in enumerate(pool)}
                df_f = pd.DataFrame(
                    [
                        {
                            "Fusion rank": i + 1,
                            "Item": asin,
                            "CF rank in pool": pool_rank.get(asin, "—"),
                            "CF norm": round(cf, 3),
                            "Text sim": round(tx, 3),
                            "Fused": round(fu, 3),
                        }
                        for i, (asin, fu, cf, tx) in enumerate(fused_rows)
                    ]
                )
                st.dataframe(df_f, use_container_width=True, hide_index=True)

                with st.expander("Offline evaluation — late fusion vs. held-out test (sample)"):
                    st.markdown(
                        "Runs the **same** Precision/Recall/NDCG@N protocol as the main pipeline, "
                        "but recommendations are built with late fusion. Uses the **test split** "
                        "for this dataset choice; can take a minute."
                    )
                    ev_max_u = st.slider("Max users to evaluate", 50, 2500, 400, 50, key="fusion_ev_users")
                    if st.button("Run late fusion evaluation", key="fusion_ev_run"):
                        with st.spinner("Evaluating late fusion…"):
                            m = evaluate_late_fusion_recommendations(
                                model,
                                train_data,
                                test_data,
                                text_index_fusion,
                                alpha=float(fusion_alpha),
                                top_n=int(top_n),
                                pool_size=int(pool_n),
                                min_train_ratings=5,
                                relevance_threshold=4.0,
                                max_users=int(ev_max_u),
                            )
                        st.metric("Users evaluated", m["n_users_eval"])
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("P@N", f"{m['Precision']:.4f}")
                        c2.metric("R@N", f"{m['Recall']:.4f}")
                        c3.metric("F1", f"{m['F-measure']:.4f}")
                        c4.metric("NDCG@N", f"{m['NDCG']:.4f}")
                        st.caption(
                            f"Settings: α={m['alpha']}, pool={m['pool_size']}, N={top_n}. "
                            "Compare to CF-only metrics from `python main.py` for the same model."
                        )

            if hasattr(model, "recommend_top_n_profile_ablation"):
                with st.expander("Counterfactual — remove items from your profile (Implicit ALS)"):
                    st.markdown(
                        "Same trained model; selected items are **removed from the user vector** "
                        "passed to scoring at recommendation time (no refit)."
                    )
                    hist_items = [e["item"] for e in history]
                    drop_cf = st.multiselect(
                        "Items to remove from profile",
                        hist_items,
                        max_selections=6,
                        key="cf_drop_items",
                    )
                    if st.button("Recompute Top-N with ablated profile", key="cf_run"):
                        if not drop_cf:
                            st.info("Select at least one rated item to remove.")
                        else:
                            rec_cf = model.recommend_top_n_profile_ablation(
                                user_id, n=top_n, drop_asins=set(drop_cf)
                            )
                            st.write("**Ablated Top-N:**", ", ".join(rec_cf) if rec_cf else "(empty)")
                            lost = [x for x in recs if x not in rec_cf]
                            gained = [x for x in rec_cf if x not in recs]
                            if lost or gained:
                                st.caption(f"Removed from list: {lost}  ·  New in list: {gained}")
                            else:
                                st.caption("List unchanged for this ablation (ties or robust ranking).")

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

# ---- Tab 5: Fairness audit (NDCG by training-activity buckets) ----
with tab5:
    st.subheader("Fairness audit")
    st.caption(
        "Per-user Top-10 metrics, then aggregates by training-set activity (quantile buckets). "
        "Disparity ratios compare the best vs. worst bucket **mean** (positive values only) for each metric."
    )

    st.markdown("#### Settings")
    fair_model = st.selectbox(
        "Model for fairness audit",
        MODEL_OPTIONS,
        index=0,
        help="Uses the same Top-10 protocol as the main pipeline (slower for large max users).",
        key="fair_model",
    )
    max_users_f = st.slider("Max users to evaluate (fairness audit)", 200, 8000, 1500, step=100, key="fair_max_u")
    rel_thr = st.slider("Relevance threshold (test items ≥ this rating)", 3.0, 5.0, 4.0, step=0.5, key="fair_rel")
    fair_buckets = st.slider(
        "Number of activity buckets (quantiles)",
        2,
        12,
        4,
        1,
        key="fair_n_buckets",
        help="More buckets = finer train-count slices (qcut; fewer if duplicates drop).",
    )

    if st.button("Run fairness audit", key="fair_run"):
        with st.spinner("Computing per-user metrics…"):
            rows, summary, ratios = run_user_activity_fairness_audit(
                train_path, test_path, fair_model, max_users_f, rel_thr, n_buckets=int(fair_buckets)
            )
        st.metric("Users evaluated", len(rows))
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "NDCG disparity (max/min bucket mean)",
            f"{ratios['ndcg']:.2f}x" if ratios["ndcg"] == ratios["ndcg"] else "—",
        )
        c2.metric(
            "Precision disparity",
            f"{ratios['precision']:.2f}x" if ratios["precision"] == ratios["precision"] else "—",
        )
        c3.metric(
            "Recall disparity",
            f"{ratios['recall']:.2f}x" if ratios["recall"] == ratios["recall"] else "—",
        )

        st.markdown("##### Bucket summary (mean ± dispersion)")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        csv_summary = summary.to_csv(index=False).encode("utf-8")
        csv_users = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download bucket summary (CSV)",
                data=csv_summary,
                file_name="fairness_activity_buckets.csv",
                mime="text/csv",
                key="fair_dl_summary",
            )
        with d2:
            st.download_button(
                "Download per-user metrics (CSV)",
                data=csv_users,
                file_name="fairness_per_user_metrics.csv",
                mime="text/csv",
                key="fair_dl_users",
            )

        plot_df = summary.copy()
        plot_df["activity_bucket"] = plot_df["bucket"].astype(str)
        st.markdown("##### Mean NDCG by bucket")
        st.bar_chart(plot_df.set_index("activity_bucket")["mean_ndcg"])
        st.markdown("##### Mean Precision by bucket")
        st.bar_chart(plot_df.set_index("activity_bucket")["mean_precision"])
        st.caption(
            "Bucket table includes **std** and **median** per metric. "
            "High disparity ratios flag uneven quality across activity groups (audit only; no re-ranking)."
        )
