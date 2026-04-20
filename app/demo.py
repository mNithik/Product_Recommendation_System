"""
Lightweight demo for the recommender project.

Run:
    streamlit run app/demo.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import ItemSimilarityIndex, explain_recommendation
from src.explainability.counterfactual import build_counterfactual_explanation
from src.pipeline import rank_items_for_user
from src.postprocessing import CausalAdjustmentConfig, apply_causal_adjustment
from src.evaluation import (
    evaluate_beyond_accuracy,
    evaluate_recommendations_per_user,
    run_activity_fairness_audit,
)
from src.hybrid import ContentHybridConfig, ContentHybridRanker
from src.trustworthiness import ReviewTextProfileIndex
from src.utils.data_loader import load_data

st.set_page_config(
    page_title="Personalized Product Recommendation Demo",
    page_icon="RS",
    layout="wide",
)

st.markdown(
    """
<style>
    .card {
        background: linear-gradient(135deg, #0f766e 0%, #164e63 100%);
        color: white;
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
    }
    .card h4 { margin: 0 0 6px 0; }
    .soft-box {
        background: #f5f7fa;
        border-left: 4px solid #0f766e;
        border-radius: 0 10px 10px 0;
        padding: 12px 14px;
        margin: 8px 0;
    }
    .metric-box {
        background: #f7fafc;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-box h3 {
        margin: 0;
        color: #164e63;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _dependency_available(module_name: str) -> tuple[bool, str | None]:
    if importlib.util.find_spec(module_name) is None:
        return False, f"Missing optional dependency: `{module_name}`"
    try:
        __import__(module_name)
    except (ImportError, OSError, RuntimeError) as exc:
        return False, f"`{module_name}` is unavailable here: {exc}"
    return True, None


def available_model_options() -> tuple[list[str], dict[str, str]]:
    options = ["Popularity Baseline"]
    unavailable: dict[str, str] = {}

    implicit_ok, implicit_reason = _dependency_available("implicit")
    torch_ok, torch_reason = _dependency_available("torch")

    if implicit_ok:
        options.extend(["Implicit ALS", "BPR (implicit library)"])
    else:
        unavailable["Implicit ALS"] = implicit_reason or "Missing `implicit`."
        unavailable["BPR (implicit library)"] = implicit_reason or "Missing `implicit`."

    if torch_ok:
        options.extend(["WARP (custom PyTorch)", "BPR (custom PyTorch)"])
    else:
        unavailable["WARP (custom PyTorch)"] = torch_reason or "Missing `torch`."
        unavailable["BPR (custom PyTorch)"] = torch_reason or "Missing `torch`."

    return options, unavailable


@st.cache_data(show_spinner="Loading train/test data...")
def load_train_test(train_path: str, test_path: str):
    return load_data(train_path), load_data(test_path)


@st.cache_data(show_spinner="Building display metadata...")
def build_display_metadata():
    meta_file = Path("AMAZON_FASHION_5.json") / "meta_Arts_Crafts_and_Sewing.json"
    raw_dir = Path("Arts_Crafts_and_Sewing_5.json")
    raw_file = next(raw_dir.glob("*.json"), None) if raw_dir.exists() else None
    user_names: dict[str, str] = {}
    item_titles: dict[str, str] = {}

    if meta_file.exists():
        with meta_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = row.get("asin")
                title = (row.get("title") or "").strip()
                brand = (row.get("brand") or "").strip()
                if item_id and title and item_id not in item_titles:
                    item_titles[item_id] = (
                        f"{title} - {brand}"
                        if brand and brand.lower() not in title.lower()
                        else title
                    )

    if raw_file is not None:
        with raw_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                user_id = row.get("reviewerID")
                reviewer_name = (row.get("reviewerName") or "").strip()
                if user_id and reviewer_name and user_id not in user_names:
                    user_names[user_id] = reviewer_name

                item_id = row.get("asin")
                summary = (row.get("summary") or "").strip()
                if item_id and summary and item_id not in item_titles:
                    item_titles[item_id] = summary[:80]

    return user_names, item_titles


@st.cache_data(show_spinner="Building user/item maps...")
def build_user_item_maps(train_data_json: str):
    train_data = json.loads(train_data_json)
    user_items = defaultdict(list)
    item_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})

    for row in train_data:
        user_id = row["reviewerID"]
        item_id = row["asin"]
        rating = float(row["overall"])
        user_items[user_id].append({"item": item_id, "rating": rating})
        item_stats[item_id]["sum"] += rating
        item_stats[item_id]["count"] += 1

    for item_id, stats in item_stats.items():
        stats["avg_rating"] = stats["sum"] / stats["count"]

    return dict(user_items), dict(item_stats)


@st.cache_resource(show_spinner="Building explanation index...")
def build_explanation_index(train_path: str):
    return ItemSimilarityIndex(load_data(train_path))


@st.cache_resource(show_spinner="Loading model...")
def fit_model(train_path: str, model_type: str):
    train_data = load_data(train_path)

    try:
        if model_type == "Popularity Baseline":
            from src.models import PopularityBaseline

            model = PopularityBaseline()
            model.fit(train_data)
            return model

        if model_type == "Implicit ALS":
            from src.models import ImplicitALSRanker

            model = ImplicitALSRanker(n_factors=64, n_epochs=15, reg=0.0001, pos_threshold=4.0)
            model.fit(train_data)
            return model

        if model_type == "BPR (implicit library)":
            from src.models import ImplicitBPRRanker

            model = ImplicitBPRRanker(n_factors=64, n_epochs=20, lr=0.01, reg=0.01, pos_threshold=4.0)
            model.fit(train_data)
            return model

        if model_type == "WARP (custom PyTorch)":
            from src.models import WARPModel

            model = WARPModel(n_factors=64, n_epochs=20, lr=0.001, reg=0.0001, pos_threshold=4.0)
            model.fit(train_data)
            return model

        if model_type == "BPR (custom PyTorch)":
            from src.models import BPRMatrixFactorization

            model = BPRMatrixFactorization(
                n_factors=64, n_epochs=20, lr=0.001, reg=0.0001, pos_threshold=4.0
            )
            model.fit(train_data)
            return model
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        st.warning(f"{model_type} is unavailable ({exc}). Falling back to Popularity Baseline.")

    from src.models import PopularityBaseline

    model = PopularityBaseline()
    model.fit(train_data)
    return model


@st.cache_resource(show_spinner="Building content profiles...")
def build_text_profile_index(train_path: str):
    return ReviewTextProfileIndex(load_data(train_path), max_features=3500, min_df=3)


@st.cache_data(show_spinner="Running fairness audit...")
def build_fairness_snapshot(train_path: str, test_path: str, model_type: str):
    train_data, test_data = load_train_test(train_path, test_path)
    model = fit_model(train_path, model_type)
    per_user_rows = evaluate_recommendations_per_user(
        model,
        train_data,
        test_data,
        top_n=10,
        batch_size=256,
        min_train_ratings=5,
        max_candidates=100000,
        relevance_threshold=4.0,
        min_item_ratings=0,
        max_users=1200,
    )
    return run_activity_fairness_audit(
        per_user_rows,
        n_buckets=4,
        cold_max_train=5,
        warm_min_train=20,
    )


@st.cache_data(show_spinner="Computing coverage and diversity metrics...")
def build_beyond_accuracy_snapshot(train_path: str, test_path: str, model_type: str):
    train_data, test_data = load_train_test(train_path, test_path)
    model = fit_model(train_path, model_type)
    return evaluate_beyond_accuracy(
        model,
        train_data,
        test_data,
        top_n=10,
        batch_size=256,
        min_train_ratings=5,
        max_candidates=100000,
        relevance_threshold=4.0,
        min_item_ratings=0,
        max_users=1200,
    )


def user_label(user_id: str, user_names: dict[str, str]) -> str:
    return f"{user_names[user_id]} ({user_id})" if user_id in user_names else user_id


def item_label(item_id: str, item_titles: dict[str, str]) -> str:
    return f"{item_titles[item_id]} [{item_id}]" if item_id in item_titles else item_id


train_path = "data/train.json"
test_path = "data/test.json"

if not Path(train_path).exists() or not Path(test_path).exists():
    st.error("Missing `data/train.json` or `data/test.json`. Run `python main.py` first.")
    st.stop()

train_data, test_data = load_train_test(train_path, test_path)
train_json = json.dumps(train_data)
user_items, item_stats = build_user_item_maps(train_json)
user_names, item_titles = build_display_metadata()
explanation_index = build_explanation_index(train_path)

MODEL_OPTIONS, UNAVAILABLE_MODELS = available_model_options()

st.title("Personalized Product Recommendation Demo")
st.caption(
    "Explore how the system ranks products for a user, which items make the final recommendation list, "
    "why those items were chosen, and what could make them fall in the ranking."
)

top_row = st.columns(4)
stats = [
    ("Users", f"{len(user_items):,}"),
    ("Items", f"{len(item_stats):,}"),
    ("Train interactions", f"{len(train_data):,}"),
    ("Test interactions", f"{len(test_data):,}"),
]
for col, (label, value) in zip(top_row, stats):
    with col:
        st.markdown(
            f'<div class="metric-box"><h3>{value}</h3><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

st.sidebar.title("Demo Controls")
model_choice = st.sidebar.selectbox("Recommendation model", MODEL_OPTIONS, index=0)
top_n = st.sidebar.slider("How many recommendations to show", min_value=5, max_value=20, value=10)
enable_causal = st.sidebar.checkbox("Apply causal-style reranking", value=False)
support_weight = st.sidebar.slider("Support weight", 0.0, 0.5, 0.20, 0.05)
popularity_penalty = st.sidebar.slider("Popularity penalty", 0.0, 0.5, 0.10, 0.05)
show_ranked_candidates = st.sidebar.checkbox("Show full ranked list", value=True)
enable_hybrid = st.sidebar.checkbox("Use content-aware hybrid fallback", value=False)
hybrid_alpha = st.sidebar.slider("Hybrid collaborative weight", 0.0, 1.0, 0.70, 0.05)
hybrid_cold_alpha = st.sidebar.slider("Hybrid cold-start weight", 0.0, 1.0, 0.35, 0.05)

if UNAVAILABLE_MODELS:
    st.sidebar.caption("Unavailable backends:")
    for model_name, reason in UNAVAILABLE_MODELS.items():
        st.sidebar.caption(f"- {model_name}: {reason}")

all_users = sorted(user_items.keys())
user_id = st.selectbox(
    "Choose a user profile",
    all_users,
    format_func=lambda uid: user_label(uid, user_names),
    index=0,
)

history = sorted(user_items.get(user_id, []), key=lambda row: -row["rating"])
left, right = st.columns([1, 2])

with left:
    st.subheader("User profile and past activity")
    st.metric("Items rated", len(history))
    if history:
        st.metric("Average rating", f"{np.mean([row['rating'] for row in history]):.2f}")
    for row in history[:15]:
        stars = "*" * int(row["rating"])
        st.text(f"{item_label(row['item'], item_titles)}  {stars}")

with right:
    st.subheader(f"Recommended products for this user")
    active_model_label = model_choice
    if enable_hybrid:
        active_model_label += " + content-aware hybrid"
    st.caption(f"Current model: **{active_model_label}**")

    with st.spinner(f"Running {model_choice}..."):
        fit_start = time.perf_counter()
        model = fit_model(train_path, model_choice)
        if enable_hybrid:
            try:
                text_index = build_text_profile_index(train_path)
                model = ContentHybridRanker(
                    model,
                    train_data,
                    text_index,
                    config=ContentHybridConfig(
                        enabled=True,
                        alpha=hybrid_alpha,
                        cold_start_alpha=hybrid_cold_alpha,
                        pool_size=max(80, top_n * 4),
                        cold_start_threshold=5,
                    ),
                )
            except ValueError as exc:
                st.warning(f"Hybrid fallback unavailable: {exc}")
        fit_time = time.perf_counter() - fit_start

    exclude_items = {row["item"] for row in history}
    rank_start = time.perf_counter()
    ranking_result = rank_items_for_user(
        model,
        user_id=user_id,
        n_candidates=max(top_n + 5, 15),
        exclude_items=exclude_items,
    )
    explanations_by_item = {
        row.item_id: explain_recommendation(explanation_index, user_id, row.item_id)
        for row in ranking_result.items
    }
    active_ranking = apply_causal_adjustment(
        ranking_result,
        explanations_by_item,
        item_popularity={item_id: stats["count"] for item_id, stats in item_stats.items()},
        config=CausalAdjustmentConfig(
            enabled=enable_causal,
            support_weight=support_weight,
            popularity_penalty_weight=popularity_penalty,
        ),
    )
    rank_time = time.perf_counter() - rank_start

    m1, m2, m3 = st.columns(3)
    m1.metric("Model fit/load", f"{fit_time:.2f}s")
    m2.metric("Ranking + explanation", f"{rank_time:.2f}s")
    m3.metric("Recommendations shown", str(top_n))

    if show_ranked_candidates:
        st.markdown("#### Ranked candidates before the final display list")
        candidate_rows = []
        for row in active_ranking.items[: max(top_n + 3, 10)]:
            candidate_rows.append(
                {
                    "Rank": row.rank,
                    "Item": item_label(row.item_id, item_titles),
                    "Score": row.score,
                    "Base rank": row.metadata.get("base_rank", row.rank),
                    "Raw score": row.metadata.get("raw_model_score", row.score),
                }
            )
        st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)

    recommended_items = active_ranking.items[:top_n]
    if not recommended_items:
        st.warning("No recommendations available for this user.")
    else:
        st.markdown("#### Final recommendation list")
        for row in recommended_items:
            item_id = row.item_id
            item_info = item_stats.get(item_id, {})
            avg_rating = float(item_info.get("avg_rating", 0.0))
            count = int(item_info.get("count", 0))
            explanation = explanations_by_item[item_id]
            counterfactual = build_counterfactual_explanation(explanation, active_ranking)
            score_text = f"{row.score:.4f}" if row.score is not None else "Unavailable"

            st.markdown(
                f"""
                <div class="card">
                    <h4>#{row.rank} - {item_label(item_id, item_titles)}</h4>
                    <p>Average rating: {avg_rating:.2f} | Ratings: {count} | Score: {score_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="soft-box"><b>Why this product appeared:</b> {explanation.explanation_text}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="soft-box"><b>Explanation confidence:</b> {explanation.support_confidence:.3f}'
                f' | <b>Reason type:</b> {explanation.explanation_type}</div>',
                unsafe_allow_html=True,
            )
            if explanation.supporting_history:
                support_text = ", ".join(
                    f"{item_label(s.item_id, item_titles)} (overlap={s.user_overlap}, sim={s.similarity:.2f})"
                    for s in explanation.supporting_history
                )
                st.markdown(
                    f'<div class="soft-box"><b>Past interactions that support this choice:</b> {support_text}</div>',
                    unsafe_allow_html=True,
                )
            if explanation.supporting_similar_items:
                similar_text = ", ".join(
                    f"{item_label(s.item_id, item_titles)} (sim={s.similarity:.2f})"
                    for s in explanation.supporting_similar_items
                )
                st.markdown(
                    f'<div class="soft-box"><b>Similar products that reinforce it:</b> {similar_text}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="soft-box"><b>What could make it drop:</b> '
                f'{counterfactual.minimal_change_text}</div>',
                unsafe_allow_html=True,
            )

st.divider()
st.subheader("Fairness snapshot")
st.caption(
    "This audit checks whether recommendation quality changes across users with short versus rich histories."
)
fairness = build_fairness_snapshot(train_path, test_path, model_choice)
fairness_disparity = fairness["disparity"]
fairness_cold_start = fairness["cold_start"]
f1, f2, f3 = st.columns(3)
f1.metric(
    "Activity-bucket NDCG disparity",
    (
        f"{fairness_disparity['ndcg']:.2f}x"
        if fairness_disparity["ndcg"] == fairness_disparity["ndcg"] else "Unavailable"
    ),
)
f2.metric("Cold-user mean NDCG", f"{fairness_cold_start['cold_mean_ndcg']:.4f}")
f3.metric("Warm-user mean NDCG", f"{fairness_cold_start['warm_mean_ndcg']:.4f}")
st.caption(
    "Cold users have 5 or fewer training interactions. Warm users have 20 or more. "
    "Larger gaps suggest the model favors users with richer histories."
)
st.dataframe(fairness["summary"], use_container_width=True, hide_index=True)

st.divider()
st.subheader("Coverage and diversity snapshot")
st.caption(
    "These metrics show whether the model spreads attention across the catalog or keeps recommending the same popular items."
)
beyond_accuracy = build_beyond_accuracy_snapshot(train_path, test_path, model_choice)
b1, b2, b3, b4 = st.columns(4)
b1.metric("Catalog coverage", f"{beyond_accuracy['CatalogCoverage']:.4f}")
b2.metric("Diversity", f"{beyond_accuracy['Diversity']:.4f}")
b3.metric("Novelty", f"{beyond_accuracy['Novelty']:.4f}")
b4.metric("Popularity concentration", f"{beyond_accuracy['PopularityConcentration']:.4f}")
st.caption(
    "Higher diversity and novelty usually mean less repetitive recommendation lists. "
    "Higher popularity concentration means the model leans more heavily toward already-popular products."
)

st.divider()
st.subheader("How to read this page")
st.markdown(
    """
- The model first builds a ranked list of candidate products for the selected user.
- The final recommendation list is taken from the top of that ranking.
- If the content-aware hybrid option is enabled, the system reorders the collaborative candidate pool using review-text similarity.
- Explanations are added after ranking so you can see why an item was surfaced.
- The counterfactual note shows a simple condition that could lower an item's position.
- Causal-style reranking is optional and only changes the final ordering, not the trained model.
- The fairness snapshot compares recommendation quality across low-activity and high-activity users.
- The coverage and diversity snapshot shows whether the model explores the catalog or stays close to popular items.
"""
)
