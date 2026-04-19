# Architecture Audit And Refactor Plan

This document records the current project architecture before any research-grade
upgrade work. The goal is to preserve the working baseline, make the current
ownership explicit, and define the minimum-safe refactor path.

## 1. Current Project Position

The repository already has a solid baseline recommender project with:

- explicit rating prediction evaluation
- Top-N recommendation evaluation
- multiple baseline and ranking models
- a Streamlit demo
- tests for preprocessing, metrics, popularity, late fusion, fairness, and
  text-profile utilities

The code is not yet fully organized as a research-grade pipeline with explicit
ranking, recommendation, explanation, and counterfactual stages. However, many
useful pieces already exist and should be preserved rather than replaced.

## 2. Current Task Separation

The project already separates tasks at the experiment level in `main.py`.

- Rating prediction:
  - `PopularityBaseline`
  - `MatrixFactorizationGPU`
  - evaluated with `evaluate_rating_prediction()`
- Top-N ranking / recommendation:
  - `PopularityBaseline`
  - `BPRMatrixFactorization`
  - `ImplicitBPRRanker`
  - `ImplicitALSRanker`
  - `WARPModel`
  - evaluated with `evaluate_recommendations()`

This is a good starting point and matches the intended research rules:

- do not use rating metrics to judge ranking models
- do not use ranking metrics to judge the rating-prediction model

## 3. File And Function Ownership

### 3.1 Entry Points

- `main.py`
  - orchestrates preprocessing, training, evaluation, and experiment logging
  - `_run_rating_eval()`
  - `_run_ranking_eval()`
  - `main()`
- `app/demo.py`
  - Streamlit demo entry point
  - model loading, recommendation display, simple explanation display,
    late-fusion comparison, and fairness audit UI

### 3.2 Configuration And Tracking

- `configs/default.yaml`
  - main experiment settings
- `configs/small.yaml`
  - smaller/faster experiment variant
- `src/utils/config_loader.py`
  - `load_config()`
  - `parse_args()`
- `src/utils/experiment_tracker.py`
  - `ExperimentTracker`

### 3.3 Data Preprocessing And Loading

- `src/preprocessing/preprocess.py`
  - `load_reviews()`
  - `split_per_user()`
  - `run_preprocessing()`
- `src/preprocessing/make_small_dataset.py`
  - reduced dataset generation for faster experiments
- `src/utils/data_loader.py`
  - `load_data()`
  - `build_index()`

Current behavior:

- raw Amazon JSONL reviews are filtered and normalized
- train/test split is random per user
- records retain rating plus optional text fields for later analysis

### 3.4 Models

- `src/models/popularity.py`
  - `PopularityBaseline`
  - ownership:
    - item mean rating prediction
    - popularity ranking fallback
    - cold-start-friendly baseline
- `src/models/matrix_factorization.py`
  - `MatrixFactorizationGPU`
  - ownership:
    - ALS-style explicit rating model
    - `predict()` and `predict_batch()` for rating prediction
    - currently also contains ranking helpers, but those should not define the
      default ranking pipeline going forward
- `src/models/bpr.py`
  - `BPRMatrixFactorization`
  - ownership:
    - custom BPR training
    - candidate generation via item co-occurrence
    - direct Top-N recommendation
- `src/models/implicit_bpr.py`
  - `ImplicitBPRRanker`
  - ownership:
    - implicit-library BPR ranking
    - direct Top-N recommendation
- `src/models/implicit_als.py`
  - `ImplicitALSRanker`
  - ownership:
    - implicit-library ALS ranking
    - direct Top-N recommendation
    - profile ablation for counterfactual-style UI analysis
- `src/models/warp.py`
  - `WARPModel`
  - ownership:
    - custom WARP training
    - direct Top-N recommendation
- `src/models/item_cf.py`
  - `ItemBasedCF`
  - ownership:
    - item-based CF implementation
    - available but not a mainline baseline in `main.py`

### 3.5 Evaluation

- `src/evaluation/metrics.py`
  - `mae()`
  - `rmse()`
  - `precision_at_k()`
  - `recall_at_k()`
  - `f_measure()`
  - `dcg_at_k()`
  - `ndcg_at_k()`
  - `evaluate_rating_prediction()`
  - `evaluate_recommendations()`
  - `evaluate_recommendations_per_user()`
  - `compute_candidate_hit_rate()`
- `src/evaluation/late_fusion.py`
  - post-hoc reranking experiment using collaborative order and text similarity
- `src/evaluation/fairness.py`
  - activity-bucket fairness-style audit summaries

### 3.6 Explainability / Trustworthiness-Like Utilities

- `src/trustworthiness/text_profiles.py`
  - TF-IDF item profile construction
  - user-item text similarity scoring
- `src/trustworthiness/sentence_embeddings.py`
  - optional sentence-transformer user-item text similarity scoring
- `app/demo.py`
  - `find_explanation()`
  - simple heuristic explanation for displayed recommendations

These are useful foundations, but they are not yet a dedicated explanation
module with a reusable API.

### 3.7 Tests

- `tests/test_preprocessing.py`
- `tests/test_metrics.py`
- `tests/test_popularity.py`
- `tests/test_recommendation.py`
- `tests/test_late_fusion.py`
- `tests/test_fairness.py`
- `tests/test_review_snippet_and_profiles.py`
- `tests/test_sentence_embeddings.py`

Current tests provide decent baseline coverage for utility behavior, but there
are no tests yet for explicit ranking-stage vs recommendation-stage separation,
no explanation schema checks, and no causal toggle tests.

## 4. Current Pipeline Behavior

### 4.1 Offline Experiment Flow

Current mainline flow in `main.py`:

1. load config
2. run preprocessing if train/test split does not already exist
3. load train/test JSONL data
4. train each model
5. evaluate:
   - rating models on MAE/RMSE
   - ranking models on Precision/Recall/F1/NDCG
6. save experiment results JSON

### 4.2 Online Demo Flow

Current demo flow in `app/demo.py`:

1. load cached train/test data
2. fit selected recommendation model
3. call `model.recommend_top_n(...)`
4. render recommendation cards
5. optionally show:
   - heuristic explanation text
   - TF-IDF similarity
   - Sentence-BERT similarity
   - late-fusion comparison
   - fairness audit
   - Implicit ALS profile ablation

## 5. What Is Already Good Enough To Preserve

The following should be preserved with minimal changes:

- current preprocessing and config flow
- current baseline model implementations
- task-specific evaluation distinction in `main.py`
- experiment tracking
- working Streamlit demo
- trustworthiness/text-similarity utilities
- fairness and late-fusion as optional side experiments

These are assets, not clutter.

## 6. Current Architectural Gaps

### 6.1 Ranking And Recommendation Are Not Explicit Pipeline Stages

The largest structural gap is that most ranking models go directly from model
state to `recommend_top_n(...)`.

What is missing:

- explicit candidate ranking output
- reusable ranked list object
- separate recommendation stage that truncates or post-processes ranking output

This makes explanation, causal adjustment, and evaluation extensions harder than
they need to be.

### 6.2 No Common Ranking Interface

The ranking logic is model-specific:

- MF uses `_score()`, `_rank_scores()`, and `_get_candidates()`
- BPR scores inside `recommend_top_n()`
- WARP scores full item vectors inside `recommend_top_n()`
- implicit models hide ranking inside library calls

That is workable for a class project baseline but not ideal for a research-grade
extension pipeline.

### 6.3 Explanation Logic Lives In The Demo

`find_explanation()` in `app/demo.py` is useful, but it is:

- UI-coupled
- string-based
- not reusable offline
- not testable as a structured explanation component

### 6.4 Counterfactual Logic Is Only Partial

The repo already has one useful seed:

- `ImplicitALSRanker.recommend_top_n_profile_ablation()`

This is promising, but it is:

- model-specific
- exposed only through the demo
- not yet a general post-hoc counterfactual explanation layer

### 6.5 No Causal Adjustment Layer

There is no configurable score adjustment or reweighting stage yet.

### 6.6 Evaluation Is Task-Separated But Not Yet Module-Separated By Research Concern

Current evaluation is already split between rating and recommendation metrics,
which is good. However, future work will benefit from explicit modules for:

- rating prediction evaluation
- ranking / Top-K recommendation evaluation
- explanation evaluation

## 7. Safe Constraints For The Upgrade

The refactor should follow these rules:

- preserve current baseline behavior unless intentionally extending it
- keep ranking and recommendation as separate stages
- keep `MatrixFactorizationGPU` as an explicit-rating model
- keep popularity as the fallback and cold-start baseline
- do not replace working ranking baselines
- make explainability post-hoc and modular
- make causal adjustment optional and toggleable
- keep any transformer work fully separate from the default pipeline

## 8. Minimum-Safe Refactor Plan

This is the recommended order of implementation.

### Phase 1: Architecture Lock And Documentation

Goal:

- make current ownership explicit
- define the safe refactor scope

Deliverables:

- this document
- file/function ownership map
- minimum-safe upgrade sequence

### Phase 2: Explicit Ranking And Recommendation Pipeline

Goal:

- separate "rank candidates" from "return Top-N recommendations"

Recommended additions:

- `src/pipeline/ranking.py`
- `src/pipeline/recommendation.py`

Recommended behavior:

- ranking models produce ranked candidate rows with scores
- recommendation stage takes ranked rows and outputs final Top-N items

Important note:

- baseline model classes should remain intact
- adapters/wrappers are preferred over rewriting model internals

### Phase 3: Common Ranking Interface

Goal:

- make ranking outputs consistent across BPR, WARP, implicit ALS, implicit BPR,
  and popularity

Recommended additions:

- a small typed structure for ranked candidates
- one adapter per ranking model if needed

Example responsibilities:

- `score_candidates(user_id, candidate_items)`
- `rank_for_user(user_id, exclude_items, max_candidates)`

### Phase 4: Modular Explainability

Goal:

- move explanation logic out of the Streamlit app

Recommended additions:

- `src/explainability/item_similarity.py`
- `src/explainability/explanation_engine.py`

Required explanation outputs per recommendation:

- supporting past user interactions
- supporting similar items
- support or confidence score
- explanation type or source

### Phase 5: Counterfactual Explainability

Goal:

- add a CountER-inspired post-hoc layer without replacing the recommender

Recommended additions:

- `src/explainability/counterfactual.py`

Expected behavior:

- for each recommended item, estimate a minimal weakening condition
- approximate which historical support items, if removed or weakened, would
  likely reduce score or rank

This should stay practical and approximate, not a full paper reproduction.

### Phase 6: Optional Causal Adjustment Layer

Goal:

- add a simple score-adjustment or reweighting layer

Recommended additions:

- `src/postprocessing/causal_adjustment.py`

Expected behavior:

- optional on/off toggle
- small, interpretable score adjustment
- no heavy causal graph machinery

### Phase 7: Evaluation Refactor

Goal:

- keep task metrics separated and add explanation metrics

Recommended additions:

- `src/evaluation/rating.py`
- `src/evaluation/recommendation.py`
- `src/evaluation/explanation.py`

The current `metrics.py` can either be gradually split or remain as the shared
metric implementation file while new orchestration wrappers are added.

### Phase 8: Demo Refactor

Goal:

- surface pipeline stages clearly in the UI

Desired UI sections:

- ranked candidate scores
- final Top-N recommendations
- explanation cards
- support items
- counterfactual weakening information
- causal adjustment toggle state

### Phase 9: Tests

Goal:

- enforce the new architecture invariants

Recommended new tests:

- ranking models are not evaluated with rating metrics
- rating models are not evaluated with Top-N metrics
- recommendation output is derived from ranking output
- explanations exist for recommended items
- counterfactual outputs follow the expected schema
- causal adjustment toggles on/off cleanly

## 9. Specific Risks To Avoid

- Do not rewrite all models into a shared inheritance hierarchy unless the
  benefits clearly outweigh the migration cost.
- Do not move explanation logic into training code.
- Do not add a transformer branch into the default pipeline.
- Do not couple explanation generation to Streamlit-only rendering code.
- Do not judge MF with ranking-only metrics as the main rating result.

## 10. Recommended Immediate Next Step

The next coding step should be Phase 2:

- introduce explicit ranking and recommendation pipeline modules
- keep existing model classes untouched as much as possible
- route current ranking models through adapters or thin wrappers
- make the recommendation stage explicitly consume ranking output

This is the highest-leverage change because it creates the foundation needed for:

- explanation modules
- counterfactual post-processing
- causal adjustment
- cleaner evaluation
- a clearer research report story

## 11. Parts That Can Be Treated As Future Work

These should not block the core refactor:

- a PETER-style transformer module
- heavy neural explainability architectures
- full causal collaborative filtering reproduction
- production serving abstractions

They can be documented as future extensions after the baseline + explanation
pipeline is clean.
