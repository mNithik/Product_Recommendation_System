# CS550 Presentation Outline

## Slide 1 - Title

**Recommender System Project for Sparse E-Commerce Ratings**  
Amazon Arts, Crafts & Sewing (5-core)

- Nithik Pandya
- Ayush Dodia
- Vraj Shah
- CS 550: Massive Data Mining and Learning

Speaker note:
Start by framing this as a baseline recommender that we upgraded into a cleaner project pipeline rather than a complete rewrite.

## Slide 2 - Problem

**Why this project matters**

- E-commerce catalogs are large and user-item matrices are extremely sparse
- Rating prediction and Top-N recommendation are related but different tasks
- Good recommenders also need to be interpretable and reliable

Speaker note:
Emphasize that a low RMSE model is not automatically the best Top-10 recommender.

## Slide 3 - Dataset

**Amazon Arts, Crafts & Sewing 5-core**

- 56,210 users
- 22,917 items
- 494,485 interactions
- 1-5 rating scale
- About 99.96% sparse
- Per-user 80/20 train-test split

## Slide 4 - Baseline Pipeline

**Original baseline we preserved**

- Rating prediction:
  - Popularity baseline
  - Matrix factorization (ALS-style)
- Ranking / Top-N:
  - Popularity
  - BPR (custom)
  - BPR (implicit library)
  - Implicit ALS
  - WARP (custom)

Speaker note:
Say clearly that we did not replace working baseline models unless there was a strong reason.

## Slide 5 - Main Refactor

**What we changed architecturally**

- Explicit stage separation:
  - preprocessing
  - rating prediction
  - ranking
  - recommendation extraction
  - post-hoc explanation
  - evaluation
- Commoner interfaces and cleaner module boundaries
- Demo aligned with the same staged flow

## Slide 6 - Model-Task Separation

**Correct model use**

- Matrix factorization is used only for rating prediction
- BPR, WARP, and Implicit ALS are used for ranking and Top-N recommendation
- Rating metrics are not used to judge ranking models
- Ranking metrics are not used to judge the rating model

## Slide 7 - Counterfactual Explanations

**Counterfactual explanation layer**

- Applied after ranking as a post-processing layer
- For each recommendation we show:
  - supporting past user interactions
  - similar supporting items
  - support/confidence score
  - weakening condition that could lower the item rank

Speaker note:
Mention that this is practical and approximate, not a full paper reproduction.

## Slide 8 - Optional Causal Layer

**Optional score adjustment reranking**

- Toggleable on or off
- Starts from the base ranking score
- Adds support-based boost
- Applies popularity penalty
- Reorders items without retraining the model

## Slide 9 - Optional Hybrid Branch

**Content-aware hybrid reranking**

- Keeps the collaborative model intact
- Builds TF-IDF text profiles from review text
- Re-ranks candidate pool using collaborative + content signal
- Helps sparse-user and cold-start analysis

## Slide 10 - Evaluation Framework

**Evaluation is now split by task**

- Rating prediction:
  - MAE
  - RMSE
- Ranking / Top-K:
  - Precision@10
  - Recall@10
  - F1
  - NDCG@10
- Explanation:
  - explanation coverage
  - support coverage
  - counterfactual coverage

## Slide 11 - Beyond Accuracy

**Additional evaluation beyond accuracy**

- Fairness audit by activity bucket
- Cold-start vs warm-user benchmark
- Catalog coverage
- Diversity
- Novelty
- Popularity concentration

## Slide 12 - Main Baseline Results

**Stable baseline results from completed runs**

Rating prediction:

- Popularity baseline: MAE 0.6234, RMSE 0.9340
- Matrix factorization: MAE 0.4632, RMSE 0.8193

Top-10 recommendation:

- Popularity baseline: NDCG@10 0.0086
- BPR custom: NDCG@10 0.0079
- BPR implicit: NDCG@10 0.0015
- WARP custom: NDCG@10 0.0462
- Implicit ALS: NDCG@10 0.0667

## Slide 13 - Demo

**What the demo shows**

- user history
- ranked candidates
- final recommendations
- explanation support
- counterfactual weakening
- causal reranking view
- fairness snapshot
- coverage/diversity snapshot

## Slide 14 - Contributions

**What we added to the baseline project**

- Preserved strong baselines
- Clear ranking vs recommendation stages
- Modular explainability layer
- Modular score-adjustment layer
- Fairness and beyond-accuracy evaluation
- Cold-start and hybrid analysis
- Cleaner demo and reporting story

## Slide 15 - Limitations and Future Work

- Some optional models depend on environment-specific packages
- Counterfactual explanations are approximate
- Causal layer is lightweight and post-hoc
- Hybrid branch currently uses TF-IDF rather than richer metadata
- Future work:
  - metadata-aware hybrid models
  - fairness-aware reranking
  - stronger natural-language explanations
  - future-only transformer/sequential experimental branch

## Slide 16 - Closing

**Takeaway**

- The project now goes beyond a baseline recommender
- It is modular, explainable, and easier to evaluate responsibly
- It is suitable for both submission and demo presentation
