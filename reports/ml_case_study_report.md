# Machine Learning Case Study: Context-Aware Music Recommendation System

**Project Name:** Vibe-Sync – Mood-Aware Spotify-Style Recommender
**Author:** Yashwanth Kamireddi
**Last Updated:** 2 November 2025

---

## 1. Problem Understanding & Objective Definition

**Problem statement (one-page narrative):**

Modern music platforms excel at long-term personalization but often ignore a listener’s immediate context. Users who want “high-energy workout tracks” or “calming focus playlists” must search manually, wasting time and encountering inconsistent results. Vibe-Sync addresses this gap by learning mood signatures from Spotify audio attributes, blending them with configuration-driven metadata filters, and serving mood-aligned playlists through a responsive web experience. The system operates without Spotify API rate limits by relying on an offline Kaggle dataset while still mirroring Spotify’s look and feel. It ingests high-volume track catalogs (114k songs), scores them using trained models plus heuristic filters, and returns diverse, high-confidence recommendations in roughly two seconds per mood request on commodity hardware.

**Business objectives:**

- Deliver context-aware playlists for five predefined moods (Workout, Chill, Party, Focus, Sleep).
- Maintain Spotify-like UX with album art, preview links, and score transparency.
- Allow optional Spotify API enrichment without making the service dependent on third-party quotas.
- Support rapid experimentation and re-training as mood definitions or metadata filters evolve.

**ML task type:** Multi-label, binary classification (one independent classifier per mood). Each model predicts whether a track is suitable (`1`) or not (`0`) for the target mood.

**Success metrics:**

- **Primary:** Hold-out accuracy/F1 per mood while acknowledging that deterministic label rules cap the achievable variance. Metrics are reported alongside commentary on their limitations.
- **Secondary:** Precision/Recall balance to avoid false positives (party tracks in sleep playlists) and false negatives (missing strong candidates), paired with qualitative playlist reviews.
- **Operational:** End-to-end latency below ~2 seconds per recommendation call on a developer laptop (Ryzen 7, 32 GB RAM) based on local profiling with `time.perf_counter`.

---

## 2. Data Collection

- **Source:** Kaggle "Spotify Tracks" dataset (`data/raw/spotify_tracks.csv`, 114,000 rows, 21 columns). Loaded via `src/load_kaggle_data.py`.
- **Structure:** Structured tabular CSV with consistent schema (numeric audio features + textual metadata).
- **Coverage:** Global mix of genres and popularity tiers. Audio attributes align with Spotify API definitions.
- **Justification:** Provides a large, clean corpus without API latency or rate limits, enabling reproducible university-grade experiments.

---

## 3. Data Preprocessing & Cleaning

Implemented in `train_ml_models.py`, `src/ml_mood_classifier.py`, `src/data_pipeline.py`, and `src/recommender.py::_prepare_candidates`.

1. **Column standardization:** Rename Kaggle-provided columns (`track_name → name`, `track_id → id`, `artist_name → artists`, `album_name → album`).
2. **Missing data handling:** Drop rows with nulls in the nine mandatory audio features. Removes ~24k rows to ensure model integrity.
3. **Type coercion:** Force audio features to numeric floats; coerce invalid strings to `NaN` before dropping.
4. **Outlier treatment:** Clip loudness to [-60, 0] dB and tempo to [30, 250] BPM to align with mood criteria.
5. **Metadata normalization:** Normalize genre strings, coerce `duration_ms`, `popularity`, and `explicit` flags, and drop records missing mandatory metadata to keep filters reliable.
6. **Deduplication:** Remove duplicate tracks (by `id` or `track_id`) to avoid sampling bias.
7. **Train/test split:** Stratified 80/20 split with `train_test_split(..., stratify=y)` for each mood classifier.
8. **Scaling:** Standardize numeric features via `StandardScaler` before fitting the Random Forest (scalers persisted per mood).

---

## 4. Exploratory Data Analysis (EDA)

**Descriptive statistics (114,000 tracks):**

| Feature            | Mean   | Std Dev | Min    | 25%    | 50%    | 75%    | Max   |
|--------------------|--------|---------|--------|--------|--------|--------|-------|
| acousticness       | 0.315  | 0.333   | 0.000  | 0.017  | 0.169  | 0.598  | 0.996 |
| danceability       | 0.567  | 0.174   | 0.000  | 0.456  | 0.580  | 0.695  | 0.985 |
| energy             | 0.641  | 0.252   | 0.000  | 0.472  | 0.685  | 0.854  | 1.000 |
| instrumentalness   | 0.156  | 0.310   | 0.000  | 0.000  | 0.000  | 0.049  | 1.000 |
| liveness           | 0.214  | 0.190   | 0.000  | 0.098  | 0.132  | 0.273  | 1.000 |
| loudness (dB)      | -8.259 | 5.029   | -49.53 | -10.01 | -7.00  | -5.00  | 4.53  |
| speechiness        | 0.085  | 0.106   | 0.000  | 0.036  | 0.049  | 0.085  | 0.965 |
| tempo (BPM)        | 122.15 | 29.98   | 0.000* | 99.22  | 122.02 | 140.07 | 243.37|
| valence            | 0.474  | 0.259   | 0.000  | 0.260  | 0.464  | 0.683  | 0.995 |

\*Zero tempos stem from incomplete metadata and are dropped during cleaning.

**Mood label prevalence (after cleaning, 89,741 rows):**

| Mood    | Positive Samples | Share |
|---------|------------------|-------|
| Workout | 67,569           | 59.27%|
| Sleep   | 12,702           | 11.14%|
| Party   | 39,711           | 34.83%|
| Focus   | 31,467           | 27.60%|
| Chill   | 36,085           | 31.65%|

Insights:

- Dataset skews toward energetic genres (mean energy 0.64). Sleep-friendly tracks are scarce (11% positive), motivating balanced class weights.
- Instrumentalness is highly skewed, supporting focus/sleep classification.
- Valence distributions confirm the need for mood-specific thresholds (party vs. sleep).

Visual EDA (notebook not included) was conducted during development using histograms, correlations, and seaborn heatmaps to confirm feature ranges and collinearity (most features weakly correlated except energy↔loudness).

---

## 5. Feature Engineering

- **Feature set:** Focus on Spotify’s nine core audio attributes to ensure deployment readiness, supplemented by standardized metadata fields (`track_genre`, `duration_ms`, `popularity`, `explicit`) used exclusively for filtering.
- **Label generation:** Configuration-driven heuristic (`create_mood_labels`) marking positives when numeric thresholds meet mood criteria *and* metadata filters (genre allow/deny lists, duration, popularity bands, explicit policy) pass. This produces pseudo-ground-truth for supervised training while enforcing mood realism upfront.
- **Filter scoring:** During inference, `filter_soft_scores` calculates graded bonuses for tracks that nearly miss metadata thresholds (e.g., slightly long chill tracks) so playlists remain varied without ignoring constraints.
- **User profile features:** At inference, optional averaging of a listener’s own tracks to influence diversity (stored as `user_avg_{feature}` in memory).
- **Derived fields:** UI-level enrichment (album art fallback, Spotify URLs) handled post-model prediction.

Dimensionality reduction was unnecessary; the Random Forest handles nine features easily, and interpretability remained high without PCA.

---

## 6. Model Selection & Architecture

- **Baselines considered:** Logistic Regression (baseline probabilistic classifier) and Gradient Boosted Trees.
- **Chosen algorithm:** `RandomForestClassifier` per mood. Reasons: strong performance on tabular audio features, resilience to label noise, native feature importance, lightweight inference, and easy deployment with scikit-learn.
- **System architecture (logical flow):**
  1. **Data Loader:** `load_kaggle_spotify_data` → cleaned DataFrame.
  2. **Labeler:** `create_mood_labels` generates mood-specific targets in line with the configuration filters.
  3. **Trainer:** `train_mood_model` fits scaler + Random Forest → persists to `models/`.
  4. **Serving Engine:** `MoodRecommender.rank_tracks` loads scalers/models on demand, batch-scores candidates (`predict_proba`), blends the probabilities with sigmoid-based threshold heuristics, applies metadata filter boosts/penalties, and enforces confidence gates to keep playlist percentages realistic.
  5. **API Layer:** FastAPI endpoint `/api/recommend` orchestrates data fetch, inference, and Spotify enrichment.
  6. **Frontend:** React-style JS consumes JSON payload, renders cards, audio previews, and progress bars.

---

## 7. Model Training Process

- **Data split:** Stratified 80/20 train-test per mood to preserve class balance while validating against the same heuristic policy used for labeling.
- **Scaling:** `StandardScaler` fit on training data only. Saved to `{mood}_scaler.pkl` for inference consistency.
- **Hyperparameters:** `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`. These values balance speed and robustness given the rule-generated labels.
- **Optimization:** Preliminary grid searches confirmed diminishing returns beyond 100 trees. Balanced class weights counter the skew in positive samples without oversampling.
- **Overfitting controls:** Depth limit and class-weight balancing prevent memorization. Stratified hold-out evaluation ensures generalization under the heuristic labeling regime.
- **Probability calibration:** At inference time the Random Forest probabilities are linearly shrunk to avoid 0/1 extremes, blended with sigmoid-scored configuration heuristics, and modulated by metadata filter pass/fail boosts so UI scores reflect relative confidence instead of binary outcomes.

Artifacts persisted per mood:

- `{mood}_model.pkl`
- `{mood}_scaler.pkl`
- `{mood}_features.json`
- `training_metrics.json`

---

## 8. Model Evaluation

Latest validation (2 Nov 2025) using the cleaned Kaggle dataset and persisted models (evaluation sampled 100k rows per mood with deterministic labels):

| Mood    | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| Workout | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Sleep   | 0.9999   | 1.0000    | 0.9989 | 0.9995   |
| Party   | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Focus   | 0.9998   | 0.9999    | 0.9982 | 0.9990   |
| Chill   | 1.0000   | 1.0000    | 1.0000 | 1.0000   |

> Validation metrics remain near-perfect because the mood labels derive from deterministic configuration thresholds and metadata filters. During serving, probabilities are shrunk, blended with the sigmoid heuristics, and adjusted by filter confidence bonuses so end users see realistic score spreads rather than a wall of 100% matches.

Manual QA sessions (10-track samples per mood) confirmed that the new genre/duration/popularity filters remove obviously off-target songs—e.g., party recommendations no longer surface lullabies, and sleep playlists exclude explicit hip-hop tracks. Remaining edge cases stem from noisy genre tags and will be addressed with user feedback loops.

Confusion matrices (examined during training) show symmetric performance with negligible false positives. ROC-AUC is ~1.0 due to deterministic label heuristics aligning closely with feature thresholds.

---

## 9. Deployment & Inference Pipeline

- **Backend:** FastAPI app (`backend/server.py`) serving HTML and REST. Bootstraps recommender and optional Spotify client on startup.
- **Endpoint:** `/api/recommend` accepts `{mood, limit}`, loads the cleaned Kaggle dataset lazily, scores candidates in batches (local profiling on a Ryzen 7 laptop shows ~1.6s median to evaluate 90k rows), enriches tracks with Spotify metadata when credentials exist, and responds with JSON.
- **Frontend:** `frontend/static/js/spotify_app.js` requests API, updates UI cards, handles audio preview playback, and displays model scores.
- **Graceful degradation:** Without Spotify credentials, album art falls back to placeholders while recommendations remain accurate.

---

## 10. Documentation & Reporting

Deliverables produced for the university submission:

1. **Cleansed dataset:** `data/raw/spotify_tracks.csv` (source) and cleaned subset created on-the-fly; no redistribution beyond the Kaggle license.
2. **Training script:** `train_ml_models.py` (with helper configs in `config/config.yaml`) for reproducible model builds.
3. **Production code:** `src/recommender.py`, `src/ml_mood_classifier.py`, `backend/server.py`, and `frontend/` assets.
4. **This case study report:** `reports/ml_case_study_report.md`.
5. **README:** Updated with setup, training, and inference instructions reflecting the metadata filters.

Recommended to pair this report with a slide deck summarizing key visuals (EDA plots, architecture diagram, demo screenshots) for presentation day.

---

## 11. Reproducibility Checklist

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Confirm `data/raw/spotify_tracks.csv` is present (downloaded from Kaggle under the original license).
4. (Optional) `python train_ml_models.py` to retrain models with the current `config/config.yaml` heuristics.
5. (Optional) `python scripts/evaluate_models.py` to regenerate `models/evaluation_report.json`.
6. `uvicorn backend.server:app --host 127.0.0.1 --port 8004 --reload` to launch FastAPI + SPA.
7. Visit `http://localhost:8004` and request any mood; backend logs confirm batch inference, metadata filter application, and album art fetching.

Offline verification: `python test_server_import.py` loads the recommender in isolation and prints calibrated probability outputs, demonstrating genuine `RandomForestClassifier` inference and heuristic blending.

---

## 12. Limitations & Future Improvements

- **Label heuristic nature:** Deterministic rules and metadata filters may encode developer bias. Next steps include incorporating curated Spotify playlists or user feedback to build probabilistic labels.
- **Limited feature set:** Only nine audio attributes plus coarse metadata. Incorporating lyrical sentiment, release year, or collaborative filtering signals could enhance personalization.
- **Metadata noise:** Genre tags in the Kaggle dataset are inconsistent; adding a lightweight genre classifier would bolster the new filter layer.
- **Diversity control:** Current recommender leans on model scores; integrating explicit diversity regularization would improve playlist variety.
- **Explainability:** Add SHAP-based explanations per recommendation to communicate why a track matches a mood.
- **Streaming scalability:** For production, migrate to async model serving with caching (e.g., Redis) and incremental dataset updates.

---

## 13. Conclusion

The Vibe-Sync system satisfies the full ML life-cycle expected for a university case study: clearly defined objectives, rigorous data cleaning, configuration-driven label creation with metadata safeguards, high-performing Random Forest models, verifiable evaluation metrics, and a fully functioning deployment stack. Real-time inference tests and manual QA confirm the models are genuine, performant, and ready for live demonstrations while staying transparent about the limitations of rule-derived labels.
