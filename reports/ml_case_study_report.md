# Machine Learning Case Study: Context-Aware Music Recommendation System

**Project Name:** Vibe-Sync – Mood-Aware Spotify-Style Recommender
**Author:** Yashwanth Kamireddi
**Last Updated:** 8 October 2025

---

## 1. Problem Understanding & Objective Definition

**Problem statement (one-page narrative):**

Modern music platforms excel at long-term personalization but often ignore a listener’s immediate context. Users who want “high-energy workout tracks” or “calming focus playlists” must search manually, wasting time and encountering inconsistent results. Vibe-Sync aims to close this gap by learning mood signatures directly from Spotify audio attributes and serving mood-aligned playlists instantly. The system must operate without Spotify API rate limits by relying on an offline Kaggle dataset while still delivering a Spotify-quality web experience. It should ingest high-volume track catalogs (114k songs), score them using trained models, and return diverse, high-confidence recommendations within one second per mood request.

**Business objectives:**

- Deliver context-aware playlists for five predefined moods (Workout, Chill, Party, Focus, Sleep).
- Maintain Spotify-like UX with album art, preview links, and score transparency.
- Allow optional Spotify API enrichment without making the service dependent on third-party quotas.
- Support rapid experimentation and re-training as mood definitions evolve.

**ML task type:** Multi-label, binary classification (one independent classifier per mood). Each model predicts whether a track is suitable (`1`) or not (`0`) for the target mood.

**Success metrics:**

- **Primary:** Accuracy and F1-score on hold-out test sets per mood (>99% achieved).
- **Secondary:** Precision/Recall balance to avoid false positives (party tracks in sleep playlists) and false negatives (missing strong candidates).
- **Operational:** End-to-end latency < 1.5 seconds per recommendation call on commodity hardware (achieved ~1.0s on Windows laptop).

---

## 2. Data Collection

- **Source:** Kaggle "Spotify Tracks" dataset (`data/raw/spotify_tracks.csv`, 114,000 rows, 21 columns). Loaded via `src/load_kaggle_data.py`.
- **Structure:** Structured tabular CSV with consistent schema (numeric audio features + textual metadata).
- **Coverage:** Global mix of genres and popularity tiers. Audio attributes align with Spotify API definitions.
- **Justification:** Provides a large, clean corpus without API latency or rate limits, enabling reproducible university-grade experiments.

---

## 3. Data Preprocessing & Cleaning

Implemented in `train_kaggle_models.py`, `src/recommender.py::_prepare_candidates`, and `src/load_kaggle_data.py`.

1. **Column standardization:** Rename Kaggle-provided columns (`track_name → name`, `track_id → id`, `artist_name → artists`, `album_name → album`).
2. **Missing data handling:** Drop rows with nulls in the nine mandatory audio features. Removes ~24k rows to ensure model integrity.
3. **Type coercion:** Force audio features to numeric floats; coerce invalid strings to `NaN` before dropping.
4. **Outlier treatment:** Clip loudness to [-60, 0] dB and tempo to [30, 250] BPM to align with model training heuristics.
5. **Deduplication:** Remove duplicate tracks (by `id` or `track_id`) to avoid sampling bias.
6. **Train/test split:** Stratified 80/20 split with `train_test_split(..., stratify=y)` for each mood classifier.
7. **Scaling:** Standardize numeric features via `StandardScaler` before feeding LightGBM (scalers persisted per mood).

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

- **Feature set:** Restricted to Spotify’s nine core audio attributes to ensure deployment readiness and avoid leakage from popularity metadata.
- **Label generation:** Deterministic heuristic (`create_labels`) marking positives when ≥60% of mood-specific criteria pass. This creates pseudo-ground-truth for supervised training.
- **User profile features:** At inference, optional averaging of a listener’s own tracks to influence diversity (stored as `user_avg_{feature}` in memory).
- **Derived fields:** UI-level enrichment (album art fallback, Spotify URLs) handled post-model prediction.

Dimensionality reduction was unnecessary; LightGBM handles nine features easily, and interpretability remained high without PCA.

---

## 6. Model Selection & Architecture

- **Baselines considered:** Logistic Regression and Random Forest (initial experiments showed underfitting and slower scoring).
- **Chosen algorithm:** `LGBMClassifier` per mood. Reasons: robustness on tabular data, native handling of class imbalance via gradient boosting, fast inference, and feature importance introspection.
- **System architecture (logical flow):**
  1. **Data Loader:** `load_kaggle_spotify_data` → cleaned DataFrame.
  2. **Labeler:** `create_labels` generates mood-specific targets.
  3. **Trainer:** `train_mood_model` fits scaler + LightGBM → persists to `models/`.
  4. **Serving Engine:** `MoodRecommender.rank_tracks` loads scalers/models on demand, batch-scores candidates (`predict_proba`), and ranks by probability.
  5. **API Layer:** FastAPI endpoint `/api/recommend` orchestrates data fetch, inference, and Spotify enrichment.
  6. **Frontend:** React-style JS consumes JSON payload, renders cards, audio previews, and progress bars.

---

## 7. Model Training Process

- **Data split:** Stratified 80/20 train-test per mood to preserve class balance.
- **Scaling:** `StandardScaler` fit on training data only. Saved to `{mood}_scaler.pkl` for inference consistency.
- **Hyperparameters:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `random_state=42`, `verbose=-1`. These values strike a balance between speed and high accuracy given the deterministic labels.
- **Optimization:** Preliminary grid searches confirmed diminishing returns beyond 100 trees. Class weights rely on LightGBM’s default gradient boosting behavior because positive rate >10% for all moods.
- **Overfitting controls:** Depth limit and learning rate prevent memorization. Stratified hold-out evaluation ensures generalization.

Artifacts persisted per mood:

- `{mood}_lightgbm.pkl`
- `{mood}_scaler.pkl`
- `{mood}_features.json`

---

## 8. Model Evaluation

Latest validation (8 Oct 2025) using the cleaned Kaggle dataset and persisted models:

| Mood    | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| Workout | 0.9993   | 0.9991    | 0.9997 | 0.9994   |
| Sleep   | 0.9999   | 0.9996    | 0.9998 | 0.9997   |
| Party   | 0.9993   | 0.9988    | 0.9991 | 0.9990   |
| Focus   | 0.9989   | 0.9975    | 0.9989 | 0.9982   |
| Chill   | 0.9978   | 0.9957    | 0.9973 | 0.9965   |

Confusion matrices (examined during training) show symmetric performance with negligible false positives. ROC-AUC is ~1.0 due to deterministic label heuristics aligning closely with feature thresholds.

---

## 9. Deployment & Inference Pipeline

- **Backend:** FastAPI app (`backend/server.py`) serving HTML and REST. Bootstraps recommender and optional Spotify client on startup.
- **Endpoint:** `/api/recommend` accepts `{mood, limit}`, loads Kaggle dataset lazily, scores candidates in batches (~90k rows processed in <1.1s), enriches tracks with Spotify metadata when credentials exist, and responds with JSON.
- **Frontend:** `frontend/static/js/spotify_app.js` requests API, updates UI cards, handles audio preview playback, and displays model scores.
- **Graceful degradation:** Without Spotify credentials, album art falls back to placeholders while recommendations remain accurate.

---

## 10. Documentation & Reporting

Deliverables produced for the university submission:

1. **Cleansed dataset:** `data/raw/spotify_tracks.csv` (source) and cleaned subset created on-the-fly; no redistribution beyond Kaggle license.
2. **Training script:** `train_kaggle_models.py` for reproducible model builds.
3. **Production code:** `src/recommender.py`, `backend/server.py`, `frontend/` assets.
4. **This case study report:** `reports/ml_case_study_report.md`.
5. **README:** Updated with setup, training, and inference instructions.

Recommended to pair this report with a slide deck summarizing key visuals (EDA plots, architecture diagram, demo screenshots) for presentation day.

---

## 11. Reproducibility Checklist

1. `python -m venv .venv && .\.venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Verify `data/raw/spotify_tracks.csv` exists (Kaggle download).
4. (Optional) `python train_kaggle_models.py` to retrain models.
5. `./start_server.ps1` to launch FastAPI + SPA.
6. Hit `http://localhost:8000` and request any mood; backend logs confirm batch inference and album art fetching.

Offline verification: `python -c "...MoodRecommender..."` script (see README) confirms genuine `LGBMClassifier` usage and non-trivial probability outputs.

---

## 12. Limitations & Future Improvements

- **Label heuristic nature:** Deterministic rules may encode bias; future work could gather user feedback or Spotify playlist labels for supervised learning.
- **Limited feature set:** Only nine audio attributes. Incorporating lyrical sentiment, release era, or collaborative filtering signals could enhance personalization.
- **Diversity control:** Current recommender leans on model scores; integrating explicit diversity regularization would improve playlist variety.
- **Explainability:** Add SHAP-based explanations per recommendation to communicate why a track matches a mood.
- **Streaming scalability:** For production, migrate to async model serving with caching (e.g., Redis) and incremental dataset updates.

---

## 13. Conclusion

The Vibe-Sync system satisfies the full ML life-cycle expected for a university case study: clearly defined objectives, rigorous data cleaning, deterministic yet meaningful label creation, high-performing LightGBM models, verifiable evaluation metrics, and a fully functioning deployment stack. Real-time inference tests and offline metrics confirm the models are genuine, performant, and ready for live demonstrations.
