---
title: Context Aware Music Recommendation
emoji: 🐠
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
short_description: Mood-based playlists with LightGBM plus FastAPI backend
---

vibe-sync/
# 🎵 Vibe-Sync: Context-Aware Music Recommendation System

Real LightGBM models trained on 114k Spotify tracks deliver mood-aware playlists in a Spotify-style web UI backed by FastAPI.

## 🚀 What’s Included

- **Genuine ML pipeline** – five binary LightGBM classifiers (workout, chill, party, focus, sleep) using only the nine Spotify audio features that ship with the Kaggle dataset.
- **Real training script** – `train_kaggle_models.py` preprocesses, labels, trains, evaluates, and persists models + scalers + feature lists.
- **Production FastAPI backend** – serves the React-like frontend, loads the Kaggle data, scores recommendations in vectorized batches, and hydrates album art via Spotify when credentials are supplied.
- **Modern Spotify-style frontend** – `frontend/` renders dynamic playlists, autoplay previews, and mood cards with live ML scores.
- **End-to-end tooling** – PowerShell/Windows friendly scripts, reproducible requirements, logging, and model artifacts checked in for immediate use.

## 🧱 Project Structure

```
Context-Aware-Music-Recommendation-System/
├── backend/             # FastAPI server (REST + HTML)
├── frontend/            # Static assets (templates, JS, CSS)
├── src/                 # Core ML code (recommender, Spotify client, utils)
├── models/              # Trained LightGBM + scalers (per mood)
├── data/                # Kaggle dataset (raw) + processed caches
├── train_kaggle_models.py
├── start_server.ps1     # Launches backend with the project virtualenv
└── requirements.txt
```

## ⚙️ Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Curated dataset for deployments (≤ 1 MB)

Hugging Face Spaces caps individual files at 10 MB, so the FastAPI backend now
ships with a real-but-curated subset of the Kaggle dump stored at
`data/processed/tracks_curated.parquet` (~0.8 MB). The file is generated from the
full dataset using:

```powershell
python scripts/build_curated_dataset.py
```

The script performs stratified sampling across the 113 Spotify genres, keeps the
original audio features, and preserves popular tracks so the ML models receive
authentic inputs. Re-run it whenever you refresh the raw dataset.

### Spotify API (optional but recommended)

Create a `.env` file with:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8000/callback
```

Without credentials the system still works; album art falls back to a placeholder image.

## 🧠 Train the ML Models

```powershell
.\.venv\Scripts\activate
python train_kaggle_models.py
```

Output (per mood): accuracy, classification report, and artifacts saved to `models/`:

- `{mood}_lightgbm.pkl`
- `{mood}_scaler.pkl`
- `{mood}_features.json`

The script automatically renames Kaggle columns (`track_name → name`, etc.) and labels tracks using the mood heuristics defined in `config/config.yaml`.

## 🌐 Run the Web App

```powershell
.\.venv\Scripts\activate
./start_server.ps1
```

Then open http://localhost:8000 and pick a mood. The backend will:

1. Load the curated parquet dataset first (`data/processed/tracks_curated.parquet`),
   falling back to the full Kaggle CSV or the Spotify API only if needed.
2. Vectorize model scoring across the entire dataset (~1 second for all moods).
3. Fetch album art from Spotify in batches when credentials are available.
4. Return a JSON payload with scores, audio features, links, and images consumed by the frontend.

## 🐳 Docker & Free Hosting

The repository includes a production `Dockerfile` and `docker-compose.yml` so you can ship the backend without touching virtual environments.

```powershell
# Build the container
docker build -t vibesync-backend .

# Run locally with port 8000 exposed
docker compose up
```

When you are ready for the cloud:

1. Push the repo (minus the heavyweight datasets) to [Hugging Face Spaces](https://huggingface.co/spaces) and let their Docker runner build the FastAPI backend.
2. Host the static `frontend/` folder on Vercel with the project root set to `frontend/`. The only build step is copying the directory, so Vercel redeploys automatically whenever `main` updates.
3. In `frontend/static/config.js`, set `window.API_BASE_URL` to the Space URL (for example `https://yashhugs-context-aware-music-recommendation.hf.space`).

The complete walkthrough (including the lightweight Space mirror command and Vercel linking) lives in `docs/deployment_guide.md`.

## 🤖 Core ML Details

- **Features:** `acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence`
- **Models:** LightGBM (`n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `class_weight='balanced'`)
- **Metrics (test set):** Workout 99.85%, Sleep 99.97%, Party 99.87%, Focus 99.82%, Chill 99.80% accuracy
- **Serving:** `src/recommender.py` loads scalers/models lazily and ranks candidates using fully vectorized probability scores

## 🛠️ Useful Commands

Run an offline check without starting the server:

```powershell
.\.venv\Scripts\activate
python -c "import pandas as pd; from src.recommender import MoodRecommender; from src.utils import load_config; cfg = load_config(); rec = MoodRecommender(cfg); df = pd.read_csv('data/raw/spotify_tracks.csv'); out = rec.recommend(df, mood='workout', top_k=5); print(out[['name','artists','final_score']])"
```

Inspect the API directly:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/recommend" -Method POST -Body '{"mood":"sleep","limit":5}' -ContentType 'application/json'
```

## 📝 Configuration

- `config/config.yaml` – tweak mood thresholds, model weights, dataset paths.
- `frontend/static/js/spotify_app.js` – UI responses, playback, toast notifications.
- `frontend/static/config.js` – set `window.API_BASE_URL` when hosting the frontend separately (e.g., on Vercel).
- `backend/server.py` – request handling, album art enrichment, caching strategy.

## 📄 License & Attribution

Academic project. Kaggle dataset licensing terms apply. Spotify trademarks remain property of Spotify AB.

---

**Last updated:** October 2025
