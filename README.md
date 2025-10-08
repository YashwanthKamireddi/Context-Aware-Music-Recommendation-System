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

1. Load `data/raw/spotify_tracks.csv` (114,000 tracks).
2. Vectorize model scoring across the entire dataset (~1 second for all moods).
3. Fetch album art from Spotify in batches when credentials are available.
4. Return a JSON payload with scores, audio features, links, and images consumed by the frontend.

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
- `backend/server.py` – request handling, album art enrichment, caching strategy.

## 📄 License & Attribution

Academic project. Kaggle dataset licensing terms apply. Spotify trademarks remain property of Spotify AB.

---

**Last updated:** October 2025
