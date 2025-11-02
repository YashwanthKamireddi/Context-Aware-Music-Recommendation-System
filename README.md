vibe-sync/
# 🎵 Vibe-Sync: Context-Aware Music Recommendation System

Random Forest models trained on 114k Spotify tracks deliver mood-aware playlists in a Spotify-style web UI backed by FastAPI.

## 🚀 What’s Included

- **Genuine ML pipeline** – five binary Random Forest classifiers (workout, chill, party, focus, sleep) using only the nine Spotify audio features that ship with the Kaggle dataset.
- **Real training script** – `scripts/train_models.py` preprocesses, labels, trains, evaluates, and persists models + scalers + feature lists.
- **Production FastAPI backend** – serves the React-like frontend, loads the Kaggle data, scores recommendations in vectorized batches, and hydrates album art via Spotify when credentials are supplied.
- **Modern Spotify-style frontend** – `frontend/` renders dynamic playlists, autoplay previews, and mood cards with live ML scores.
- **End-to-end tooling** – PowerShell/Windows friendly scripts, reproducible requirements, logging, and model artifacts checked in for immediate use.

## 🧱 Project Structure

```
Context-Aware-Music-Recommendation-System/
├── backend/                 # FastAPI server (REST + HTML)
├── frontend/                # Static assets (templates, JS, CSS)
├── src/                     # Core ML code (recommender, classifier, utils)
├── models/                  # Trained ML models + scalers (per mood)
├── data/                    # Kaggle dataset (raw) + processed caches
├── config/                  # Configuration files (YAML)
├── results/                 # Model evaluation results & plots
├── reports/                 # ML case study reports
├── scripts/                 # Utility scripts
├── ML_Case_Study_Workflow.ipynb  # Educational notebook
├── scripts/train_models.py  # Model training script
├── requirements.txt         # Python dependencies
├── pyrightconfig.json       # Python type checking config
└── README.md
```

## ⚙️ Setup

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YashwanthKamireddi/Context-Aware-Music-Recommendation-System.git
cd Context-Aware-Music-Recommendation-System

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Spotify tracks dataset from Kaggle and place it in `data/raw/`:
- File: `spotify_tracks.csv`
- Place at: `data/raw/spotify_tracks.csv`

### 3. Spotify API (Optional but Recommended)

Create a `.env` file in the root directory:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8004/callback
```

Get credentials from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

Without credentials, the system works but album art falls back to placeholders.

## 🧠 Train the ML Models

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

python scripts/train_models.py
```

This will:
- Load the Spotify dataset from `data/raw/spotify_tracks.csv`
- Train 5 binary classifiers (workout, chill, party, focus, sleep)
- Save models, scalers, and feature lists to `models/` directory
- Generate evaluation reports in `results/`

**Output files per mood:**
- `{mood}_model.pkl` - Trained classifier
- `{mood}_scaler.pkl` - Feature scaler
- `{mood}_features.json` - Feature configuration

## 🌐 Run the Web App

### Start the Server

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Start the FastAPI server
uvicorn backend.server:app --host 127.0.0.1 --port 8004 --reload
```

### Access the Application

- **Web Interface**: http://localhost:8004
- **API Endpoint**: http://localhost:8004/api/recommend

The system will:
1. Load 114K+ Spotify tracks from `data/raw/spotify_tracks.csv`
2. Score all tracks using trained ML models (~1 second)
3. Fetch album art from Spotify (if credentials provided)
4. Display mood-based recommendations in a Spotify-style UI

Scores shown in the UI are calibrated by blending model probabilities with the configuration heuristics, so each playlist displays a realistic confidence range instead of every card reading 100%.

## 🤖 Core ML Details

- **Features:** `acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence`
- **Models:** Random Forest Classifier (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`)
- **Training Data:** 114K+ Spotify tracks with mood-based labeling
- **Architecture:** Binary classification for each mood (workout, chill, party, focus, sleep)
- **Serving:** Batch prediction with vectorized scoring, sigmoid-based heuristic blending, and confidence filtering so playlist percentages span a realistic range instead of defaulting to 100%

## 🛠️ Useful Commands

### Test Recommendations Offline

```bash
python -c "import pandas as pd; from src.recommender import MoodRecommender; from src.utils import load_config; cfg = load_config(); rec = MoodRecommender(cfg); df = pd.read_csv('data/raw/spotify_tracks.csv'); out = rec.recommend(df, mood='workout', top_k=5); print(out[['name','artists','final_score']])"
```

### Test API Endpoint

```bash
# Using curl
curl -X POST "http://localhost:8004/api/recommend" \
     -H "Content-Type: application/json" \
     -d '{"mood":"sleep","limit":5}'

# Using PowerShell
Invoke-RestMethod -Uri "http://localhost:8004/api/recommend" -Method POST -Body '{"mood":"sleep","limit":5}' -ContentType 'application/json'
```

### Check Model Status

```bash
python -c "from src.ml_mood_classifier import MLMoodClassifier; clf = MLMoodClassifier(); print('Models loaded:', len(clf.models))"
```

## 📝 Configuration

- `config/config.yaml` – Mood thresholds, model parameters, dataset paths
- `frontend/static/js/spotify_app.js` – UI interactions, playback controls
- `backend/server.py` – API endpoints, Spotify integration, caching
- `.env` – Spotify API credentials (create from `.env.example`)


### Local Development

```bash
# Activate virtual environment
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Start the server
uvicorn backend.server:app --host 127.0.0.1 --port 8004 --reload
```

## 📄 License & Attribution

Academic project for ML coursework. Kaggle dataset licensing terms apply. Spotify trademarks remain property of Spotify AB.

---

**Last updated:** October 14, 2025
