"""
FastAPI Backend Server - Production Grade
Real-time mood-based music recommendations with Spotify integration
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import sys
import os
import pandas as pd

# Add parent directory and src to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.spotify_client import SpotifyClient
from src.recommender import MoodRecommender
from src.utils import load_config, setup_logging

# Setup
logger = setup_logging()
app = FastAPI(title="Vibe-Sync API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Global instances
spotify_client = None
recommender = None
tracks_df = None


# Pydantic Models
class RecommendationRequest(BaseModel):
    mood: str
    limit: int = 20
    user_tracks: Optional[List[str]] = None


class PlaylistCreate(BaseModel):
    mood: str
    name: str
    track_ids: List[str]


class SpotifySetup(BaseModel):
    client_id: str
    client_secret: str


# Startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global spotify_client, recommender, tracks_df

    logger.info("🚀 Starting Vibe-Sync API...")

    try:
        # Try to load Spotify client
        spotify_client = SpotifyClient()
        logger.info("✅ Spotify client initialized")
    except Exception as e:
        logger.warning(f"⚠️ Spotify client not initialized: {e}")
        spotify_client = None

    try:
        # Load recommender
        config = load_config()
        recommender = MoodRecommender(config)
        logger.info("✅ Recommender system loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load recommender: {e}")
        recommender = None

    logger.info("✅ Vibe-Sync API ready!")


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface - Spotify Clone UI"""
    return templates.TemplateResponse("index_new.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "spotify_connected": spotify_client is not None,
        "recommender_loaded": recommender is not None,
        "version": "1.0.0"
    }


@app.get("/api/moods")
async def get_moods():
    """Get available moods"""
    moods = [
        {"id": "workout", "name": "Workout", "emoji": "🏋️", "description": "High energy, fast tempo"},
        {"id": "chill", "name": "Chill", "emoji": "😌", "description": "Calm and relaxing vibes"},
        {"id": "party", "name": "Party", "emoji": "🎉", "description": "Dance-worthy beats"},
        {"id": "focus", "name": "Focus", "emoji": "📚", "description": "Concentration music"},
        {"id": "sleep", "name": "Sleep", "emoji": "😴", "description": "Peaceful and slow"},
    ]
    return {"moods": moods}


@app.post("/api/recommend")
async def recommend_tracks(request: RecommendationRequest):
    """
    Get personalized recommendations for a mood
    """
    global tracks_df, spotify_client, recommender

    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender system not loaded"
        )

    try:
        # Load or fetch tracks
        if tracks_df is None or len(tracks_df) == 0:
            # PRIORITY 1: Try Kaggle dataset (NO API RATE LIMITS!)
            kaggle_path = os.path.join(parent_dir, 'data', 'raw', 'spotify_tracks.csv')

            if os.path.exists(kaggle_path):
                logger.info(f"📁 Loading Kaggle dataset from {kaggle_path}...")
                try:
                    tracks_df = pd.read_csv(kaggle_path, encoding='latin-1')

                    # Column mapping for Kaggle dataset format
                    column_mapping = {
                        'track_name': 'name',
                        'track_id': 'id',
                        'artist_name': 'artists',
                        'album_name': 'album'
                    }

                    for old_col, new_col in column_mapping.items():
                        if old_col in tracks_df.columns:
                            tracks_df.rename(columns={old_col: new_col}, inplace=True)

                    # Add placeholder for album art (Kaggle dataset doesn't have images)
                    if 'album_image' not in tracks_df.columns and 'album_art' not in tracks_df.columns:
                        tracks_df['album_image'] = 'https://via.placeholder.com/300x300.png?text=No+Image'

                    # Ensure we have the required audio features
                    required_features = ['acousticness', 'danceability', 'energy',
                                       'instrumentalness', 'liveness', 'loudness',
                                       'speechiness', 'tempo', 'valence']

                    if all(feat in tracks_df.columns for feat in required_features):
                        # Clean data
                        tracks_df = tracks_df.dropna(subset=required_features)
                        logger.info(f"✅ Loaded {len(tracks_df)} tracks from Kaggle dataset (NO API LIMITS!)")
                    else:
                        logger.warning(f"⚠️ Kaggle data missing features. Has: {list(tracks_df.columns)}")
                        tracks_df = None
                except Exception as e:
                    logger.warning(f"⚠️ Could not load Kaggle data: {e}")
                    tracks_df = None

            # PRIORITY 2: Try cached processed data
            if tracks_df is None or len(tracks_df) == 0:
                cached_data_path = os.path.join(parent_dir, 'data', 'processed', 'tracks_labeled.csv')

                if os.path.exists(cached_data_path):
                    logger.info(f"📁 Loading cached tracks from {cached_data_path}...")
                    try:
                        tracks_df = pd.read_csv(cached_data_path, encoding='latin-1')
                        required_features = ['acousticness', 'danceability', 'energy',
                                           'instrumentalness', 'liveness', 'loudness',
                                           'speechiness', 'tempo', 'valence']

                        if all(feat in tracks_df.columns for feat in required_features):
                            logger.info(f"✅ Loaded {len(tracks_df)} tracks from cache")
                        else:
                            tracks_df = None
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load cached data: {e}")
                        tracks_df = None

            # PRIORITY 3: Try bundled lightweight sample dataset (for Spaces deployments)
            if tracks_df is None or len(tracks_df) == 0:
                sample_data_path = os.path.join(parent_dir, 'data', 'sample', 'tracks_sample.csv')

                if os.path.exists(sample_data_path):
                    logger.info(f"📁 Loading sample tracks from {sample_data_path}...")
                    try:
                        tracks_df = pd.read_csv(sample_data_path, encoding='utf-8')
                        required_features = ['acousticness', 'danceability', 'energy',
                                           'instrumentalness', 'liveness', 'loudness',
                                           'speechiness', 'tempo', 'valence']

                        available_features = [feat for feat in required_features if feat in tracks_df.columns]
                        if available_features:
                            tracks_df = tracks_df.dropna(subset=available_features)
                            logger.info(f"✅ Loaded {len(tracks_df)} tracks from bundled sample dataset")
                        else:
                            logger.warning("⚠️ Sample dataset missing audio features")
                            tracks_df = None
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load sample data: {e}")
                        tracks_df = None

            # PRIORITY 4: Fetch from Spotify API (LAST RESORT - rate limits!)
            if tracks_df is None or len(tracks_df) == 0:
                if spotify_client is None:
                    raise HTTPException(
                        status_code=503,
                        detail="No dataset available. Please add data/raw/spotify_tracks.csv or configure Spotify API"
                    )

                logger.warning("⚠️ Fetching from Spotify API (may hit rate limits)...")

                genres = [
                    'edm', 'hardstyle', 'metal', 'punk', 'drum-and-bass',
                    'ambient', 'piano', 'classical', 'acoustic', 'meditation',
                    'dance', 'disco', 'house', 'techno', 'funk',
                    'lo-fi', 'study', 'instrumental', 'chillout',
                    'pop', 'rock', 'electronic', 'hip-hop', 'jazz',
                    'r-n-b', 'indie', 'country', 'blues', 'soul'
                ]
                tracks_df = spotify_client.build_dataset_from_genres(
                    genres,
                    tracks_per_genre=50
                )

                if tracks_df.empty:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to load any dataset. Please add Kaggle dataset to data/raw/spotify_tracks.csv"
                    )

                logger.info(f"✅ Loaded {len(tracks_df)} tracks from Spotify API")

        # Get recommendations
        logger.info(f"Generating recommendations for mood: {request.mood}")

        recommendations = recommender.recommend(
            tracks_df,
            mood=request.mood,
            top_k=request.limit
        )

        # Fetch album art for tracks with Spotify IDs (batch operation)
        track_ids = []
        album_art_map = {}

        for _, track in recommendations.iterrows():
            track_id = track.get('id', '') or track.get('track_id', '')
            if track_id and track_id != 'Unknown':
                track_ids.append(track_id)

        # Batch fetch album art from Spotify (up to 50 at a time)
        if track_ids and spotify_client:
            try:
                logger.info(f"🎨 Fetching album art for {len(track_ids)} tracks from Spotify...")
                for i in range(0, len(track_ids), 50):
                    batch = track_ids[i:i+50]
                    tracks_info = spotify_client.sp.tracks(batch)

                    if tracks_info and 'tracks' in tracks_info:
                        for track_info in tracks_info['tracks']:
                            if track_info and 'id' in track_info and 'album' in track_info:
                                tid = track_info['id']
                                images = track_info['album'].get('images', [])
                                if images:
                                    album_art_map[tid] = images[0]['url']  # Largest image

                logger.info(f"✅ Fetched {len(album_art_map)} album artworks")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch album art: {e}")
        else:
            if not spotify_client:
                logger.info("🎨 Spotify credentials not provided - using placeholder album art")

        # Format response with correct field names
        results = []
        for _, track in recommendations.iterrows():
            track_id = track.get('id', '') or track.get('track_id', '')

            # Build proper Spotify URL
            spotify_url = ''
            if track_id and track_id != 'Unknown':
                spotify_url = f"https://open.spotify.com/track/{track_id}"

            # Get album art (from Spotify batch fetch or use placeholder)
            album_art = album_art_map.get(track_id) or track.get('album_image', '') or track.get('album_art', '')

            # Fallback to nice placeholder if still no image
            if not album_art:
                album_art = 'https://via.placeholder.com/300x300/1DB954/FFFFFF?text=🎵'

            # Get album name (handle different column names)
            album_name = track.get('album', '') or track.get('album_name', 'Unknown Album')

            # Get artist name (handle different formats)
            artist_name = track.get('artists', '') or track.get('artist', 'Unknown Artist')

            # Build complete track object with audio features
            audio_features = {
                "energy": float(track.get('energy', 0)),
                "danceability": float(track.get('danceability', 0)),
                "valence": float(track.get('valence', 0)),
                "tempo": float(track.get('tempo', 0)),
                "acousticness": float(track.get('acousticness', 0)),
                "instrumentalness": float(track.get('instrumentalness', 0)),
                "speechiness": float(track.get('speechiness', 0)),
                "liveness": float(track.get('liveness', 0)),
                "loudness": float(track.get('loudness', 0))
            }

            results.append({
                "id": track_id,
                "title": track.get('name', 'Unknown Track'),
                "name": track.get('name', 'Unknown Track'),
                "artist": artist_name,
                "artists": artist_name,
                "album": album_name,
                "album_name": album_name,
                "score": float(track.get('final_score', 0.0)),
                "spotify_url": spotify_url,
                "preview_url": track.get('preview_url', ''),
                "album_art": album_art,
                "album_image": album_art,
                "duration_ms": int(track.get('duration_ms', 0) or 0),
                "audio_features": audio_features
            })

        logger.info(f"✅ Generated {len(results)} recommendations")
        logger.info(f"📊 Score range: {min([r['score'] for r in results]):.3f} - {max([r['score'] for r in results]):.3f}")

        return {
            "mood": request.mood,
            "count": len(results),
            "recommendations": results
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/setup-spotify")
async def setup_spotify(setup: SpotifySetup):
    """
    Configure Spotify API credentials
    """
    global spotify_client

    try:
        # Save credentials to .env
        env_path = ".env"
        with open(env_path, "w") as f:
            f.write(f"SPOTIFY_CLIENT_ID={setup.client_id}\n")
            f.write(f"SPOTIFY_CLIENT_SECRET={setup.client_secret}\n")
            f.write("SPOTIFY_REDIRECT_URI=http://localhost:8000/callback\n")

        # Reinitialize client
        spotify_client = SpotifyClient()

        logger.info("✅ Spotify credentials configured")

        return {
            "status": "success",
            "message": "Spotify API configured successfully"
        }

    except Exception as e:
        logger.error(f"Error setting up Spotify: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_tracks(query: str, limit: int = 20):
    """
    Search for tracks on Spotify
    """
    if spotify_client is None:
        raise HTTPException(
            status_code=503,
            detail="Spotify client not configured"
        )

    try:
        tracks = spotify_client.search_tracks(query, limit=limit)

        results = []
        for track in tracks:
            results.append({
                "id": track['id'],
                "name": track['name'],
                "artist": ', '.join([a['name'] for a in track['artists']]),
                "album": track['album']['name'],
                "spotify_url": f"https://open.spotify.com/track/{track['id']}",
                "preview_url": track.get('preview_url', ''),
                "image": track['album']['images'][0]['url'] if track['album']['images'] else ''
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """
    Get system statistics
    """
    stats = {
        "total_tracks": len(tracks_df) if tracks_df is not None else 0,
        "moods_available": 5,
        "models_trained": 5 if recommender is not None else 0,
        "spotify_connected": spotify_client is not None
    }

    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
