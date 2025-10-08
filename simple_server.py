"""
Simple API Server - Testing Version
No templates, just API endpoints
"""

from fastapi import FastAPI, HTTPException
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

# Global instances
spotify_client = None
recommender = None
tracks_df = None


# Pydantic Models
class RecommendationRequest(BaseModel):
    mood: str
    limit: int = 20


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
    logger.info("📡 API available at: http://localhost:8000")
    logger.info("📖 API docs at: http://localhost:8000/docs")


# Routes
@app.get("/")
async def home():
    """API Home"""
    return {
        "message": "🎵 Vibe-Sync API",
        "description": "Context-aware music recommendation system",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "moods": "/api/moods",
            "recommend": "/api/recommend",
            "search": "/api/search",
            "stats": "/api/stats"
        }
    }


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
    
    if spotify_client is None:
        raise HTTPException(
            status_code=503,
            detail="Spotify client not configured. Please set up API credentials."
        )
    
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender system not loaded"
        )
    
    try:
        # Load or fetch tracks
        if tracks_df is None or len(tracks_df) == 0:
            logger.info("Fetching tracks from Spotify...")
            
            # Fetch diverse tracks from Spotify
            genres = ['pop', 'rock', 'electronic', 'hip-hop', 'workout']
            tracks_df = spotify_client.build_dataset_from_genres(
                genres, 
                tracks_per_genre=20
            )
            
            if tracks_df.empty:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to fetch tracks from Spotify"
                )
            
            logger.info(f"✅ Loaded {len(tracks_df)} tracks from Spotify")
        
        # Get recommendations
        logger.info(f"Generating recommendations for mood: {request.mood}")
        
        recommendations = recommender.recommend(
            tracks_df,
            mood=request.mood,
            top_k=request.limit
        )
        
        # Format response
        results = []
        for _, track in recommendations.iterrows():
            results.append({
                "id": track.get('id', ''),
                "name": track.get('name', 'Unknown'),
                "artist": track.get('artist', 'Unknown Artist'),
                "album": track.get('album', ''),
                "score": float(track.get('final_score', 0)),
                "spotify_url": f"https://open.spotify.com/track/{track.get('id', '')}",
                "duration_ms": int(track.get('duration_ms', 0))
            })
        
        logger.info(f"✅ Generated {len(results)} recommendations")
        
        return {
            "mood": request.mood,
            "count": len(results),
            "recommendations": results
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
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
                "spotify_url": f"https://open.spotify.com/track/{track['id']}"
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
    print("🚀 Starting Vibe-Sync API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
