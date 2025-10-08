"""
Real Spotify API Client
Fetches real music data from Spotify Web API
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class SpotifyClient:
    """
    Real Spotify API integration for fetching tracks and audio features
    """
    
    def __init__(self, use_auth: bool = False):
        """
        Initialize Spotify client
        
        Args:
            use_auth: If True, use OAuth for user-specific features
        """
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8000/callback')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not found! "
                "Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file"
            )
        
        if use_auth:
            # OAuth for user-specific operations (playlists, etc.)
            scope = "user-library-read playlist-modify-public playlist-modify-private"
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=scope
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
        else:
            # Client credentials for general API access
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
        
        logger.info("✅ Spotify client initialized successfully")
    
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Search for tracks on Spotify
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            List of track dictionaries
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = results['tracks']['items']
            logger.info(f"Found {len(tracks)} tracks for query: {query}")
            return tracks
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Get audio features for multiple tracks
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of audio feature dictionaries
        """
        try:
            # Spotify API allows max 100 tracks per request
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                features.extend([f for f in batch_features if f is not None])
            
            logger.info(f"Retrieved audio features for {len(features)} tracks")
            return features
        except Exception as e:
            logger.error(f"Error getting audio features: {e}")
            return []
    
    def get_playlist_tracks(self, playlist_id: str) -> pd.DataFrame:
        """
        Get all tracks from a Spotify playlist with audio features
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            DataFrame with track info and audio features
        """
        try:
            # Get playlist tracks
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            
            # Get all pages
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
            
            # Extract track info
            track_data = []
            track_ids = []
            
            for item in tracks:
                if item['track'] is None:
                    continue
                    
                track = item['track']
                track_data.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                })
                track_ids.append(track['id'])
            
            # Get audio features
            audio_features = self.get_audio_features(track_ids)
            
            # Merge track info with audio features
            df_tracks = pd.DataFrame(track_data)
            df_features = pd.DataFrame(audio_features)
            
            if not df_features.empty:
                df = pd.merge(df_tracks, df_features, on='id', how='left')
            else:
                df = df_tracks
            
            logger.info(f"Retrieved {len(df)} tracks from playlist")
            return df
            
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return pd.DataFrame()
    
    def fetch_tracks_by_genre(self, genre: str, limit: int = 50) -> pd.DataFrame:
        """
        Fetch tracks by genre with audio features
        
        Args:
            genre: Genre name (e.g., 'pop', 'rock', 'electronic')
            limit: Number of tracks to fetch
            
        Returns:
            DataFrame with track info and audio features
        """
        try:
            # Search for tracks in genre
            query = f"genre:{genre}"
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = results['tracks']['items']
            
            if not tracks:
                logger.warning(f"No tracks found for genre: {genre}")
                return pd.DataFrame()
            
            # Extract track info with FULL details
            track_data = []
            track_ids = []
            
            for track in tracks:
                # Get album art (highest quality available)
                album_art = ''
                if track['album'].get('images'):
                    album_art = track['album']['images'][0]['url']  # Highest resolution
                
                track_data.append({
                    'id': track['id'],
                    'track_id': track['id'],  # Duplicate for compatibility
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artists': ', '.join([artist['name'] for artist in track['artists']]),
                    'album': track['album']['name'],
                    'album_image': album_art,
                    'album_art': album_art,
                    'preview_url': track.get('preview_url', ''),
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'spotify_url': f"https://open.spotify.com/track/{track['id']}"
                })
                track_ids.append(track['id'])
            
            # Get audio features
            audio_features = self.get_audio_features(track_ids)
            
            # Merge
            df_tracks = pd.DataFrame(track_data)
            df_features = pd.DataFrame(audio_features)
            
            if not df_features.empty:
                df = pd.merge(df_tracks, df_features, on='id', how='left')
            else:
                df = df_tracks
            
            logger.info(f"Fetched {len(df)} tracks for genre: {genre}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching tracks by genre: {e}")
            return pd.DataFrame()
    
    def build_dataset_from_genres(self, genres: List[str], tracks_per_genre: int = 100) -> pd.DataFrame:
        """
        Build a comprehensive dataset from multiple genres
        
        Args:
            genres: List of genre names
            tracks_per_genre: Tracks to fetch per genre
            
        Returns:
            Combined DataFrame with all tracks
        """
        all_tracks = []
        
        for genre in genres:
            logger.info(f"Fetching tracks for genre: {genre}")
            df_genre = self.fetch_tracks_by_genre(genre, limit=tracks_per_genre)
            if not df_genre.empty:
                df_genre['genre'] = genre
                all_tracks.append(df_genre)
        
        if all_tracks:
            df_combined = pd.concat(all_tracks, ignore_index=True)
            # Remove duplicates
            df_combined = df_combined.drop_duplicates(subset=['id'])
            logger.info(f"✅ Built dataset with {len(df_combined)} unique tracks")
            return df_combined
        else:
            logger.error("Failed to build dataset - no tracks retrieved")
            return pd.DataFrame()
    
    def create_playlist(self, user_id: str, name: str, track_uris: List[str], 
                       description: str = "") -> Optional[str]:
        """
        Create a new playlist for the user
        
        Args:
            user_id: Spotify user ID
            name: Playlist name
            track_uris: List of track URIs to add
            description: Playlist description
            
        Returns:
            Playlist ID if successful
        """
        try:
            playlist = self.sp.user_playlist_create(
                user=user_id,
                name=name,
                public=True,
                description=description
            )
            
            # Add tracks to playlist (max 100 per request)
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i:i+100]
                self.sp.playlist_add_items(playlist['id'], batch)
            
            logger.info(f"✅ Created playlist: {name} with {len(track_uris)} tracks")
            return playlist['id']
            
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None


def test_spotify_connection():
    """
    Test Spotify API connection
    """
    try:
        client = SpotifyClient()
        
        # Test search
        tracks = client.search_tracks("workout motivation", limit=5)
        print(f"✅ Found {len(tracks)} tracks")
        
        if tracks:
            print(f"First track: {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📝 Setup Instructions:")
        print("1. Go to https://developer.spotify.com/dashboard")
        print("2. Create an app")
        print("3. Copy Client ID and Client Secret")
        print("4. Create a .env file with:")
        print("   SPOTIFY_CLIENT_ID=your_client_id")
        print("   SPOTIFY_CLIENT_SECRET=your_client_secret")
        return False


if __name__ == "__main__":
    test_spotify_connection()
