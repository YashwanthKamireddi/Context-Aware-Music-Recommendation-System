"""
Test Spotify API Connection
Run this to verify your credentials work
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.spotify_client import SpotifyClient

def main():
    print("=" * 60)
    print("🎵 SPOTIFY API CONNECTION TEST")
    print("=" * 60)
    print()
    
    try:
        print("📡 Initializing Spotify client...")
        client = SpotifyClient()
        print("✅ Client initialized successfully!")
        print()
        
        print("🔍 Testing search...")
        tracks = client.search_tracks("workout motivation", limit=5)
        
        if tracks:
            print(f"✅ Found {len(tracks)} tracks!")
            print("\n📀 Sample tracks:")
            for i, track in enumerate(tracks[:3], 1):
                print(f"  {i}. {track['name']} - {track['artists'][0]['name']}")
            print()
        
        print("🎵 Testing audio features...")
        track_ids = [t['id'] for t in tracks[:3]]
        features = client.get_audio_features(track_ids)
        
        if features:
            print(f"✅ Retrieved audio features for {len(features)} tracks!")
            print(f"\n📊 Sample features from '{tracks[0]['name']}':")
            feat = features[0]
            print(f"  Energy: {feat['energy']:.2f}")
            print(f"  Danceability: {feat['danceability']:.2f}")
            print(f"  Valence: {feat['valence']:.2f}")
            print(f"  Tempo: {feat['tempo']:.0f} BPM")
            print()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Your Spotify API is working perfectly!")
        print("=" * 60)
        print()
        print("✅ Ready to start the backend server:")
        print("   python backend/server.py")
        print()
        
        return True
        
    except ValueError as e:
        print("❌ ERROR: " + str(e))
        print()
        print("📝 SETUP INSTRUCTIONS:")
        print("=" * 60)
        print("1. Go to: https://developer.spotify.com/dashboard")
        print("2. Create a new app")
        print("3. Copy your Client ID and Client Secret")
        print("4. Create a .env file in the project root with:")
        print()
        print("   SPOTIFY_CLIENT_ID=your_client_id_here")
        print("   SPOTIFY_CLIENT_SECRET=your_client_secret_here")
        print("   SPOTIFY_REDIRECT_URI=http://localhost:8000/callback")
        print()
        print("5. Run this test again!")
        print("=" * 60)
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
