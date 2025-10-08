"""
Load Spotify Dataset from Kaggle CSV
This bypasses Spotify API completely - perfect for rate limit issues!
"""

import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_kaggle_spotify_data(csv_path: str) -> pd.DataFrame:
    """
    Load Kaggle Spotify dataset and standardize columns

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with standardized columns
    """
    print(f"📁 Loading Kaggle dataset from: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} tracks")

    # Print available columns
    print(f"\n📊 Available columns: {list(df.columns)}")

    # Column mapping (handles different naming conventions)
    column_mapping = {
        # Track info
        'track_name': 'name',
        'track_id': 'id',
        'artist_name': 'artists',
        'artist': 'artists',
        'track_artist': 'artists',

        # Audio features (these should match exactly)
        'acousticness': 'acousticness',
        'danceability': 'danceability',
        'energy': 'energy',
        'instrumentalness': 'instrumentalness',
        'liveness': 'liveness',
        'loudness': 'loudness',
        'speechiness': 'speechiness',
        'tempo': 'tempo',
        'valence': 'valence',
    }

    # Rename columns
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            df.rename(columns={old_name: new_name}, inplace=True)
            print(f"  📝 Renamed '{old_name}' → '{new_name}'")

    # Required columns
    required_features = [
        'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness',
        'speechiness', 'tempo', 'valence'
    ]

    required_metadata = ['name', 'artists']

    # Check for required columns
    missing_features = [col for col in required_features if col not in df.columns]
    missing_metadata = [col for col in required_metadata if col not in df.columns]

    if missing_features:
        print(f"\n❌ Missing audio features: {missing_features}")
        print(f"   Available columns: {list(df.columns)}")
        raise ValueError(f"Dataset missing required audio features: {missing_features}")

    if missing_metadata:
        print(f"\n⚠️ Warning: Missing metadata: {missing_metadata}")
        # Create dummy metadata if missing
        if 'name' not in df.columns:
            df['name'] = df.index.map(lambda x: f"Track {x}")
        if 'artists' not in df.columns:
            df['artists'] = "Unknown Artist"

    # Handle 'id' column (optional)
    if 'id' not in df.columns:
        print("  ⚠️ No 'id' column, creating sequential IDs")
        df['id'] = df.index.map(lambda x: f"track_{x}")

    # Clean data
    # Remove rows with missing audio features
    before = len(df)
    df = df.dropna(subset=required_features)
    after = len(df)
    if before != after:
        print(f"  🧹 Removed {before - after} rows with missing features")

    # Normalize audio features to 0-1 range (if needed)
    for feature in ['acousticness', 'danceability', 'energy', 'instrumentalness',
                   'liveness', 'speechiness', 'valence']:
        if feature in df.columns:
            # Clip to 0-1 range
            df[feature] = df[feature].clip(0, 1)

    # Normalize loudness (typically -60 to 0 dB)
    if 'loudness' in df.columns:
        df['loudness'] = df['loudness'].clip(-60, 0)

    # Normalize tempo (typically 30-250 BPM)
    if 'tempo' in df.columns:
        df['tempo'] = df['tempo'].clip(30, 250)

    print(f"\n✅ Final dataset: {len(df)} tracks with {len(df.columns)} columns")
    print(f"   Audio features: {required_features}")
    print(f"   Metadata: name, artists, id")

    return df


def find_kaggle_dataset():
    """
    Search for Kaggle dataset in common locations
    """
    possible_paths = [
        'data/raw/spotify_tracks.csv',
        'data/raw/spotify_songs.csv',
        'data/raw/tracks.csv',
        'data/raw/spotify_dataset.csv',
        'data/spotify_tracks.csv',
        'spotify_tracks.csv',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


if __name__ == '__main__':
    print("=" * 70)
    print("🎵 KAGGLE SPOTIFY DATASET LOADER")
    print("=" * 70)

    # Try to find dataset
    csv_path = find_kaggle_dataset()

    if csv_path:
        print(f"\n✅ Found dataset: {csv_path}")
        try:
            df = load_kaggle_spotify_data(csv_path)

            # Show sample
            print("\n📋 Sample tracks:")
            print(df[['name', 'artists', 'energy', 'valence', 'tempo']].head())

            # Show statistics
            print("\n📊 Audio Feature Statistics:")
            print(df[['energy', 'valence', 'danceability', 'tempo']].describe())

            print("\n✅ Dataset loaded successfully!")
            print(f"   Ready to use with {len(df)} tracks")

        except Exception as e:
            print(f"\n❌ Error loading dataset: {e}")
    else:
        print("\n❌ No Kaggle dataset found!")
        print("\nPlease download a Spotify dataset from Kaggle and place it in:")
        print("  - data/raw/spotify_tracks.csv")
        print("\nRecommended datasets:")
        print("  1. https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets")
        print("  2. https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db")
        print("  3. Search Kaggle for 'spotify audio features'")
