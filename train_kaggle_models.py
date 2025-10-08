"""
Train Real LightGBM Models Using Kaggle Dataset
This creates 5 trained models (one per mood) using ONLY the 9 Spotify features
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The 9 features we have from Spotify/Kaggle
SPOTIFY_FEATURES = [
    'acousticness', 'danceability', 'energy',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]


# Define what makes a track suitable for each mood
MOOD_CRITERIA = {
    'workout': {
        'energy': (0.6, 1.0),        # High energy
        'tempo': (110, 200),          # Fast tempo
        'valence': (0.3, 1.0),        # Positive/energetic
        'danceability': (0.5, 1.0),   # Danceable
    },
    'sleep': {
        'energy': (0.0, 0.4),         # Low energy
        'tempo': (50, 100),            # Slow tempo
        'acousticness': (0.3, 1.0),   # Acoustic
        'instrumentalness': (0.2, 1.0), # Instrumental
    },
    'party': {
        'energy': (0.7, 1.0),         # Very high energy
        'danceability': (0.6, 1.0),   # Very danceable
        'valence': (0.5, 1.0),        # Happy/positive
        'tempo': (110, 180),           # Dance tempo
    },
    'focus': {
        'energy': (0.2, 0.6),         # Medium-low energy
        'instrumentalness': (0.3, 1.0), # Mostly instrumental
        'speechiness': (0.0, 0.3),    # No vocals
        'tempo': (80, 130),            # Moderate tempo
    },
    'chill': {
        'energy': (0.2, 0.6),         # Relaxed
        'valence': (0.3, 0.8),        # Pleasant
        'tempo': (70, 120),            # Slow-moderate
        'danceability': (0.3, 0.7),   # Somewhat groovy
    }
}


def create_labels(df: pd.DataFrame, mood: str) -> np.ndarray:
    """
    Create labels for tracks based on mood criteria
    Returns 1 if track matches mood, 0 otherwise
    """
    criteria = MOOD_CRITERIA[mood]
    labels = np.ones(len(df), dtype=int)

    # For each criterion, check if track meets it
    matches = []
    for feature, (min_val, max_val) in criteria.items():
        if feature in df.columns:
            feature_values = df[feature].values

            # Handle tempo separately (not normalized to 0-1)
            if feature == 'tempo':
                match = (feature_values >= min_val) & (feature_values <= max_val)
            # Handle loudness (negative dB values)
            elif feature == 'loudness':
                match = (feature_values >= min_val) & (feature_values <= max_val)
            # Handle normalized features (0-1)
            else:
                match = (feature_values >= min_val) & (feature_values <= max_val)

            matches.append(match)

    # Track is suitable if it meets at least 60% of criteria
    if matches:
        total_matches = np.sum(matches, axis=0)
        threshold = len(matches) * 0.6
        labels = (total_matches >= threshold).astype(int)

    return labels


def train_mood_model(df: pd.DataFrame, mood: str, output_dir: str = 'models'):
    """
    Train a LightGBM model for a specific mood
    """
    print(f"\n{'='*70}")
    print(f"🎯 Training model for: {mood.upper()}")
    print(f"{'='*70}")

    # Create labels
    print("📝 Creating labels based on mood criteria...")
    y = create_labels(df, mood)

    positive_count = np.sum(y)
    negative_count = len(y) - positive_count
    print(f"   Positive samples: {positive_count:,} ({positive_count/len(y)*100:.1f}%)")
    print(f"   Negative samples: {negative_count:,} ({negative_count/len(y)*100:.1f}%)")

    # Prepare features
    X = df[SPOTIFY_FEATURES].fillna(0).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples")

    # Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n🤖 Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = np.asarray(model.predict(X_test_scaled))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {accuracy:.2%}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Suitable', 'Suitable']))

    # Save model
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f'{mood}_lightgbm.pkl')
    scaler_path = os.path.join(output_dir, f'{mood}_scaler.pkl')
    features_path = os.path.join(output_dir, f'{mood}_features.json')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, 'w') as f:
        json.dump(SPOTIFY_FEATURES, f, indent=2)

    print(f"\n💾 Model saved:")
    print(f"   {model_path}")
    print(f"   {scaler_path}")
    print(f"   {features_path}")

    return model, scaler, accuracy


def main():
    """
    Train all mood models
    """
    print("=" * 70)
    print("🎵 TRAINING LIGHTGBM MODELS FROM KAGGLE DATASET")
    print("=" * 70)

    # Load dataset
    dataset_path = 'data/raw/spotify_tracks.csv'

    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("\nPlease make sure dataset.csv is copied to data/raw/spotify_tracks.csv")
        return

    print(f"\n📁 Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='latin-1')

    # Rename columns if needed
    if 'track_name' in df.columns:
        df.rename(columns={'track_name': 'name'}, inplace=True)
    if 'track_id' in df.columns:
        df.rename(columns={'track_id': 'id'}, inplace=True)

    print(f"✅ Loaded {len(df):,} tracks")

    # Check for required features
    missing_features = [f for f in SPOTIFY_FEATURES if f not in df.columns]
    if missing_features:
        print(f"\n❌ Missing features: {missing_features}")
        return

    print(f"✅ All 9 Spotify features present")

    # Clean data
    print("\n🧹 Cleaning data...")
    before = len(df)
    df = df.dropna(subset=SPOTIFY_FEATURES)
    after = len(df)
    print(f"   Removed {before - after:,} rows with missing features")
    print(f"   Final dataset: {after:,} tracks")

    # Train models for each mood
    moods = ['workout', 'sleep', 'party', 'focus', 'chill']
    results = {}

    for mood in moods:
        model, scaler, accuracy = train_mood_model(df, mood)
        results[mood] = accuracy

    # Summary
    print("\n" + "=" * 70)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📊 Model Accuracies:")
    for mood, accuracy in results.items():
        print(f"   {mood.capitalize():10} → {accuracy:.2%}")

    print("\n✅ Models saved in: models/")
    print("   5 models × 3 files each = 15 files total")
    print("\n🚀 Ready to use! Restart the server to load new models.")


if __name__ == '__main__':
    main()
