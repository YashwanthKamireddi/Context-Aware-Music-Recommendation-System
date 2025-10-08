"""
Utility functions for Vibe-Sync
"""

import os
import yaml
import logging
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist

    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk using pickle

    Args:
        model: Model object to save
        filepath: Path to save model
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load model from disk

    Args:
        filepath: Path to model file

    Returns:
        Loaded model object
    """
    import joblib
    model = joblib.load(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save JSON
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified columns to 0-1 range

    Args:
        df: DataFrame
        columns: List of column names to normalize

    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    return df_normalized


def print_section(title: str, emoji: str = "🎵") -> None:
    """
    Print formatted section header

    Args:
        title: Section title
        emoji: Emoji to display
    """
    print(f"\n{emoji} {'='*60}")
    print(f"{emoji} {title}")
    print(f"{emoji} {'='*60}\n")


def format_track_info(track: Dict) -> str:
    """
    Format track information for display

    Args:
        track: Track dictionary

    Returns:
        Formatted string
    """
    return f"{track.get('name', 'Unknown')} - {track.get('artists', 'Unknown Artist')}"


def calculate_diversity(recommendations: List[Dict], feature_cols: List[str]) -> float:
    """
    Calculate diversity score of recommendations

    Args:
        recommendations: List of recommended tracks
        feature_cols: Features to consider for diversity

    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if len(recommendations) < 2:
        return 1.0

    # Extract feature vectors
    features = []
    for track in recommendations:
        feature_vec = [track.get(col, 0) for col in feature_cols]
        features.append(feature_vec)

    features = np.array(features)

    # Calculate pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(features, metric='euclidean')

    # Return average distance as diversity score
    return np.mean(distances)


def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic Spotify-like dataset for testing

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic track data
    """
    np.random.seed(42)

    data = {
        'track_id': [f'track_{i}' for i in range(n_samples)],
        'name': [f'Song {i}' for i in range(n_samples)],
        'artists': [f'Artist {i % 100}' for i in range(n_samples)],
        'acousticness': np.random.random(n_samples),
        'danceability': np.random.random(n_samples),
        'energy': np.random.random(n_samples),
        'instrumentalness': np.random.random(n_samples),
        'liveness': np.random.random(n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),
        'speechiness': np.random.random(n_samples),
        'tempo': np.random.uniform(60, 200, n_samples),
        'valence': np.random.random(n_samples),
        'popularity': np.random.randint(0, 100, n_samples)
    }

    return pd.DataFrame(data)


def get_project_root() -> str:
    """
    Get project root directory

    Returns:
        Project root path
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("Utilities module loaded successfully")

    # Test config loading
    try:
        config = load_config()
        print("✅ Config loaded successfully")
        print(f"Moods defined: {list(config['moods'].keys())}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
