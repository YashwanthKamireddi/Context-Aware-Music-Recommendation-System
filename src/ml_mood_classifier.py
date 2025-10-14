"""
MACHINE LEARNING MOOD CLASSIFIER
Binary classification models for each mood using scikit-learn

This implements proper ML models for mood classification as required for ML assignments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

logger = logging.getLogger(__name__)

class MLMoodClassifier:
    """
    Machine Learning-based mood classifier using trained models for each mood
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize ML mood classifier

        Args:
            models_dir: Directory containing trained models and scalers
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}

        # Define moods
        self.moods = ['workout', 'chill', 'party', 'focus', 'sleep']

        # Load models and scalers for each mood
        self._load_models()

    def _load_models(self):
        """Load trained models, scalers, and feature lists for each mood"""
        for mood in self.moods:
            try:
                # Load model
                model_path = os.path.join(self.models_dir, f"{mood}_model.pkl")
                if os.path.exists(model_path):
                    self.models[mood] = joblib.load(model_path)
                    logger.info(f"Loaded {mood} model from {model_path}")
                else:
                    logger.warning(f"Model not found: {model_path}")
                    self.models[mood] = None

                # Load scaler
                scaler_path = os.path.join(self.models_dir, f"{mood}_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[mood] = joblib.load(scaler_path)
                    logger.info(f"Loaded {mood} scaler from {scaler_path}")
                else:
                    logger.warning(f"Scaler not found: {scaler_path}")
                    self.scalers[mood] = None

                # Load feature columns
                features_path = os.path.join(self.models_dir, f"{mood}_features.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.feature_columns[mood] = json.load(f)
                    logger.info(f"Loaded {mood} features: {self.feature_columns[mood]}")
                else:
                    # Default audio features
                    self.feature_columns[mood] = [
                        'acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
                    ]
                    logger.warning(f"Features not found for {mood}, using defaults")

            except Exception as e:
                logger.error(f"Error loading {mood} model: {e}")
                self.models[mood] = None
                self.scalers[mood] = None

    def predict_mood(self, features: Dict[str, float], mood: str) -> float:
        """
        Predict mood compatibility score using trained ML model

        Args:
            features: Audio features dictionary
            mood: Target mood ('workout', 'chill', 'party', 'focus', 'sleep')

        Returns:
            Probability score (0-1) for the mood
        """
        if mood not in self.models or self.models[mood] is None:
            logger.warning(f"No trained model available for mood: {mood}")
            return 0.5  # Neutral score

        try:
            # Get feature columns for this mood
            feature_cols = self.feature_columns.get(mood, [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
            ])

            # Extract features in correct order
            feature_values = []
            for col in feature_cols:
                value = features.get(col, 0.0)
                # Handle missing values
                if pd.isna(value):
                    value = 0.0
                feature_values.append(value)

            # Convert to numpy array and reshape
            X = np.array(feature_values).reshape(1, -1)

            # Scale features if scaler available
            if self.scalers[mood] is not None:
                # Create DataFrame with feature names for scaler compatibility
                X_df = pd.DataFrame(X, columns=feature_cols)
                X_scaled = self.scalers[mood].transform(X_df)
                X = X_scaled

            # Get probability prediction
            proba = self.models[mood].predict_proba(X)[0]

            # Return probability of positive class (mood compatibility)
            return float(proba[1])

        except Exception as e:
            logger.error(f"Error predicting {mood} score: {e}")
            return 0.5

    def predict_all_moods(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict compatibility scores for all moods

        Args:
            features: Audio features dictionary

        Returns:
            Dictionary with mood scores
        """
        scores = {}
        for mood in self.moods:
            scores[mood] = self.predict_mood(features, mood)
        return scores

    def batch_predict_mood(self, df: pd.DataFrame, mood: str) -> np.ndarray:
        """
        Batch predict mood scores for multiple tracks

        Args:
            df: DataFrame with audio features
            mood: Target mood

        Returns:
            Array of probability scores
        """
        if mood not in self.models or self.models[mood] is None:
            logger.warning(f"No trained model available for mood: {mood}")
            return np.full(len(df), 0.5)

        try:
            # Get feature columns
            feature_cols = self.feature_columns.get(mood, [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
            ])

            # Extract features
            X = df[feature_cols].fillna(0.0).values

            # Scale features if scaler available
            if self.scalers[mood] is not None:
                # Create DataFrame with feature names for scaler compatibility
                X_df = pd.DataFrame(X, columns=feature_cols)
                X_scaled = self.scalers[mood].transform(X_df)
                X = X_scaled

            # Get predictions
            proba = self.models[mood].predict_proba(X)

            # Return probability of positive class
            return proba[:, 1]

        except Exception as e:
            logger.error(f"Error in batch prediction for {mood}: {e}")
            return np.full(len(df), 0.5)


def create_mood_labels(df: pd.DataFrame, mood: str, config: Dict) -> pd.Series:
    """
    Create binary labels for a specific mood based on configuration criteria

    Args:
        df: DataFrame with audio features
        mood: Target mood
        config: Mood configuration with criteria

    Returns:
        Binary labels (0/1) for the mood
    """
    criteria = config['moods'][mood]['criteria']
    labels = pd.Series([True] * len(df), index=df.index)

    # Apply each criterion
    for feature, condition in criteria.items():
        if feature.endswith('_min'):
            base_feature = feature[:-4]  # Remove '_min'
            if base_feature in df.columns:
                labels &= (df[base_feature] >= condition)
        elif feature.endswith('_max'):
            base_feature = feature[:-4]  # Remove '_max'
            if base_feature in df.columns:
                labels &= (df[base_feature] <= condition)

    return labels.astype(int)


def train_mood_model(df: pd.DataFrame, mood: str, config: Dict) -> Tuple[Any, Any, List[str], Dict]:
    """
    Train a machine learning model for a specific mood

    Args:
        df: DataFrame with audio features and mood labels
        mood: Target mood to train
        config: Configuration dictionary

    Returns:
        Tuple of (trained_model, scaler, feature_columns, metrics)
    """
    logger.info(f"Training {mood} model...")

    # Audio features to use
    audio_features = config['features']['audio_features']

    # Create labels based on mood criteria
    labels = create_mood_labels(df, mood, config)

    # Prepare features
    X = df[audio_features].fillna(0.0)
    y = labels

    # Split data
    test_size = config.get('data', {}).get('train_test_split', 0.2)
    random_seed = config.get('data', {}).get('random_seed', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model (Random Forest for better interpretability)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_seed,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'positive_class_ratio': y_test.mean()
    }

    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    metrics['classification_report'] = report

    logger.info(f"{mood} model trained - Accuracy: {metrics['accuracy']:.3f}, "
               f"F1: {metrics['f1_score']:.3f}")

    return model, scaler, audio_features, metrics


def save_trained_model(model: Any, scaler: Any, features: List[str],
                      mood: str, models_dir: str = "models") -> None:
    """
    Save trained model, scaler, and feature list to disk

    Args:
        model: Trained ML model
        scaler: Fitted feature scaler
        features: List of feature names
        mood: Mood name
        models_dir: Directory to save models
    """
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f"{mood}_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved {mood} model to {model_path}")

    # Save scaler
    scaler_path = os.path.join(models_dir, f"{mood}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved {mood} scaler to {scaler_path}")

    # Save features
    features_path = os.path.join(models_dir, f"{mood}_features.json")
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    logger.info(f"Saved {mood} features to {features_path}")


def load_and_train_all_models(data_path: str, config_path: str = "config/config.yaml",
                            models_dir: str = "models") -> Dict[str, Dict]:
    """
    Load data and train ML models for all moods

    Args:
        data_path: Path to CSV data file
        config_path: Path to configuration file
        models_dir: Directory to save trained models

    Returns:
        Dictionary with training metrics for each mood
    """
    from src.utils import load_config

    # Load configuration
    config = load_config(config_path)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Limit data size if specified
    sample_size = config.get('data', {}).get('sample_size')
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=config.get('data', {}).get('random_seed', 42))
        logger.info(f"Sampled {sample_size} tracks from {len(df)} total")

    logger.info(f"Training models on {len(df)} tracks")

    # Train models for each mood
    moods = ['workout', 'chill', 'party', 'focus', 'sleep']
    all_metrics = {}

    for mood in moods:
        try:
            model, scaler, features, metrics = train_mood_model(df, mood, config)
            save_trained_model(model, scaler, features, mood, models_dir)
            all_metrics[mood] = metrics

        except Exception as e:
            logger.error(f"Failed to train {mood} model: {e}")
            all_metrics[mood] = {'error': str(e)}

    # Save overall metrics
    metrics_path = os.path.join(models_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Training complete! Metrics saved to {metrics_path}")
    return all_metrics
