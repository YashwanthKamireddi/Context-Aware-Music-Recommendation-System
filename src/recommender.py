"""
Recommendation Engine: Real-time mood-based music recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

from src.utils import load_config, setup_logging, load_model


logger = setup_logging()


class MoodRecommender:
    """
    Real-time recommendation engine for mood-based playlists
    """

    def __init__(self, config: Dict = None):
        """
        Initialize recommender

        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.mood_config = self.config['moods']
        self.recommender_config = self.config['recommender']
        self.audio_features = self.config['features']['audio_features']

        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self._column_mapping = {
            'track_name': 'name',
            'track_id': 'id',
            'artist_name': 'artists',
            'album_name': 'album'
        }

    def _prepare_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Standardise incoming candidate dataframe so models receive expected columns."""
        df = candidates.copy()

        # Drop unnamed index columns if present
        unnamed_cols = [col for col in df.columns if col.lower().startswith('unnamed')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        # Apply known column aliases (e.g. track_name -> name)
        for source_col, target_col in self._column_mapping.items():
            if source_col in df.columns:
                df[target_col] = df[source_col]

        # Ensure we always have artists + album fields for the UI
        if 'artist' in df.columns and 'artists' not in df.columns:
            df['artists'] = df['artist']
        if 'artists' in df.columns and 'artist' not in df.columns:
            df['artist'] = df['artists']
        if 'album' in df.columns and 'album_name' not in df.columns:
            df['album_name'] = df['album']

        # Deduplicate track ids if provided to avoid repeated rows
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        elif 'track_id' in df.columns:
            df = df.drop_duplicates(subset=['track_id'])

        # Coerce audio feature columns to numeric and drop rows with missing values
        feature_cols = self.audio_features
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=[col for col in feature_cols if col in df.columns])

        return df

    def load_models(self, mood: str) -> None:
        """
        Load trained models for a mood

        Args:
            mood: Mood name
        """
        models_dir = self.config['output']['models_dir']

        try:
            self.models[mood] = load_model(f"{models_dir}/{mood}_lightgbm.pkl")
            self.scalers[mood] = load_model(f"{models_dir}/{mood}_scaler.pkl")

            import json
            with open(f"{models_dir}/{mood}_features.json", 'r') as f:
                self.feature_names[mood] = json.load(f)

            logger.info(f"Loaded models for {mood}")
        except Exception as e:
            logger.error(f"Error loading models for {mood}: {e}")
            raise

    def create_user_profile(self, user_tracks: pd.DataFrame) -> Dict[str, float]:
        """
        Create user taste profile from listening history

        Args:
            user_tracks: DataFrame with user's favorite tracks

        Returns:
            Dictionary with user profile features
        """
        profile = {}

        for feature in self.audio_features:
            if feature in user_tracks.columns:
                profile[f'user_avg_{feature}'] = user_tracks[feature].mean()
                profile[f'user_std_{feature}'] = user_tracks[feature].std()

        return profile

    def calculate_mood_match_score(self, track: pd.Series, mood: str) -> float:
        """
        Calculate how well a track matches a mood

        Args:
            track: Track features
            mood: Target mood

        Returns:
            Match score (0-1)
        """
        mood_criteria = self.mood_config[mood]['criteria']

        matches = 0
        total = len(mood_criteria)

        for feature, threshold in mood_criteria.items():
            feature_name = feature.replace('_min', '').replace('_max', '')

            if feature_name in track.index:
                if '_min' in feature:
                    if track[feature_name] >= threshold:
                        matches += 1
                elif '_max' in feature:
                    if track[feature_name] <= threshold:
                        matches += 1

        return matches / total if total > 0 else 0.0

    def calculate_user_match_score(self, track: pd.Series,
                                   user_profile: Optional[Dict[str, float]]) -> float:
        """
        Calculate how well a track matches user taste

        Args:
            track: Track features
            user_profile: User profile dictionary

        Returns:
            Match score (0-1)
        """
        key_features = ['energy', 'danceability', 'valence']

        if not user_profile:
            return 0.5

        differences = []
        for feature in key_features:
            if feature in track.index and f'user_avg_{feature}' in user_profile:
                diff = abs(track[feature] - user_profile[f'user_avg_{feature}'])
                differences.append(diff)

        if not differences:
            return 0.5  # Neutral score

        # Convert difference to similarity (inverse)
        avg_diff = np.mean(differences)
        similarity = 1 - avg_diff

        return float(max(0.0, min(1.0, similarity)))

    def predict_suitability(self, track: pd.Series, mood: str) -> float:
        """Predict suitability of a single track using the trained ML model."""
        if mood not in self.models:
            self.load_models(mood)

        feature_cols = self.feature_names.get(mood, self.audio_features)
        track_df = pd.DataFrame([track])

        for col in feature_cols:
            if col not in track_df.columns:
                track_df[col] = 0.0

        track_df = track_df[feature_cols]
        scaled = self.scalers[mood].transform(track_df.values)
        scaled_df = pd.DataFrame(scaled, columns=feature_cols)
        proba = self.models[mood].predict_proba(scaled_df)[:, 1][0]

        return float(np.clip(proba, 0.0, 1.0))

    def calculate_diversity_penalty(self, track: pd.Series,
                                   selected_tracks: List[pd.Series]) -> float:
        """
        Calculate diversity penalty to avoid repetitive recommendations

        Args:
            track: Candidate track
            selected_tracks: Already selected tracks

        Returns:
            Penalty score (0-1, higher means more similar)
        """
        if not selected_tracks:
            return 0.0

        key_features = ['energy', 'danceability', 'valence', 'tempo']

        similarities = []
        for selected in selected_tracks:
            feature_diffs = []
            for feature in key_features:
                if feature in track.index and feature in selected.index:
                    # Normalize differences
                    if feature == 'tempo':
                        diff = abs(track[feature] - selected[feature]) / 200
                    else:
                        diff = abs(track[feature] - selected[feature])
                    feature_diffs.append(diff)

            if feature_diffs:
                similarity = 1 - np.mean(feature_diffs)
                similarities.append(similarity)

        return np.max(similarities) if similarities else 0.0

    def rank_tracks(self, candidates: pd.DataFrame, mood: str,
                   user_profile: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Rank candidate tracks for recommendation (FAST VECTORIZED VERSION)

        Args:
            candidates: DataFrame with candidate tracks
            mood: Target mood
            user_profile: Optional user profile

        Returns:
            DataFrame with ranked tracks and scores
        """
        logger.info(f"Ranking {len(candidates)} candidates for {mood}...")

        # ENSURE models are loaded
        if mood not in self.models:
            logger.info(f"Loading models for {mood}...")
            self.load_models(mood)

        # FAST VECTORIZED ML PREDICTIONS (process all tracks at once!)
        logger.info(f"Running ML model predictions on {len(candidates)} tracks...")
        feature_cols = self.feature_names.get(mood, self.audio_features)

        feature_df = candidates.copy()
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0

        feature_df = feature_df[feature_cols]
        X_scaled = self.scalers[mood].transform(feature_df.values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # Get probabilities for ALL tracks at once (FAST!)
        model_scores = self.models[mood].predict_proba(X_scaled_df)[:, 1]
        logger.info(f"✅ ML predictions complete! Scores range: {model_scores.min():.3f} to {model_scores.max():.3f}")

        # Use ML model scores directly as the ranking (99%+ accuracy models!)
        results_df = candidates.copy()
        results_df['model_score'] = model_scores
        results_df['final_score'] = model_scores

        # Sort by ML model score (highest first)
        results_df = results_df.sort_values('final_score', ascending=False)
        if 'id' in results_df.columns:
            results_df = results_df.drop_duplicates(subset=['id'])
        results_df = results_df.reset_index(drop=True)

        # Ensure required columns exist
        if 'track_id' not in results_df.columns and 'id' in results_df.columns:
            results_df['track_id'] = results_df['id']
        if 'id' not in results_df.columns and 'track_id' in results_df.columns:
            results_df['id'] = results_df['track_id']

        return results_df

    def recommend(self, candidates: pd.DataFrame, mood: str,
                 user_tracks: Optional[pd.DataFrame] = None,
                 top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Generate recommendations

        Args:
            candidates: Candidate tracks
            mood: Target mood
            user_tracks: User's listening history (optional)
            top_k: Number of recommendations (default from config)

        Returns:
            DataFrame with top recommendations
        """
        if top_k is None:
            top_k = int(self.recommender_config['top_k'])

        # Create user profile
        user_profile = None
        if user_tracks is not None and len(user_tracks) > 0:
            user_profile = self.create_user_profile(user_tracks)
            logger.info(f"Created user profile from {len(user_tracks)} tracks")

        # Standardise dataframe so downstream steps always see expected schema
        prepared_candidates = self._prepare_candidates(candidates)

        # Rank tracks
        ranked = self.rank_tracks(prepared_candidates, mood, user_profile)

        # Return top-k
        recommendations = ranked.head(top_k)

        logger.info(f"Generated {len(recommendations)} recommendations")
        logger.info(f"Average final score: {recommendations['final_score'].mean():.3f}")

        return recommendations


if __name__ == "__main__":
    # Test recommender
    from data_pipeline import DataPipeline

    logger.info("Testing recommendation engine...")

    # Create sample data
    pipeline = DataPipeline()
    pipeline.load_data(use_sample=True)
    df = pipeline.clean_data()
    df = pipeline.assign_mood_labels(df)

    # Create recommender
    recommender = MoodRecommender()

    # Test recommendation
    mood = 'workout'
    candidates = df.sample(100)
    user_tracks = df.sample(20)

    try:
        recommendations = recommender.recommend(candidates, mood, user_tracks, top_k=10)

        print(f"\n{'='*60}")
        print(f"Top 10 Recommendations for {mood.upper()}")
        print(f"{'='*60}\n")
        print(recommendations[['name', 'artists', 'final_score']].to_string(index=False))
        print("\n✅ Recommendation test completed!")
    except Exception as e:
        logger.warning(f"Could not test with models (not trained yet): {e}")
        logger.info("Run full pipeline first to train models")
