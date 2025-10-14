"""
Recommendation Engine: Real-time mood-based music recommendations
Using machine learning models for mood classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

from src.utils import load_config, setup_logging
from src.ml_mood_classifier import MLMoodClassifier


logger = setup_logging()


class MoodRecommender:
    """
    Real-time recommendation engine using machine learning mood classification
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize recommender with ML mood classifier

        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.mood_config = self.config['moods']
        self.recommender_config = self.config['recommender']
        self.audio_features = self.config['features']['audio_features']

        # Initialize ML mood classifier
        self.mood_classifier = MLMoodClassifier(models_dir=self.config['output']['models_dir'])

        self._column_mapping = {
            'track_name': 'name',
            'track_id': 'id',
            'artist_name': 'artists',
            'album_name': 'album'
        }

        logger.info("Initialized ML-based mood recommender")

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
        DEPRECATED: No longer needed with realistic mood classification
        This method is kept for backward compatibility but does nothing.
        """
        logger.info(f"Realistic mood classification - no ML models needed for {mood}")

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
        """
        Predict suitability using realistic mood classification

        Args:
            track: Track features
            mood: Target mood

        Returns:
            Suitability score (0-1) based on mood compatibility
        """
        # Convert track series to feature dictionary
        features = {}
        for feature in self.audio_features:
            if feature in track.index:
                value = track[feature]
                # Handle scalar values properly
                if isinstance(value, (int, float)):
                    features[feature] = float(value)
                else:
                    features[feature] = 0.5  # Default value

        # Get mood scores using ML classification for all moods
        mood_scores = {}
        for mood_name in self.mood_config.keys():
            score = self.mood_classifier.predict_mood(features, mood_name)
            mood_scores[mood_name] = score

        # Return compatibility score for the target mood
        return mood_scores.get(mood, 0.0)

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
        Rank candidate tracks using realistic mood classification

        Args:
            candidates: DataFrame with candidate tracks
            mood: Target mood
            user_profile: Optional user profile

        Returns:
            DataFrame with ranked tracks and scores
        """
        logger.info(f"Ranking {len(candidates)} candidates for {mood} using realistic mood classification...")

        # Calculate mood compatibility scores using batch prediction for efficiency
        mood_scores = self.mood_classifier.batch_predict_mood(candidates, mood)

        # Create results dataframe
        results_df = candidates.copy()
        results_df['mood_score'] = mood_scores

        # Calculate user match score if profile available
        if user_profile:
            user_scores = []
            for idx, track in results_df.iterrows():
                user_score = self.calculate_user_match_score(track, user_profile)
                user_scores.append(user_score)
            results_df['user_score'] = user_scores

            # Combine mood and user scores (weighted average)
            results_df['final_score'] = 0.7 * results_df['mood_score'] + 0.3 * results_df['user_score']
        else:
            results_df['final_score'] = results_df['mood_score']

        # Sort by final score (highest first)
        results_df = results_df.sort_values('final_score', ascending=False)

        # Remove duplicates
        if 'id' in results_df.columns:
            results_df = results_df.drop_duplicates(subset=['id'])
        results_df = results_df.reset_index(drop=True)

        # Ensure required columns exist
        if 'track_id' not in results_df.columns and 'id' in results_df.columns:
            results_df['track_id'] = results_df['id']
        if 'id' not in results_df.columns and 'track_id' in results_df.columns:
            results_df['id'] = results_df['track_id']

        logger.info(f"✅ Realistic mood ranking complete! Score range: {results_df['final_score'].min():.3f} to {results_df['final_score'].max():.3f}")

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

        # Calculate multi-mood compatibility scores for top recommendations
        logger.info("Calculating multi-mood compatibility scores...")
        multi_mood_scores = []
        for _, track in recommendations.iterrows():
            # Extract audio features for this track
            features = {}
            for feature in self.audio_features:
                if feature in track:
                    features[feature] = track[feature]

            # Get compatibility scores for all moods
            mood_scores = self.mood_classifier.predict_all_moods(features)
            multi_mood_scores.append(mood_scores)

        # Add multi-mood scores to recommendations
        recommendations = recommendations.copy()
        for i, mood_name in enumerate(self.mood_classifier.moods):
            recommendations[f'{mood_name}_score'] = [scores[mood_name] for scores in multi_mood_scores]

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
        print("\n✅ Realistic mood-based recommendation test completed!")
        print("   Scores represent mood compatibility (0-1 scale)")
    except Exception as e:
        logger.warning(f"Could not test with models (not trained yet): {e}")
        logger.info("Run full pipeline first to train models")
