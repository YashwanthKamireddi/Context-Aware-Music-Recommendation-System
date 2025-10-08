"""
Recommendation Engine: Real-time mood-based music recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
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
                                   user_profile: Dict[str, float]) -> float:
        """
        Calculate how well a track matches user taste

        Args:
            track: Track features
            user_profile: User profile dictionary

        Returns:
            Match score (0-1)
        """
        key_features = ['energy', 'danceability', 'valence']

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

        return max(0, min(1, similarity))

    def predict_suitability(self, track: pd.Series, mood: str) -> float:
        """
        Predict track suitability using trained model

        Args:
            track: Track features
            mood: Target mood

        Returns:
            Suitability probability
        """
        if mood not in self.models:
            self.load_models(mood)

        # Prepare features
        feature_values = []
        for feature in self.feature_names[mood]:
            if feature in track.index:
                feature_values.append(track[feature])
            else:
                feature_values.append(0.0)

        X = np.array([feature_values])

        # Predict
        probability = self.models[mood].predict(X)[0]

        return probability

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
                   user_profile: Dict[str, float] = None) -> pd.DataFrame:
        """
        Rank candidate tracks for recommendation

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

        weights = self.recommender_config['weights']

        results = []
        selected_tracks = []

        for idx, track in candidates.iterrows():
            # Calculate components
            mood_score = self.calculate_mood_match_score(track, mood)

            if user_profile:
                user_score = self.calculate_user_match_score(track, user_profile)
            else:
                user_score = 0.5  # Neutral

            # REAL Model prediction - THIS IS THE ML MODEL!
            try:
                model_score = self.predict_suitability(track, mood)
            except Exception as e:
                logger.warning(f"Model prediction failed for track: {e}")
                model_score = mood_score  # Fallback

            # Diversity penalty
            diversity_penalty = self.calculate_diversity_penalty(track, selected_tracks)

            # Combined score (WEIGHTED BY REAL ML MODEL)
            final_score = (
                weights['mood_match'] * mood_score +
                weights['user_taste'] * user_score +
                weights['diversity'] * (1 - diversity_penalty)
            )

            # BOOST WITH REAL ML MODEL PREDICTION (30% weight!)
            final_score = 0.7 * final_score + 0.3 * model_score

            # Preserve ALL track data and add scores
            track_dict = track.to_dict()
            track_dict.update({
                'track_id': track.get('track_id', track.get('id', idx)),
                'id': track.get('id', track.get('track_id', '')),
                'name': track.get('name', 'Unknown'),
                'artist': track.get('artist', track.get('artists', 'Unknown')),
                'artists': track.get('artists', track.get('artist', 'Unknown')),
                'album': track.get('album', 'Unknown Album'),
                'album_image': track.get('album_image', ''),
                'album_art': track.get('album_art', track.get('album_image', '')),
                'preview_url': track.get('preview_url', ''),
                'duration_ms': track.get('duration_ms', 0),
                'mood_score': mood_score,
                'user_score': user_score,
                'model_score': model_score,
                'diversity_penalty': diversity_penalty,
                'final_score': final_score
            })
            results.append(track_dict)

            # Add to selected if score is high
            if final_score > self.recommender_config['min_confidence']:
                selected_tracks.append(track)

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False)

        return results_df

    def recommend(self, candidates: pd.DataFrame, mood: str,
                 user_tracks: pd.DataFrame = None, top_k: int = None) -> pd.DataFrame:
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
            top_k = self.recommender_config['top_k']

        # Create user profile
        user_profile = None
        if user_tracks is not None and len(user_tracks) > 0:
            user_profile = self.create_user_profile(user_tracks)
            logger.info(f"Created user profile from {len(user_tracks)} tracks")

        # Rank tracks
        ranked = self.rank_tracks(candidates, mood, user_profile)

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
