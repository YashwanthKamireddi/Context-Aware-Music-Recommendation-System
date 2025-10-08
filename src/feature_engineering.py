"""
Feature Engineering: Creating features for mood-based recommendation
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

from src.utils import load_config, setup_logging


logger = setup_logging()


class FeatureEngineer:
    """
    Feature engineering for recommendation model
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.audio_features = self.config['features']['audio_features']
        self.mood_config = self.config['moods']
        
    def calculate_mood_profile(self, df: pd.DataFrame, mood: str) -> Dict[str, float]:
        """
        Calculate average audio features for a mood
        
        Args:
            df: DataFrame with mood labels
            mood: Mood name
        
        Returns:
            Dictionary with average feature values
        """
        mood_tracks = df[df[f'is_{mood}'] == 1]
        
        profile = {}
        for feature in self.audio_features:
            if feature in mood_tracks.columns:
                profile[f'{mood}_avg_{feature}'] = mood_tracks[feature].mean()
                profile[f'{mood}_std_{feature}'] = mood_tracks[feature].std()
        
        return profile
    
    def add_interaction_features(self, df: pd.DataFrame, mood: str, 
                                 mood_profile: Dict[str, float]) -> pd.DataFrame:
        """
        Add interaction features (difference/ratio between track and mood profile)
        
        Args:
            df: DataFrame with tracks
            mood: Target mood
            mood_profile: Mood profile dictionary
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Key features for interaction
        key_features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness']
        
        for feature in key_features:
            if feature in df.columns:
                avg_key = f'{mood}_avg_{feature}'
                if avg_key in mood_profile:
                    # Difference
                    df[f'{feature}_diff'] = df[feature] - mood_profile[avg_key]
                    
                    # Absolute difference
                    df[f'{feature}_abs_diff'] = np.abs(df[f'{feature}_diff'])
                    
                    # Ratio (avoid division by zero)
                    if mood_profile[avg_key] != 0:
                        df[f'{feature}_ratio'] = df[feature] / mood_profile[avg_key]
                    else:
                        df[f'{feature}_ratio'] = 1.0
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical aggregation features
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        # Create feature combinations
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_danceability'] = df['energy'] * df['danceability']
        
        if 'energy' in df.columns and 'valence' in df.columns:
            df['energy_valence'] = df['energy'] * df['valence']
        
        if 'acousticness' in df.columns and 'energy' in df.columns:
            df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 0.01)
        
        # Tempo category
        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(df['tempo'], 
                                         bins=[0, 90, 120, 150, 300],
                                         labels=['slow', 'moderate', 'fast', 'very_fast'])
            df['tempo_category'] = df['tempo_category'].cat.codes
        
        # Energy category
        if 'energy' in df.columns:
            df['energy_category'] = pd.cut(df['energy'],
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=['low', 'medium', 'high'])
            df['energy_category'] = df['energy_category'].cat.codes
        
        return df
    
    def create_feature_matrix(self, df: pd.DataFrame, mood: str) -> pd.DataFrame:
        """
        Create complete feature matrix for modeling
        
        Args:
            df: DataFrame with tracks
            mood: Target mood
        
        Returns:
            DataFrame with all features
        """
        logger.info(f"Creating feature matrix for {mood}...")
        
        # Calculate mood profile
        mood_profile = self.calculate_mood_profile(df, mood)
        logger.info(f"  Calculated {mood} profile with {len(mood_profile)} features")
        
        # Add mood profile as features
        for key, value in mood_profile.items():
            df[key] = value
        
        # Add interaction features
        df = self.add_interaction_features(df, mood, mood_profile)
        logger.info(f"  Added interaction features")
        
        # Add statistical features
        df = self.add_statistical_features(df)
        logger.info(f"  Added statistical features")
        
        logger.info(f"  Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_names(self, mood: str = None) -> List[str]:
        """
        Get list of feature names for modeling
        
        Args:
            mood: Optional mood name for mood-specific features
        
        Returns:
            List of feature names
        """
        features = self.audio_features.copy()
        
        # Add interaction features
        key_features = ['energy', 'danceability', 'valence', 'tempo', 'acousticness']
        for feature in key_features:
            features.extend([
                f'{feature}_diff',
                f'{feature}_abs_diff',
                f'{feature}_ratio'
            ])
        
        # Add statistical features
        features.extend([
            'energy_danceability',
            'energy_valence',
            'acoustic_energy_ratio',
            'tempo_category',
            'energy_category'
        ])
        
        # Add mood profile features
        if mood:
            for audio_feature in self.audio_features:
                features.extend([
                    f'{mood}_avg_{audio_feature}',
                    f'{mood}_std_{audio_feature}'
                ])
        
        return features
    
    def select_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Select specific features from DataFrame
        
        Args:
            df: DataFrame
            feature_list: List of features to select
        
        Returns:
            DataFrame with selected features
        """
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = set(feature_list) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        return df[available_features]


def engineer_features_for_mood(df: pd.DataFrame, mood: str, 
                               config: Dict = None) -> pd.DataFrame:
    """
    Convenience function to engineer features for a specific mood
    
    Args:
        df: Input DataFrame
        mood: Target mood
        config: Configuration dictionary
    
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(config)
    return engineer.create_feature_matrix(df, mood)


if __name__ == "__main__":
    # Test feature engineering
    from data_pipeline import DataPipeline
    
    logger.info("Testing feature engineering...")
    
    # Create sample data
    pipeline = DataPipeline()
    pipeline.load_data(use_sample=True)
    df = pipeline.clean_data()
    df = pipeline.assign_mood_labels(df)
    
    # Test feature engineering
    engineer = FeatureEngineer()
    
    for mood in ['workout', 'chill', 'party']:
        print(f"\n{'='*60}")
        print(f"Testing {mood.upper()} features")
        print(f"{'='*60}")
        
        df_features = engineer.create_feature_matrix(df, mood)
        feature_names = engineer.get_feature_names(mood)
        
        print(f"Total features: {len(feature_names)}")
        print(f"DataFrame shape: {df_features.shape}")
        print(f"\nSample features:")
        print(df_features[feature_names[:10]].head())
