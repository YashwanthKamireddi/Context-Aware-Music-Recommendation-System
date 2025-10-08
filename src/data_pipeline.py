"""
Data Pipeline: Loading and preprocessing Spotify track data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.model_selection import train_test_split

from src.utils import load_config, ensure_dir, setup_logging, create_sample_dataset


logger = setup_logging()


class DataPipeline:
    """
    Data loading and preprocessing pipeline
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.data_config = self.config['data']
        self.mood_config = self.config['moods']
        self.features = self.config['features']['audio_features']
        
        self.df_raw = None
        self.df_processed = None
        
    def load_data(self, use_sample: bool = False) -> pd.DataFrame:
        """
        Load Spotify track dataset
        
        Args:
            use_sample: If True, create synthetic data for testing
        
        Returns:
            DataFrame with track data
        """
        logger.info("Loading data...")
        
        if use_sample:
            logger.info("Creating sample dataset for testing...")
            self.df_raw = create_sample_dataset(
                n_samples=self.data_config.get('sample_size', 10000)
            )
        else:
            # Load from CSV (Kaggle dataset)
            try:
                dataset_path = self.data_config['dataset_path']
                self.df_raw = pd.read_csv(dataset_path)
                logger.info(f"Loaded {len(self.df_raw)} tracks from {dataset_path}")
                
                # Sample if specified
                if self.data_config['sample_size']:
                    self.df_raw = self.df_raw.sample(
                        n=min(self.data_config['sample_size'], len(self.df_raw)),
                        random_state=self.data_config['random_seed']
                    )
                    logger.info(f"Sampled to {len(self.df_raw)} tracks")
            except FileNotFoundError:
                logger.warning(f"Dataset not found at {dataset_path}. Using sample data.")
                self.df_raw = create_sample_dataset(10000)
        
        return self.df_raw
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and validate data
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        df = self.df_raw.copy()
        
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['track_id'] if 'track_id' in df.columns else None)
        logger.info(f"Removed {initial_size - len(df)} duplicate tracks")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill missing audio features with median
        for feature in self.features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        # Fill missing metadata with placeholder
        if 'name' in df.columns:
            df['name'] = df['name'].fillna('Unknown Track')
        if 'artists' in df.columns:
            df['artists'] = df['artists'].fillna('Unknown Artist')
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        # Remove rows with invalid feature values (outside expected range)
        for feature in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'liveness', 'speechiness', 'valence']:
            if feature in df.columns:
                df = df[(df[feature] >= 0) & (df[feature] <= 1)]
        
        if 'tempo' in df.columns:
            df = df[(df['tempo'] >= 30) & (df['tempo'] <= 250)]
        
        logger.info(f"Final dataset size: {len(df)} tracks")
        
        return df
    
    def assign_mood_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign mood labels to tracks based on audio features
        
        Args:
            df: DataFrame with audio features
        
        Returns:
            DataFrame with mood labels
        """
        logger.info("Assigning mood labels...")
        
        df = df.copy()
        
        # Initialize mood columns
        for mood_name in self.mood_config.keys():
            df[f'is_{mood_name}'] = 0
        
        # Assign labels based on criteria
        for mood_name, mood_info in self.mood_config.items():
            criteria = mood_info['criteria']
            mask = pd.Series([True] * len(df), index=df.index)
            
            for feature, threshold in criteria.items():
                if '_min' in feature:
                    feature_name = feature.replace('_min', '')
                    if feature_name in df.columns:
                        mask &= (df[feature_name] >= threshold)
                elif '_max' in feature:
                    feature_name = feature.replace('_max', '')
                    if feature_name in df.columns:
                        mask &= (df[feature_name] <= threshold)
            
            df.loc[mask, f'is_{mood_name}'] = 1
            count = mask.sum()
            logger.info(f"  {mood_info['emoji']} {mood_info['name']}: {count} tracks")
        
        return df
    
    def create_training_data(self, df: pd.DataFrame, mood: str, 
                            negative_ratio: float = 1.0) -> pd.DataFrame:
        """
        Create training dataset for a specific mood
        
        Args:
            df: DataFrame with mood labels
            mood: Target mood
            negative_ratio: Ratio of negative to positive samples
        
        Returns:
            Training DataFrame with balanced samples
        """
        logger.info(f"Creating training data for {mood}...")
        
        # Positive samples (tracks that match the mood)
        positive_samples = df[df[f'is_{mood}'] == 1].copy()
        positive_samples['target'] = 1
        
        # Negative samples (tracks that don't match the mood)
        negative_samples = df[df[f'is_{mood}'] == 0].copy()
        
        # Balance dataset
        n_negative = int(len(positive_samples) * negative_ratio)
        if n_negative < len(negative_samples):
            negative_samples = negative_samples.sample(
                n=n_negative, 
                random_state=self.data_config['random_seed']
            )
        
        negative_samples['target'] = 0
        
        # Combine
        training_data = pd.concat([positive_samples, negative_samples], ignore_index=True)
        training_data = training_data.sample(frac=1, random_state=self.data_config['random_seed'])
        
        logger.info(f"  Positive samples: {len(positive_samples)}")
        logger.info(f"  Negative samples: {len(negative_samples)}")
        logger.info(f"  Total training samples: {len(training_data)}")
        
        return training_data
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame to split
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df,
            test_size=1 - self.data_config['train_test_split'],
            random_state=self.data_config['random_seed'],
            stratify=df['target'] if 'target' in df.columns else None
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to disk
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.data_config['processed_path']
        ensure_dir(output_path)
        
        filepath = f"{output_path}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def run_pipeline(self, use_sample: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run complete data pipeline
        
        Args:
            use_sample: Use synthetic sample data
        
        Returns:
            Dictionary with processed datasets for each mood
        """
        logger.info("🚀 Starting data pipeline...")
        
        # Load and clean data
        self.load_data(use_sample=use_sample)
        df_clean = self.clean_data()
        
        # Assign mood labels
        df_labeled = self.assign_mood_labels(df_clean)
        
        # Save labeled data
        self.save_processed_data(df_labeled, "tracks_labeled.csv")
        
        # Create training datasets for each mood
        datasets = {}
        for mood_name in self.mood_config.keys():
            training_data = self.create_training_data(df_labeled, mood_name)
            train_df, test_df = self.split_data(training_data)
            
            datasets[mood_name] = {
                'train': train_df,
                'test': test_df,
                'full': training_data
            }
            
            # Save
            self.save_processed_data(train_df, f"{mood_name}_train.csv")
            self.save_processed_data(test_df, f"{mood_name}_test.csv")
        
        logger.info("✅ Data pipeline completed successfully!")
        
        return datasets


if __name__ == "__main__":
    # Test data pipeline
    pipeline = DataPipeline()
    datasets = pipeline.run_pipeline(use_sample=True)
    
    print("\n📊 Dataset Summary:")
    for mood, data in datasets.items():
        print(f"\n{mood.upper()}:")
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Test: {len(data['test'])} samples")
