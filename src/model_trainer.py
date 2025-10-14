"""
Model Training: Train and optimize ML models for mood classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.utils import load_config, setup_logging, save_model, ensure_dir


logger = setup_logging()


class ModelTrainer:
    """
    Train and optimize models for mood-based classification
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.model_config = self.config['model']
        self.eval_config = self.config['evaluation']

        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame,
                    feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names

        Returns:
            Tuple of (X, y)
        """
        # Select features that exist in DataFrame
        available_features = [col for col in feature_cols if col in df.columns]

        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        X = df[available_features].values.astype(np.float64)
        y = df['target'].values.astype(np.int64)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        self.feature_names = available_features

        return X, y

    def train_baseline(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train baseline logistic regression model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        logger.info("Training baseline Logistic Regression...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        params = self.model_config['baseline']['params']
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Cross-validation score
        if self.eval_config['cross_validation']['enabled']:
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=self.eval_config['cross_validation']['folds'],
                scoring='f1'
            )
            logger.info(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """
        Train LightGBM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained model
        """
        logger.info("Training LightGBM model...")

        params = self.model_config['main']['params'].copy()

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=params.get('n_estimators', 100),
            callbacks=[lgb.log_evaluation(period=10)]
        )

        logger.info("  ✅ LightGBM training completed")

        return model

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Best parameters dictionary
        """
        if not self.model_config['hyperparameter_tuning']['enabled']:
            logger.info("Hyperparameter tuning disabled")
            return self.model_config['main']['params']

        logger.info("Performing hyperparameter tuning...")

        param_grid = self.model_config['hyperparameter_tuning']['param_grid']

        # Create LightGBM classifier
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)

        # Grid search
        grid_search = GridSearchCV(
            model,  # type: ignore
            param_grid,
            cv=self.model_config['hyperparameter_tuning']['cv_folds'],
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"  Best F1-Score: {grid_search.best_score_:.4f}")
        logger.info(f"  Best parameters: {grid_search.best_params_}")

        return grid_search.best_params_

    def train_models_for_mood(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                             feature_cols: List[str], mood: str) -> Dict[str, Any]:
        """
        Train all models for a specific mood

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            feature_cols: Feature column names
            mood: Mood name

        Returns:
            Dictionary with trained models
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for {mood.upper()}")
        logger.info(f"{'='*60}\n")

        # Prepare data
        X_train, y_train = self.prepare_data(train_df, feature_cols)
        X_val, y_val = self.prepare_data(val_df, feature_cols)

        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)\n")

        models = {}

        # Train baseline
        models['baseline'] = self.train_baseline(X_train, y_train)

        # Hyperparameter tuning (optional)
        best_params = self.hyperparameter_tuning(X_train, y_train)

        # Update config with best params
        self.model_config['main']['params'].update(best_params)

        # Train main model
        models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)

        # Store scaler and feature names
        models['scaler'] = self.scaler
        models['feature_names'] = self.feature_names

        return models

    def save_models(self, models: Dict[str, Any], mood: str) -> None:
        """
        Save trained models to disk

        Args:
            models: Dictionary with models
            mood: Mood name
        """
        output_dir = self.config['output']['models_dir']
        ensure_dir(output_dir)

        for model_name, model in models.items():
            if model_name in ['baseline', 'lightgbm', 'scaler']:
                filepath = f"{output_dir}/{mood}_{model_name}.pkl"
                save_model(model, filepath)

        # Save feature names
        import json
        with open(f"{output_dir}/{mood}_features.json", 'w') as f:
            json.dump(models['feature_names'], f, indent=2)

        logger.info(f"✅ Models saved for {mood}")


def train_all_moods(datasets: Dict[str, Dict[str, pd.DataFrame]],
                   feature_cols: List[str],
                   config: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
    """
    Train models for all mood categories

    Args:
        datasets: Dictionary with train/test data for each mood
        feature_cols: Feature column names
        config: Configuration dictionary

    Returns:
        Dictionary with models for each mood
    """
    trainer = ModelTrainer(config)
    all_models = {}

    for mood, data in datasets.items():
        models = trainer.train_models_for_mood(
            data['train'],
            data['test'],
            feature_cols,
            mood
        )

        trainer.save_models(models, mood)
        all_models[mood] = models

    return all_models


if __name__ == "__main__":
    # Test model training
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer

    logger.info("Testing model training...")

    # Create sample data
    pipeline = DataPipeline()
    datasets = pipeline.run_pipeline(use_sample=True)

    # Engineer features
    engineer = FeatureEngineer()

    # Train for one mood (workout)
    mood = 'workout'
    train_df = engineer.create_feature_matrix(datasets[mood]['train'], mood)
    test_df = engineer.create_feature_matrix(datasets[mood]['test'], mood)

    feature_cols = engineer.get_feature_names(mood)

    # Train models
    trainer = ModelTrainer()
    models = trainer.train_models_for_mood(train_df, test_df, feature_cols, mood)

    print("\n✅ Model training test completed!")
    print(f"Models trained: {list(models.keys())}")
