"""
Model Evaluator: Comprehensive evaluation of trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.utils import load_config, setup_logging, ensure_dir


logger = setup_logging()


class ModelEvaluator:
    """
    Evaluate and compare trained models
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.eval_config = self.config['evaluation']
        self.output_dir = self.config['output']['results_dir']
        
        ensure_dir(self.output_dir)
        ensure_dir(self.config['output']['plots_dir'])
        
    def predict(self, model: Any, X: np.ndarray, model_type: str = 'lightgbm') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions
        
        Args:
            model: Trained model
            X: Features
            model_type: Type of model ('lightgbm' or 'baseline')
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_type == 'lightgbm':
            y_prob = model.predict(X)
            y_pred = (y_prob >= 0.5).astype(int)
        else:  # sklearn models
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
        
        return y_pred, y_prob
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        
        Returns:
            Dictionary with metric scores
        """
        metrics = {
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        return metrics
    
    def precision_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            k: Number of top predictions to consider
        
        Returns:
            Precision@K score
        """
        # Get top-k indices
        top_k_indices = np.argsort(y_prob)[-k:]
        
        # Calculate precision
        relevant = y_true[top_k_indices].sum()
        precision = relevant / k
        
        return precision
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str, model_type: str = 'lightgbm') -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
            model_type: Type of model
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred, y_prob = self.predict(model, X_test, model_type)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Precision@K
        for k in self.eval_config['precision_at_k']:
            metrics[f'precision@{k}'] = self.precision_at_k(y_test, y_prob, k)
        
        # Log results
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             mood: str, model_name: str) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            mood: Mood name
            model_name: Model name
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not ' + mood, mood],
                   yticklabels=['Not ' + mood, mood])
        plt.title(f'Confusion Matrix - {mood.capitalize()} ({model_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filepath = f"{self.config['output']['plots_dir']}/{mood}_{model_name}_confusion_matrix.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved confusion matrix to {filepath}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      mood: str, model_name: str) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            mood: Mood name
            model_name: Model name
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {mood.capitalize()} ({model_name})')
        plt.legend()
        plt.grid(alpha=0.3)
        
        filepath = f"{self.config['output']['plots_dir']}/{mood}_{model_name}_roc_curve.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved ROC curve to {filepath}")
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               mood: str, top_n: int = 20) -> None:
        """
        Plot feature importance
        
        Args:
            model: Trained LightGBM model
            feature_names: List of feature names
            mood: Mood name
            top_n: Number of top features to plot
        """
        try:
            importance = model.feature_importance(importance_type='gain')
            
            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance_df)), 
                    feature_importance_df['importance'])
            plt.yticks(range(len(feature_importance_df)), 
                      feature_importance_df['feature'])
            plt.xlabel('Importance (Gain)')
            plt.title(f'Top {top_n} Feature Importance - {mood.capitalize()}')
            plt.gca().invert_yaxis()
            
            filepath = f"{self.config['output']['plots_dir']}/{mood}_feature_importance.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved feature importance to {filepath}")
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
    
    def compare_models(self, results: Dict[str, Dict[str, float]], 
                      mood: str) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: Dictionary with results for each model
            mood: Mood name
        
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Model Comparison - {mood.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"\n{comparison_df.to_string()}\n")
        
        # Save to CSV
        filepath = f"{self.output_dir}/{mood}_model_comparison.csv"
        comparison_df.to_csv(filepath)
        logger.info(f"Saved comparison to {filepath}")
        
        return comparison_df
    
    def evaluate_all_models(self, models: Dict[str, Any], X_test: np.ndarray,
                           y_test: np.ndarray, mood: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models for a mood
        
        Args:
            models: Dictionary with trained models
            X_test: Test features
            y_test: Test labels
            mood: Mood name
        
        Returns:
            Dictionary with results for each model
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating models for {mood.upper()}")
        logger.info(f"{'='*60}\n")
        
        results = {}
        
        # Evaluate baseline
        if 'baseline' in models and 'scaler' in models:
            X_test_scaled = models['scaler'].transform(X_test)
            y_pred, y_prob = self.predict(models['baseline'], X_test_scaled, 'baseline')
            results['baseline'] = self.evaluate_model(
                models['baseline'], X_test_scaled, y_test, 'Baseline', 'baseline'
            )
            
            # Plots
            if self.eval_config['output']['save_confusion_matrix']:
                self.plot_confusion_matrix(y_test, y_pred, mood, 'baseline')
            self.plot_roc_curve(y_test, y_prob, mood, 'baseline')
        
        # Evaluate LightGBM
        if 'lightgbm' in models:
            y_pred, y_prob = self.predict(models['lightgbm'], X_test, 'lightgbm')
            results['lightgbm'] = self.evaluate_model(
                models['lightgbm'], X_test, y_test, 'LightGBM', 'lightgbm'
            )
            
            # Plots
            if self.eval_config['output']['save_confusion_matrix']:
                self.plot_confusion_matrix(y_test, y_pred, mood, 'lightgbm')
            self.plot_roc_curve(y_test, y_prob, mood, 'lightgbm')
            
            if self.eval_config['output']['save_feature_importance']:
                self.plot_feature_importance(
                    models['lightgbm'], 
                    models['feature_names'], 
                    mood
                )
        
        # Compare models
        comparison_df = self.compare_models(results, mood)
        
        return results


if __name__ == "__main__":
    # Test evaluator
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    
    logger.info("Testing model evaluation...")
    
    # Create sample data and train model
    pipeline = DataPipeline()
    datasets = pipeline.run_pipeline(use_sample=True)
    
    engineer = FeatureEngineer()
    mood = 'workout'
    
    train_df = engineer.create_feature_matrix(datasets[mood]['train'], mood)
    test_df = engineer.create_feature_matrix(datasets[mood]['test'], mood)
    
    feature_cols = engineer.get_feature_names(mood)
    
    # Train
    trainer = ModelTrainer()
    models = trainer.train_models_for_mood(train_df, test_df, feature_cols, mood)
    
    # Evaluate
    X_test, y_test = trainer.prepare_data(test_df, feature_cols)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test, mood)
    
    print("\n✅ Evaluation test completed!")
