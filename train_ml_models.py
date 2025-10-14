#!/usr/bin/env python3
"""
MACHINE LEARNING MODEL TRAINING SCRIPT
Trains ML models for mood classification using scikit-learn

This script implements the complete ML pipeline:
1. Load and preprocess data
2. Create mood labels based on configuration
3. Train ML models for each mood
4. Evaluate and save models
5. Generate training reports
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_mood_classifier import load_and_train_all_models
from src.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Train ML models for mood classification')
    parser.add_argument('--data', '-d', default='data/raw/spotify_tracks.csv',
                       help='Path to Spotify tracks CSV file')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', '-m', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level=log_level)

    logger.info("🎯 Starting ML Model Training Pipeline")
    logger.info("=" * 50)

    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"❌ Data file not found: {args.data}")
        logger.error("Please download the Spotify dataset and place it in data/raw/")
        sys.exit(1)

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    try:
        # Train all models
        logger.info("🚀 Training ML models for all moods...")
        metrics = load_and_train_all_models(
            data_path=args.data,
            config_path=args.config,
            models_dir=args.models
        )

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 TRAINING RESULTS SUMMARY")
        logger.info("=" * 50)

        for mood, mood_metrics in metrics.items():
            if 'error' in mood_metrics:
                logger.error(f"❌ {mood.upper()}: {mood_metrics['error']}")
            else:
                acc = mood_metrics['accuracy']
                f1 = mood_metrics['f1_score']
                logger.info(f"✅ {mood.upper()}: Accuracy={acc:.3f}, F1-Score={f1:.3f}")

        logger.info("\n" + "=" * 50)
        logger.info("🎉 ML Training Pipeline Complete!")
        logger.info(f"📁 Models saved to: {args.models}/")
        logger.info("🔧 Ready for deployment and evaluation")
        # Save detailed report
        report_path = os.path.join(args.models, "training_report.md")
        generate_training_report(metrics, report_path)
        logger.info(f"📄 Detailed report saved to: {report_path}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


def generate_training_report(metrics: dict, output_path: str):
    """Generate a detailed training report in Markdown format"""

    report = f"""# Machine Learning Mood Classification Training Report

## Overview
This report summarizes the training results for mood classification models using machine learning algorithms.

## Dataset Information
- **Source**: Spotify Tracks Dataset (Kaggle)
- **Features**: 9 audio features (acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence)
- **Moods Classified**: workout, chill, party, focus, sleep

## Model Architecture
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 10
  - class_weight: balanced
- **Preprocessing**: StandardScaler for feature normalization

## Training Results

"""

    for mood, mood_metrics in metrics.items():
        if 'error' in mood_metrics:
            report += f"### {mood.upper()} - ERROR\n"
            report += f"**Error**: {mood_metrics['error']}\n\n"
        else:
            report += f"### {mood.upper()}\n"
            report += f"- **Accuracy**: {mood_metrics['accuracy']:.3f}\n"
            report += f"- **Precision**: {mood_metrics['precision']:.3f}\n"
            report += f"- **Recall**: {mood_metrics['recall']:.3f}\n"
            report += f"- **F1-Score**: {mood_metrics['f1_score']:.3f}\n"
            report += f"- **Training Size**: {mood_metrics['train_size']}\n"
            report += f"- **Test Size**: {mood_metrics['test_size']}\n"
            report += f"- **Positive Class Ratio**: {mood_metrics['positive_class_ratio']:.3f}\n"
            report += "\n#### Classification Report\n"
            report += "```\n"
            report += f"              precision    recall  f1-score   support\n\n"
            report += f"           0       {mood_metrics['classification_report']['0']['precision']:.2f}      {mood_metrics['classification_report']['0']['recall']:.2f}      {mood_metrics['classification_report']['0']['f1-score']:.2f}       {mood_metrics['classification_report']['0']['support']}\n"
            report += f"           1       {mood_metrics['classification_report']['1']['precision']:.2f}      {mood_metrics['classification_report']['1']['recall']:.2f}      {mood_metrics['classification_report']['1']['f1-score']:.2f}       {mood_metrics['classification_report']['1']['support']}\n\n"
            report += f"    accuracy                           {mood_metrics['classification_report']['accuracy']:.2f}       {mood_metrics['classification_report']['macro avg']['support']}\n"
            report += f"   macro avg       {mood_metrics['classification_report']['macro avg']['precision']:.2f}      {mood_metrics['classification_report']['macro avg']['recall']:.2f}      {mood_metrics['classification_report']['macro avg']['f1-score']:.2f}       {mood_metrics['classification_report']['macro avg']['support']}\n"
            report += f"weighted avg       {mood_metrics['classification_report']['weighted avg']['precision']:.2f}      {mood_metrics['classification_report']['weighted avg']['recall']:.2f}      {mood_metrics['classification_report']['weighted avg']['f1-score']:.2f}       {mood_metrics['classification_report']['weighted avg']['support']}\n"
            report += "```\n\n"

    report += """## Model Files Generated
For each mood, the following files are saved in the `models/` directory:
- `{mood}_model.pkl`: Trained Random Forest model
- `{mood}_scaler.pkl`: Fitted StandardScaler for feature normalization
- `{mood}_features.json`: List of features used for training

## Usage in Application
The trained models are loaded by the `MLMoodClassifier` class and used for:
1. Real-time mood prediction for individual tracks
2. Batch processing for recommendation generation
3. Probability scoring for mood compatibility

## Next Steps
1. **Model Evaluation**: Analyze confusion matrices and ROC curves
2. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
3. **Feature Engineering**: Consider additional derived features
4. **Model Comparison**: Try other algorithms (SVM, XGBoost, Neural Networks)
5. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation

## Educational Value
This implementation demonstrates:
- Complete ML pipeline (data → model → evaluation → deployment)
- Binary classification for multi-class problem
- Proper train/test split and evaluation metrics
- Model serialization and loading
- Real-world application integration
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    main()
