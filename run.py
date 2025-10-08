"""
Main execution script for Vibe-Sync
Run complete pipeline: data → features → train → evaluate → demo
"""

import sys
import os
import argparse
import logging
import warnings

# Suppress ALL warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Silence joblib warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Silence Python warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, setup_logging, ensure_dir
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.recommender import MoodRecommender


def run_data_pipeline(use_sample: bool = True) -> dict:
    """
    Run data processing pipeline
    
    Args:
        use_sample: Use sample data for testing
    
    Returns:
        Dictionary with processed datasets
    """
    logger.info("🎵 STEP 1: Data Pipeline")
    logger.info("="*60)
    
    pipeline = DataPipeline()
    datasets = pipeline.run_pipeline(use_sample=use_sample)
    
    return datasets


def run_feature_engineering(datasets: dict) -> dict:
    """
    Run feature engineering for all moods
    
    Args:
        datasets: Output from data pipeline
    
    Returns:
        Dictionary with feature-engineered datasets
    """
    logger.info("\n🎵 STEP 2: Feature Engineering")
    logger.info("="*60)
    
    engineer = FeatureEngineer()
    engineered_datasets = {}
    
    for mood, data in datasets.items():
        logger.info(f"\nEngineering features for {mood}...")
        
        train_df = engineer.create_feature_matrix(data['train'], mood)
        test_df = engineer.create_feature_matrix(data['test'], mood)
        
        engineered_datasets[mood] = {
            'train': train_df,
            'test': test_df,
            'feature_names': engineer.get_feature_names(mood)
        }
    
    logger.info("\n✅ Feature engineering completed!")
    return engineered_datasets


def run_model_training(datasets: dict) -> dict:
    """
    Train models for all moods
    
    Args:
        datasets: Feature-engineered datasets
    
    Returns:
        Dictionary with trained models
    """
    logger.info("\n🎵 STEP 3: Model Training")
    logger.info("="*60)
    
    trainer = ModelTrainer()
    all_models = {}
    
    for mood, data in datasets.items():
        models = trainer.train_models_for_mood(
            data['train'],
            data['test'],
            data['feature_names'],
            mood
        )
        
        trainer.save_models(models, mood)
        all_models[mood] = models
    
    logger.info("\n✅ Model training completed!")
    return all_models


def run_evaluation(datasets: dict, models: dict) -> dict:
    """
    Evaluate all trained models
    
    Args:
        datasets: Feature-engineered datasets
        models: Trained models
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("\n🎵 STEP 4: Model Evaluation")
    logger.info("="*60)
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    for mood in models.keys():
        # Prepare test data
        trainer = ModelTrainer()
        X_test, y_test = trainer.prepare_data(
            datasets[mood]['test'],
            datasets[mood]['feature_names']
        )
        
        # Evaluate
        results = evaluator.evaluate_all_models(
            models[mood],
            X_test,
            y_test,
            mood
        )
        
        all_results[mood] = results
    
    logger.info("\n✅ Evaluation completed!")
    return all_results


def run_demo():
    """
    Run interactive recommendation demo
    """
    logger.info("\n🎵 STEP 5: Interactive Demo")
    logger.info("="*60)
    
    # Import demo module
    from app.main import run_interactive_demo
    
    run_interactive_demo()


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Vibe-Sync: Mood-Based Music Recommendation System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'data', 'train', 'evaluate', 'demo'],
                       help='Execution mode')
    parser.add_argument('--use-sample', action='store_true',
                       help='Use sample data instead of full dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Setup
    global logger
    config = load_config(args.config)
    logger = setup_logging(config['output']['log_level'])
    
    # Create output directories
    ensure_dir(config['output']['models_dir'])
    ensure_dir(config['output']['results_dir'])
    ensure_dir(config['output']['plots_dir'])
    
    logger.info("\n" + "="*60)
    logger.info("🎵 VIBE-SYNC: CONTEXT-AWARE MUSIC RECOMMENDATION SYSTEM")
    logger.info("="*60 + "\n")
    
    try:
        if args.mode in ['full', 'data']:
            datasets = run_data_pipeline(use_sample=args.use_sample)
            
            if args.mode == 'data':
                logger.info("\n✅ Data pipeline completed successfully!")
                return
            
            # Continue with feature engineering
            datasets = run_feature_engineering(datasets)
            models = run_model_training(datasets)
            results = run_evaluation(datasets, models)
            
            # Print summary
            print_final_summary(results)
            
            if args.mode == 'full':
                logger.info("\n🎮 Launching interactive demo...")
                run_demo()
        
        elif args.mode == 'train':
            # Load processed data and train
            logger.info("Loading processed data...")
            # TODO: Load from saved files
            logger.info("Please run full pipeline first or use --mode full")
        
        elif args.mode == 'evaluate':
            # Load models and evaluate
            logger.info("Loading trained models...")
            # TODO: Load from saved files
            logger.info("Please run full pipeline first or use --mode full")
        
        elif args.mode == 'demo':
            run_demo()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_final_summary(results: dict):
    """
    Print final summary of results
    
    Args:
        results: Evaluation results
    """
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60 + "\n")
    
    summary_data = []
    for mood, mood_results in results.items():
        for model_name, metrics in mood_results.items():
            summary_data.append({
                'Mood': mood.capitalize(),
                'Model': model_name.capitalize(),
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}"
            })
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("📁 Output files saved to:")
    config = load_config()
    print(f"  • Models: {config['output']['models_dir']}")
    print(f"  • Results: {config['output']['results_dir']}")
    print(f"  • Plots: {config['output']['plots_dir']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
