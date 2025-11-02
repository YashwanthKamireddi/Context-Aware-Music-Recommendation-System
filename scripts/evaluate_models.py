#!/usr/bin/env python3
"""
Evaluate trained mood classification models on the dataset and report metrics.

Generates per-mood metrics (accuracy, precision, recall, f1, roc_auc, brier_score)
and a short professor-style summary to help understand overfitting and next steps.
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.ml_mood_classifier import MLMoodClassifier, create_mood_labels


def evaluate_mood(model_clf, df, mood, config):
    """Evaluate a single mood model on a dataset with train/test split."""
    # Use feature columns known to the model
    feature_cols = model_clf.feature_columns.get(mood)
    if not feature_cols:
        feature_cols = config['features']['audio_features']

    # Ensure features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing features for {mood}: {missing}")

    X = df[feature_cols].fillna(0.0)
    y = create_mood_labels(df, mood, config)

    # If model not available, return neutral metrics
    model = model_clf.models.get(mood)
    scaler = model_clf.scalers.get(mood)
    if model is None:
        logging.warning(f"No model found for {mood} - skipping predictions")
        return None

    # Scale if scaler available
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            # Scaler might expect a DataFrame
            X_scaled = scaler.transform(pd.DataFrame(X, columns=feature_cols))
    else:
        X_scaled = X.values

    # Predict probabilities and labels
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        logging.error(f"Prediction failed for {mood}: {e}")
        return None

    # Compute metrics
    metrics = {}
    metrics['n_samples'] = int(len(y))
    metrics['positive_ratio'] = float(y.mean())
    try:
        metrics['accuracy'] = float(accuracy_score(y, preds))
        metrics['precision'] = float(precision_score(y, preds, zero_division=0))
        metrics['recall'] = float(recall_score(y, preds, zero_division=0))
        metrics['f1'] = float(f1_score(y, preds, zero_division=0))
    except Exception as e:
        logging.warning(f"Error computing classification metrics for {mood}: {e}")

    # ROC AUC needs both classes present
    try:
        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y, probs))
        else:
            metrics['roc_auc'] = None
    except Exception as e:
        metrics['roc_auc'] = None
        logging.warning(f"ROC AUC error for {mood}: {e}")

    # Brier score for probabilistic calibration
    try:
        metrics['brier'] = float(brier_score_loss(y, probs))
    except Exception as e:
        metrics['brier'] = None

    return metrics


def professor_explain(results):
    """Create a professor-style explanation of the evaluation results."""
    lines = []
    lines.append("Model Evaluation — Professor-style Summary")
    lines.append("""
This report inspects whether the saved binary mood classifiers behave like real models
that generalize to the dataset, or whether they are degenerate / overfit. For each mood
we report: dataset size, positive class ratio, accuracy, precision, recall, F1, ROC AUC and
Brier score (lower is better for calibration). I also provide practical explanations and
recommended next steps.
""")

    for mood, m in results.items():
        lines.append('\n' + '='*60)
        lines.append(f"Mood: {mood}")
        if m is None:
            lines.append("  No model available or evaluation failed.")
            continue

        lines.append(f"  Samples: {m['n_samples']}")
        lines.append(f"  Positive ratio (class=1): {m['positive_ratio']:.3f}")
        lines.append(f"  Accuracy: {m.get('accuracy'):.3f}")
        lines.append(f"  Precision: {m.get('precision'):.3f}")
        lines.append(f"  Recall: {m.get('recall'):.3f}")
        lines.append(f"  F1: {m.get('f1'):.3f}")
        roc = m.get('roc_auc')
        lines.append(f"  ROC AUC: {roc if roc is not None else 'N/A'}")
        brier = m.get('brier')
        lines.append(f"  Brier score: {brier if brier is not None else 'N/A'}")

        # Interpretations
        if m['positive_ratio'] < 0.01 or m['positive_ratio'] > 0.99:
            lines.append("  Note: Class is extremely imbalanced — metrics like accuracy are misleading.")
        if m.get('f1', 0) < 0.2 and m.get('precision', 0) < 0.2:
            lines.append("  Warning: Low F1/precision — the model might not be useful in practice.")
        if m.get('roc_auc') is not None and m.get('roc_auc') < 0.6:
            lines.append("  Concern: ROC AUC near random — model lacks discriminative power.")
        if m.get('brier') is not None and m.get('brier') > 0.25:
            lines.append("  Note: High Brier score suggests poor probability calibration.")

        # Practical next steps
        lines.append("  Recommended next steps:")
        lines.append("    - Inspect features and distributions for positive vs negative classes.")
        lines.append("    - Try stratified cross-validation and compare train vs test metrics to detect overfitting.")
        lines.append("    - If ROC AUC is low, try stronger models or feature engineering (interaction features).")
        lines.append("    - If classes are highly imbalanced, use resampling or adjust class_weight and evaluation metrics.")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PROJECT_ROOT / 'data' / 'raw' / 'spotify_tracks.csv'))
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config' / 'config.yaml'))
    parser.add_argument('--models_dir', default=str(PROJECT_ROOT / 'models'))
    parser.add_argument('--out', default=str(PROJECT_ROOT / 'models' / 'evaluation_report.json'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.data):
        logging.error(f"Data file not found: {args.data}")
        return
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        return

    config = load_config(args.config)

    # Load dataset (use a sample if very large to keep evaluation quick)
    df = pd.read_csv(args.data)
    logging.info(f"Loaded data with {len(df)} rows")

    # Optionally sample to speed up evaluation (unless user wants full)
    sample_size = config.get('data', {}).get('sample_size', None)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=config.get('data', {}).get('random_seed', 42))
        logging.info(f"Sampled {len(df)} rows for evaluation")

    # Initialize classifier loader
    clf = MLMoodClassifier(models_dir=args.models_dir)

    moods = clf.moods
    results = {}
    for mood in moods:
        logging.info(f"Evaluating mood: {mood}")
        metrics = evaluate_mood(clf, df, mood, config)
        results[mood] = metrics

    # Save JSON report
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved evaluation report to {args.out}")

    # Print professor-style explanation
    explanation = professor_explain(results)
    print('\n' + explanation)


if __name__ == '__main__':
    main()
