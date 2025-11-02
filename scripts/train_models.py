#!/usr/bin/env python3
"""
End-to-end training entry point for ML mood classifiers.

Loads the Kaggle Spotify dataset, creates mood labels from the configuration,
trains one Random Forest model per mood, and persists the trained estimators,
scalers, and feature metadata under the configured models directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_mood_classifier import load_and_train_all_models
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML mood classifiers")
    parser.add_argument(
        "--data",
        "-d",
        default="data/raw/spotify_tracks.csv",
        help="Path to the Spotify tracks CSV",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--models",
        "-m",
        default="models",
        help="Directory where trained artifacts will be written",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    data_path = Path(args.data)
    config_path = Path(args.config)
    models_dir = Path(args.models)

    logging.info("Training ML models")
    logging.info("Data: %s", data_path.resolve())
    logging.info("Config: %s", config_path.resolve())
    logging.info("Output dir: %s", models_dir.resolve())

    metrics = load_and_train_all_models(
        data_path=str(data_path),
        config_path=str(config_path),
        models_dir=str(models_dir),
    )

    logging.info("Training complete. Metrics per mood:")
    for mood, mood_metrics in metrics.items():
        logging.info("- %s: accuracy=%.3f, f1=%.3f", mood, mood_metrics.get("accuracy", 0.0), mood_metrics.get("f1_score", 0.0))


if __name__ == "__main__":
    main()
