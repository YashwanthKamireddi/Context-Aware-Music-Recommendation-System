"""Create a production-ready curated dataset from the Kaggle Spotify tracks dump.

This script selects a high-quality, representative subset of *real* tracks from the
Kaggle dataset while ensuring the resulting artifact stays comfortably below the
10 MB file-limit imposed by Hugging Face Spaces.

Usage (from repo root):
    python scripts/build_curated_dataset.py \
        --input data/raw/spotify_tracks.csv \
        --output data/processed/tracks_curated.parquet

The curated dataset preserves audio features required by the ML models along
with key metadata for the frontend. Sampling is stratified by genre so the API
has balanced coverage across moods.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("dataset_builder")


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure specified columns are numeric and drop rows with missing values."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[col for col in columns if col in df.columns])
    logger.info("Dropped %d rows with missing audio features", before - len(df))
    return df


def _select_representative_tracks(
    df: pd.DataFrame,
    *,
    tracks_per_genre: int,
    max_tracks: int,
    random_seed: int,
) -> pd.DataFrame:
    """Perform stratified sampling by genre favouring popular tracks."""
    rng = np.random.default_rng(random_seed)
    selections: List[pd.DataFrame] = []

    genre_counts = df["track_genre"].value_counts().to_dict()
    logger.info("Sampling across %d distinct genres", len(genre_counts))

    for genre, group in df.groupby("track_genre"):
        group = group.sort_values("popularity", ascending=False)

        target = min(tracks_per_genre, len(group))
        if target == 0:
            continue

        # Grab a slightly larger window of the top tracks to keep quality high
        window_size = min(len(group), target * 3)
        window = group.head(window_size)

        if len(window) > target:
            indices = rng.choice(window.index, size=target, replace=False)
            sample = window.loc[indices]
        else:
            sample = window

        selections.append(sample)

    curated = pd.concat(selections, ignore_index=True)
    curated = curated.sort_values(["popularity", "track_genre"], ascending=[False, True])
    curated = curated.drop_duplicates(subset=["track_id"])

    if len(curated) > max_tracks:
        logger.info(
            "Truncating curated dataset from %d to %d tracks to respect --max-tracks",
            len(curated),
            max_tracks,
        )
        curated = curated.head(max_tracks)

    return curated.reset_index(drop=True)


def build_dataset(
    input_path: Path,
    output_path: Path,
    *,
    tracks_per_genre: int = 120,
    max_tracks: int = 8000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build the curated dataset and persist it to disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    logger.info("Loading raw dataset from %s", input_path)
    df = pd.read_csv(input_path, encoding="latin-1")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["track_id", "track_genre"])
    df = df.drop_duplicates(subset=["track_id"])
    logger.info("Loaded %d unique tracks", len(df))

    numeric_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]

    df = _coerce_numeric(df, numeric_features + ["popularity", "duration_ms"])

    curated = _select_representative_tracks(
        df,
        tracks_per_genre=tracks_per_genre,
        max_tracks=max_tracks,
        random_seed=random_seed,
    )

    # Only keep the fields required by the backend + UI
    keep_columns = [
        "track_id",
        "track_name",
        "artists",
        "album_name",
        "duration_ms",
        "explicit",
        "popularity",
        "track_genre",
        *numeric_features,
    ]

    curated = curated[keep_columns]

    # Ensure column names align with backend expectations
    curated = curated.rename(
        columns={
            "track_name": "name",
            "track_id": "id",
            "album_name": "album",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing curated dataset to %s", output_path)
    curated.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Curated dataset contains %d tracks (%.2f MB)", len(curated), size_mb)

    return curated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated dataset from Kaggle dump")
    parser.add_argument("--input", type=Path, default=Path("data/raw/spotify_tracks.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/tracks_curated.parquet"))
    parser.add_argument("--tracks-per-genre", type=int, default=120)
    parser.add_argument("--max-tracks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        args.input,
        args.output,
        tracks_per_genre=args.tracks_per_genre,
        max_tracks=args.max_tracks,
        random_seed=args.seed,
    )