# Vibe-Sync Data Directory

## Structure

- `raw/`: Raw Spotify dataset files
  - Place your Kaggle Spotify dataset CSV here
  - Example: `spotify_tracks.csv`

- `processed/`: Processed and labeled data
  - Generated automatically by the pipeline
  - Contains mood-labeled tracks and train/test splits

## Dataset Sources

### Recommended: Kaggle Spotify Dataset
- **Dataset**: "Spotify Tracks DB" or "Ultimate Spotify Tracks DB"
- **Size**: ~600K tracks with audio features
- **Download**: https://www.kaggle.com/datasets/

### Features Included:
- Audio features: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence
- Metadata: track name, artists, album, popularity
- Genre information (in some versions)

## Usage

If you don't have the dataset:
```bash
# The system will automatically create sample data for testing
python run.py --mode full --use-sample
```

If you have the dataset:
1. Download from Kaggle
2. Place CSV in `data/raw/`
3. Update `config/config.yaml` with the correct filename
4. Run: `python run.py --mode full`
