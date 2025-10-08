# 🎵 Vibe-Sync: Context-Aware Music Recommendation System

A production-grade machine learning system that provides real-time, mood-based music recommendations by combining user taste profiles with contextual preferences.

## 🎯 Problem Statement

Traditional music recommendation systems suggest songs based solely on listening history, failing to adapt to users' immediate context or mood. Vibe-Sync bridges this gap by allowing users to select their current vibe (Workout, Chill, Party, Focus, Sleep) and generates highly relevant, situational playlists that match both their taste and mood.

## 🚀 Key Features

- **Mood-Based Recommendations**: 5 distinct mood categories (Workout, Chill, Party, Focus, Sleep)
- **Hybrid Algorithm**: Combines user taste profile (30%) + mood matching (60%) + diversity (10%)
- **Real-Time Generation**: Instant playlist creation based on current context
- **Production-Ready**: Clean, modular, scalable architecture
- **Comprehensive Evaluation**: F1-Score, ROC-AUC, Precision@K metrics
- **Interactive Demo**: CLI interface for testing recommendations

## 📊 Technical Approach

### ML Pipeline
1. **Data Processing**: Spotify track dataset with audio features
2. **Feature Engineering**: Track features + Mood features + Interaction features
3. **Model Training**: LightGBM classifier with cross-validation
4. **Evaluation**: Multi-metric assessment with business impact analysis
5. **Deployment**: Real-time recommendation engine

### Models Used
- **Baseline**: Logistic Regression
- **Main Model**: LightGBM Classifier
- **Optimization**: Hyperparameter tuning via GridSearchCV

### Evaluation Metrics
- F1-Score (balanced precision/recall)
- ROC-AUC (classification quality)
- Precision@K (top-K recommendation accuracy)
- Business metrics (engagement, diversity)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
cd Context-Aware-Music-Recommendation-System

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### One-Command Execution
```bash
# Run complete pipeline (data → train → evaluate → demo)
python run.py --mode full

# Train models only
python run.py --mode train

# Launch interactive demo
python run.py --mode demo

# Evaluate trained models
python run.py --mode evaluate
```

### Interactive Demo
```bash
python app/main.py

# Select your mood and get instant recommendations!
```

## 📁 Project Structure

```
vibe-sync/
├── src/
│   ├── data_pipeline.py        # Data loading & preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── model_trainer.py        # Model training & tuning
│   ├── recommender.py          # Recommendation engine
│   ├── evaluator.py            # Metrics & evaluation
│   └── utils.py                # Helper functions
├── app/
│   └── main.py                 # Interactive CLI demo
├── config/
│   └── config.yaml             # Configuration settings
├── data/                       # Data storage
├── models/                     # Saved models
├── results/                    # Reports & visualizations
├── run.py                      # Main execution script
└── requirements.txt
```

## 🎵 Mood Categories

| Mood | Characteristics |
|------|----------------|
| 🏋️ **Workout** | High energy (>0.7), Fast tempo (>120 BPM), High danceability |
| 😌 **Chill** | Low energy (<0.5), High acousticness, Calm valence |
| 🎉 **Party** | High danceability (>0.7), High energy, Positive valence |
| 📚 **Focus** | Instrumental, Low energy, Minimal vocals |
| 😴 **Sleep** | Very low energy (<0.3), Acoustic, Slow tempo |

## 📈 Results

Performance metrics will be displayed after training:
- Model comparison (Logistic Regression vs LightGBM)
- Feature importance analysis
- Confusion matrices
- Sample recommendations with match scores

## 🎯 Business Impact

- **User Engagement**: Increased listening time through contextual relevance
- **Discovery**: Enhanced music exploration aligned with current mood
- **Retention**: Improved user satisfaction and platform loyalty
- **Personalization**: Dynamic adaptation to user context

## 🔬 Technical Details

### Feature Engineering
- **Track Audio Features**: energy, danceability, valence, tempo, acousticness, etc.
- **Mood Profile Features**: Aggregated statistics per mood category
- **Interaction Features**: Difference/ratio between track and mood profile

### Model Architecture
- Binary classification per mood category
- Multi-label capability for hybrid moods
- Confidence scoring for ranking

## 📝 Configuration

Edit `config/config.yaml` to customize:
- Data source and size
- Mood definitions and thresholds
- Model hyperparameters
- Evaluation settings

## 🤝 Contributing

This is an academic project. Suggestions and improvements are welcome!

## 📄 License

MIT License

## 👨‍💻 Author

Built as a comprehensive ML project demonstrating:
- Production-grade code architecture
- Real-world problem solving
- End-to-end ML pipeline
- Business impact thinking

---

**Last Updated**: October 2025
