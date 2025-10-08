"""
Simple Demo - Works without trained models
Shows the recommendation system using rule-based scoring
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.data_pipeline import DataPipeline
import pandas as pd


def simple_mood_match(track: pd.Series, mood_criteria: dict) -> float:
    """Calculate simple mood match score"""
    matches = 0
    total = 0
    
    for criterion, threshold in mood_criteria.items():
        feature = criterion.replace('_min', '').replace('_max', '')
        if feature in track.index:
            total += 1
            if '_min' in criterion and track[feature] >= threshold:
                matches += 1
            elif '_max' in criterion and track[feature] <= threshold:
                matches += 1
    
    return matches / total if total > 0 else 0


def generate_playlist(tracks: pd.DataFrame, mood: str, mood_info: dict, top_n: int = 20) -> pd.DataFrame:
    """Generate playlist for a mood"""
    results = []
    
    for idx, track in tracks.iterrows():
        score = simple_mood_match(track, mood_info['criteria'])
        results.append({
            'name': track.get('name', f'Track {idx}'),
            'artists': track.get('artists', 'Unknown Artist'),
            'score': score,
            'energy': track.get('energy', 0),
            'danceability': track.get('danceability', 0),
            'valence': track.get('valence', 0)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    return results_df.head(top_n)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🎵 VIBE-SYNC DEMO - Rule-Based Recommendations")
    print("="*70 + "\n")


def display_recommendations(recommendations: pd.DataFrame, mood_name: str, emoji: str):
    """Display recommendations"""
    print(f"\n{emoji} YOUR {mood_name.upper()} PLAYLIST {emoji}")
    print("="*70 + "\n")
    
    for i, row in recommendations.iterrows():
        rank = i + 1 if isinstance(i, int) else len(recommendations) - len(recommendations) + list(recommendations.index).index(i) + 1
        name = row['name'][:40]
        artists = row['artists'][:20]
        score = row['score']
        
        # Score bar
        bar_length = int(score * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        
        print(f"{rank:2d}. {name:<40} - {artists:<20}")
        print(f"    Match: {bar} {score:.1%}")
        print(f"    Energy: {row['energy']:.2f} | Dance: {row['danceability']:.2f} | Valence: {row['valence']:.2f}\n")


def main():
    """Main demo function"""
    print_banner()
    
    # Load config
    config = load_config()
    moods = config['moods']
    
    print("🔄 Generating sample music library...")
    
    # Load sample data
    pipeline = DataPipeline(config)
    pipeline.load_data(use_sample=True)
    df = pipeline.clean_data()
    
    print(f"✅ Loaded {len(df)} tracks\n")
    
    while True:
        # Display moods
        print("🎵 Select Your Vibe:\n")
        mood_list = list(moods.keys())
        for i, (mood_key, mood_info) in enumerate(moods.items(), 1):
            print(f"  {i}. {mood_info['emoji']} {mood_info['name']}")
        print(f"\n  0. ❌ Exit\n")
        
        # Get choice
        try:
            choice = int(input("👉 Enter your choice (0-5): "))
            if choice == 0:
                print("\n👋 Thanks for trying Vibe-Sync! 🎵\n")
                break
            if 1 <= choice <= len(mood_list):
                mood_key = mood_list[choice - 1]
                mood_info = moods[mood_key]
                
                print(f"\n🎵 Generating {mood_info['emoji']} {mood_info['name']} playlist...")
                
                # Generate recommendations
                recommendations = generate_playlist(
                    df.sample(min(500, len(df))),
                    mood_key,
                    mood_info,
                    top_n=15
                )
                
                # Display
                display_recommendations(recommendations, mood_info['name'], mood_info['emoji'])
                
                print("="*70)
                print(f"📊 Stats:")
                print(f"  • Tracks analyzed: 500")
                print(f"  • Recommendations: {len(recommendations)}")
                print(f"  • Avg match score: {recommendations['score'].mean():.1%}")
                print(f"  • Top score: {recommendations['score'].max():.1%}")
                print("="*70 + "\n")
                
                cont = input("🔄 Try another vibe? (y/n): ").lower()
                if cont != 'y':
                    print("\n👋 Thanks for trying Vibe-Sync! 🎵\n")
                    break
            else:
                print("❌ Invalid choice. Please try again.\n")
        except (ValueError, KeyboardInterrupt):
            print("\n\n👋 Thanks for trying Vibe-Sync! 🎵\n")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
