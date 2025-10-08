"""
Interactive CLI Demo for Vibe-Sync
Real-time mood-based music recommendations
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.data_pipeline import DataPipeline
from src.recommender import MoodRecommender
import pandas as pd


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🎵 VIBE-SYNC 🎵                          ║
    ║           Context-Aware Music Recommendations                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_moods(config: dict):
    """
    Display available moods
    
    Args:
        config: Configuration dictionary
    """
    print("\n🎵 Select Your Current Vibe:\n")
    
    moods = config['moods']
    for i, (mood_key, mood_info) in enumerate(moods.items(), 1):
        print(f"  {i}. {mood_info['emoji']} {mood_info['name']}")
    
    print(f"\n  0. ❌ Exit")


def get_user_choice(max_choice: int) -> int:
    """
    Get user's mood choice
    
    Args:
        max_choice: Maximum valid choice
    
    Returns:
        User's choice (0-indexed)
    """
    while True:
        try:
            choice = int(input("\n👉 Enter your choice (0-5): "))
            if 0 <= choice <= max_choice:
                return choice - 1  # Convert to 0-indexed
            else:
                print(f"❌ Please enter a number between 0 and {max_choice}")
        except ValueError:
            print("❌ Please enter a valid number")


def display_recommendations(recommendations: pd.DataFrame, mood: str, emoji: str):
    """
    Display recommendations in a nice format
    
    Args:
        recommendations: DataFrame with recommendations
        mood: Mood name
        emoji: Mood emoji
    """
    print(f"\n{'='*70}")
    print(f"{emoji} YOUR PERSONALIZED {mood.upper()} PLAYLIST {emoji}")
    print(f"{'='*70}\n")
    
    for idx, row in recommendations.iterrows():
        rank = idx + 1
        name = row['name']
        artists = row['artists']
        score = row['final_score']
        
        # Create score bar
        bar_length = int(score * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        
        print(f"{rank:2d}. {name[:40]:<40} - {artists[:25]:<25}")
        print(f"    Match: {bar} {score:.1%}\n")


def run_interactive_demo():
    """
    Run the interactive demo
    """
    # Setup
    config = load_config()
    logger = setup_logging('WARNING')  # Less verbose for demo
    
    print_banner()
    
    print("\n🔄 Loading Vibe-Sync system...")
    
    # Load data
    pipeline = DataPipeline(config)
    pipeline.load_data(use_sample=True)
    df = pipeline.clean_data()
    df = pipeline.assign_mood_labels(df)
    
    print("✅ System loaded successfully!")
    
    # Create recommender
    recommender = MoodRecommender(config)
    
    # Get mood list
    mood_list = list(config['moods'].keys())
    
    while True:
        # Display moods
        display_moods(config)
        
        # Get user choice
        choice = get_user_choice(len(mood_list))
        
        if choice == -1:  # User chose 0 (exit)
            print("\n👋 Thanks for using Vibe-Sync! Keep vibing! 🎵\n")
            break
        
        # Get selected mood
        selected_mood = mood_list[choice]
        mood_info = config['moods'][selected_mood]
        
        print(f"\n🎵 Generating {mood_info['emoji']} {mood_info['name']} playlist...")
        print("⏳ Analyzing tracks and your vibe...\n")
        
        try:
            # Get candidates (sample from dataset)
            candidates = df.sample(min(500, len(df)))
            
            # Create sample user profile (random sample of tracks)
            user_tracks = df.sample(min(20, len(df)))
            
            # Generate recommendations
            recommendations = recommender.recommend(
                candidates,
                selected_mood,
                user_tracks,
                top_k=20
            )
            
            # Display
            display_recommendations(recommendations, mood_info['name'], mood_info['emoji'])
            
            # Stats
            print(f"{'='*70}")
            print(f"📊 Playlist Stats:")
            print(f"  • Total tracks analyzed: {len(candidates)}")
            print(f"  • Recommendations: {len(recommendations)}")
            print(f"  • Average match score: {recommendations['final_score'].mean():.1%}")
            print(f"  • Top match score: {recommendations['final_score'].max():.1%}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n❌ Error generating recommendations: {e}")
            print("💡 Tip: Run the full pipeline first to train models")
            print("   Command: python run.py --mode full --use-sample")
        
        # Continue?
        print("\n")
        continue_choice = input("🔄 Try another vibe? (y/n): ").lower()
        if continue_choice != 'y':
            print("\n👋 Thanks for using Vibe-Sync! Keep vibing! 🎵\n")
            break


def main():
    """Main entry point"""
    try:
        run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Thanks for using Vibe-Sync! 🎵\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
