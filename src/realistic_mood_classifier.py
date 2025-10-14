"""
REALISTIC MOOD CLASSIFICATION SYSTEM
Multi-score approach for authentic music mood prediction

This replaces the binary classification with a more realistic system
where tracks can belong to multiple moods with different intensities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os

class RealisticMoodClassifier:
    """
    Realistic mood classification that allows tracks to have scores
    for multiple moods simultaneously, reflecting real-world complexity.
    """

    def __init__(self):
        """Initialize the mood classifier with realistic scoring functions"""

        # Define mood scoring functions - each returns a score from 0-1
        self.mood_functions = {
            'workout': self._calculate_workout_score,
            'sleep': self._calculate_sleep_score,
            'party': self._calculate_party_score,
            'focus': self._calculate_focus_score,
            'chill': self._calculate_chill_score
        }

        # Mood importance weights for normalization
        self.mood_weights = {
            'workout': 1.0,
            'sleep': 1.0,
            'party': 1.0,
            'focus': 1.0,
            'chill': 1.0
        }

    def _calculate_workout_score(self, features: Dict[str, float]) -> float:
        """
        Workout mood: High energy + fast tempo + positive + danceable
        """
        energy_score = features.get('energy', 0.5)
        tempo_score = min(features.get('tempo', 120) / 180.0, 1.0)  # Normalize tempo
        valence_score = features.get('valence', 0.5)
        dance_score = features.get('danceability', 0.5)

        # Weighted combination
        score = (
            energy_score * 0.4 +      # High energy most important
            tempo_score * 0.3 +       # Fast tempo
            valence_score * 0.2 +     # Positive mood
            dance_score * 0.1         # Danceable
        )

        return min(max(score, 0.0), 1.0)  # Clamp to [0,1]

    def _calculate_sleep_score(self, features: Dict[str, float]) -> float:
        """
        Sleep mood: Low energy + slow tempo + acoustic/instrumental
        """
        energy_score = 1.0 - features.get('energy', 0.5)  # Low energy = high sleep score
        tempo_score = 1.0 - min(features.get('tempo', 90) / 100.0, 1.0)  # Slow tempo
        acoustic_score = features.get('acousticness', 0.5)
        instrumental_score = features.get('instrumentalness', 0.5)

        # Weighted combination
        score = (
            energy_score * 0.3 +      # Low energy most important
            tempo_score * 0.3 +       # Slow tempo
            acoustic_score * 0.2 +    # Acoustic
            instrumental_score * 0.2  # Instrumental
        )

        return min(max(score, 0.0), 1.0)

    def _calculate_party_score(self, features: Dict[str, float]) -> float:
        """
        Party mood: Very energetic + danceable + positive + loud
        """
        energy_score = features.get('energy', 0.5)
        dance_score = features.get('danceability', 0.5)
        valence_score = features.get('valence', 0.5)
        acoustic_penalty = 1.0 - features.get('acousticness', 0.5)  # Less acoustic = more party

        # Weighted combination
        score = (
            energy_score * 0.4 +      # Very energetic
            dance_score * 0.3 +       # Very danceable
            valence_score * 0.2 +     # Positive/happy
            acoustic_penalty * 0.1    # Not acoustic
        )

        return min(max(score, 0.0), 1.0)

    def _calculate_focus_score(self, features: Dict[str, float]) -> float:
        """
        Focus mood: Moderate energy + instrumental + low vocals + acoustic
        """
        # Moderate energy (peak around 0.5)
        energy = features.get('energy', 0.5)
        energy_score = 1.0 - abs(energy - 0.5) * 2  # Peak at 0.5

        instrumental_score = features.get('instrumentalness', 0.5)
        speech_penalty = 1.0 - features.get('speechiness', 0.5)  # Low vocals
        acoustic_score = features.get('acousticness', 0.5)

        # Weighted combination
        score = (
            energy_score * 0.3 +      # Moderate energy
            instrumental_score * 0.3 + # Instrumental
            speech_penalty * 0.2 +    # Low vocals
            acoustic_score * 0.2      # Acoustic
        )

        return min(max(score, 0.0), 1.0)

    def _calculate_chill_score(self, features: Dict[str, float]) -> float:
        """
        Chill mood: Relaxed energy + pleasant + moderate tempo + somewhat groovy
        """
        # Relaxed energy (peak around 0.4)
        energy = features.get('energy', 0.4)
        energy_score = 1.0 - abs(energy - 0.4) * 2  # Peak at 0.4

        valence_score = features.get('valence', 0.5)

        # Moderate tempo (peak around 90 BPM)
        tempo = features.get('tempo', 90)
        tempo_score = 1.0 - abs(tempo - 90) / 40.0  # Peak at 90, decay over 40 BPM range

        dance_score = features.get('danceability', 0.5)

        # Weighted combination
        score = (
            energy_score * 0.3 +      # Relaxed energy
            valence_score * 0.3 +     # Pleasant
            tempo_score * 0.2 +       # Moderate tempo
            dance_score * 0.2         # Somewhat groovy
        )

        return min(max(score, 0.0), 1.0)

    def calculate_mood_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate mood scores for all moods given track features

        Args:
            features: Dictionary of Spotify audio features

        Returns:
            Dictionary mapping mood names to scores (0-1)
        """
        mood_scores = {}

        for mood, func in self.mood_functions.items():
            score = func(features)
            mood_scores[mood] = score

        return mood_scores

    def get_primary_mood(self, features: Dict[str, float]) -> str:
        """
        Get the primary (highest scoring) mood for a track
        """
        mood_scores = self.calculate_mood_scores(features)
        return max(mood_scores.items(), key=lambda x: x[1])[0]

    def get_mood_rankings(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Get moods ranked by compatibility score (highest first)
        """
        mood_scores = self.calculate_mood_scores(features)
        return sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)

    def calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculate mood-based similarity between two tracks
        """
        scores1 = self.calculate_mood_scores(features1)
        scores2 = self.calculate_mood_scores(features2)

        # Cosine similarity of mood vectors
        vec1 = np.array(list(scores1.values()))
        vec2 = np.array(list(scores2.values()))

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

# Global instance for easy access
mood_classifier = RealisticMoodClassifier()

def get_track_mood_scores(track_features: Dict[str, float]) -> Dict[str, float]:
    """
    Convenience function to get mood scores for a track
    """
    return mood_classifier.calculate_mood_scores(track_features)

def get_mood_recommendations(track_features: Dict[str, float],
                           all_tracks: List[Dict],
                           mood: str,
                           top_k: int = 10) -> List[Tuple[Dict, float]]:
    """
    Get mood-based recommendations for a track

    Args:
        track_features: Features of the seed track
        all_tracks: List of all available tracks with features
        mood: Target mood to recommend for
        top_k: Number of recommendations to return

    Returns:
        List of (track, compatibility_score) tuples
    """
    recommendations = []

    for track in all_tracks:
        if 'features' not in track:
            continue

        # Calculate mood compatibility
        mood_scores = mood_classifier.calculate_mood_scores(track['features'])
        compatibility = mood_scores.get(mood, 0.0)

        recommendations.append((track, compatibility))

    # Sort by compatibility (highest first)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_k]

def analyze_mood_distribution(tracks: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze the mood distribution across a collection of tracks
    """
    mood_counts = {mood: 0 for mood in mood_classifier.mood_functions.keys()}
    mood_score_sums = {mood: 0.0 for mood in mood_classifier.mood_functions.keys()}
    primary_mood_counts = {mood: 0 for mood in mood_classifier.mood_functions.keys()}

    for track in tracks:
        if 'features' not in track:
            continue

        mood_scores = mood_classifier.calculate_mood_scores(track['features'])
        primary_mood = mood_classifier.get_primary_mood(track['features'])

        # Count tracks with each mood score > 0.5
        for mood, score in mood_scores.items():
            mood_score_sums[mood] += score
            if score > 0.5:
                mood_counts[mood] += 1

        primary_mood_counts[primary_mood] += 1

    total_tracks = len([t for t in tracks if 'features' in t])

    if total_tracks == 0:
        return {}

    # Calculate averages and percentages
    analysis = {}
    for mood in mood_classifier.mood_functions.keys():
        analysis[mood] = {
            'avg_score': mood_score_sums[mood] / total_tracks,
            'high_score_percentage': (mood_counts[mood] / total_tracks) * 100,
            'primary_mood_percentage': (primary_mood_counts[mood] / total_tracks) * 100
        }

    return analysis

if __name__ == '__main__':
    # Example usage
    sample_track = {
        'energy': 0.8,
        'tempo': 140,
        'valence': 0.7,
        'danceability': 0.9,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'speechiness': 0.1
    }

    print("🎵 Realistic Mood Classification Demo")
    print("=" * 50)

    mood_scores = get_track_mood_scores(sample_track)
    print(f"Track features: {sample_track}")
    print(f"Mood scores: {mood_scores}")

    primary_mood = mood_classifier.get_primary_mood(sample_track)
    print(f"Primary mood: {primary_mood}")

    rankings = mood_classifier.get_mood_rankings(sample_track)
    print(f"Mood rankings: {rankings}")

    print("\n✅ Realistic mood classification system ready!")
