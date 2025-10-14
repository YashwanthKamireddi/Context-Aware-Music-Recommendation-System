#!/usr/bin/env python3
"""
Railway Deployment Helper for Vibe-Sync
This script helps prepare and validate the project for Railway deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'requirements.txt',
        'backend/server.py',
        'config/config.yaml',
        'railway.json',
        '.env.example'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False

    print("✅ All required files present")
    return True

def check_dependencies():
    """Check if Python dependencies can be imported"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import sklearn
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_models():
    """Check if trained models exist"""
    model_files = [
        'models/workout_model.pkl',
        'models/chill_model.pkl',
        'models/party_model.pkl',
        'models/focus_model.pkl',
        'models/sleep_model.pkl'
    ]

    missing = []
    for model in model_files:
        if not Path(model).exists():
            missing.append(model)

    if missing:
        print(f"⚠️  Missing model files: {', '.join(missing)}")
        print("   Run 'python train_ml_models.py' to train models")
        return False

    print("✅ All model files present")
    return True

def check_data():
    """Check if dataset exists"""
    data_file = 'data/raw/spotify_tracks.csv'
    if not Path(data_file).exists():
        print(f"❌ Dataset not found: {data_file}")
        print("   Download from Kaggle and place in data/raw/")
        return False

    print("✅ Dataset present")
    return True

def main():
    """Main deployment check"""
    print("🚀 Vibe-Sync Railway Deployment Check")
    print("=" * 40)

    checks = [
        check_requirements,
        check_dependencies,
        check_models,
        check_data
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✅ Project is ready for Railway deployment!")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Go to https://railway.app")
        print("3. Connect your GitHub repository")
        print("4. Add Spotify API credentials in Railway environment variables")
        print("5. Deploy!")
    else:
        print("❌ Project needs fixes before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()