"""
Quick Test Script - Verify Installation and Run Sample
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - Run: pip install numpy")
        return False
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - Run: pip install pandas")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError:
        print("❌ scikit-learn - Run: pip install scikit-learn")
        return False
    
    try:
        import lightgbm
        print("✅ lightgbm")
    except ImportError:
        print("❌ lightgbm - Run: pip install lightgbm")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib")
    except ImportError:
        print("❌ matplotlib - Run: pip install matplotlib")
        return False
    
    try:
        import seaborn
        print("✅ seaborn")
    except ImportError:
        print("❌ seaborn - Run: pip install seaborn")
        return False
    
    try:
        import yaml
        print("✅ pyyaml")
    except ImportError:
        print("❌ pyyaml - Run: pip install pyyaml")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'app',
        'config',
        'data',
        'models',
        'results'
    ]
    
    required_files = [
        'config/config.yaml',
        'src/utils.py',
        'src/data_pipeline.py',
        'run.py',
        'README.md'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - missing")
            all_good = False
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            all_good = False
    
    if all_good:
        print("\n✅ Project structure is correct!")
    else:
        print("\n❌ Some files/directories are missing")
    
    return all_good


def quick_test():
    """Run a quick functionality test"""
    print("\nRunning quick functionality test...")
    
    try:
        from src.utils import load_config, create_sample_dataset
        from src.data_pipeline import DataPipeline
        
        print("✅ Modules imported successfully")
        
        # Test config loading
        config = load_config()
        print(f"✅ Config loaded - {len(config['moods'])} moods defined")
        
        # Test sample data creation
        df = create_sample_dataset(100)
        print(f"✅ Sample data created - {len(df)} tracks")
        
        # Test data pipeline
        pipeline = DataPipeline()
        pipeline.df_raw = df
        df_clean = pipeline.clean_data()
        print(f"✅ Data cleaning works - {len(df_clean)} tracks after cleaning")
        
        df_labeled = pipeline.assign_mood_labels(df_clean)
        print(f"✅ Mood labeling works")
        
        for mood_name, mood_info in config['moods'].items():
            count = df_labeled[f'is_{mood_name}'].sum()
            print(f"   {mood_info['emoji']} {mood_info['name']}: {count} tracks")
        
        print("\n✅ All tests passed! System is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("🎵 VIBE-SYNC INSTALLATION TEST")
    print("="*60 + "\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️ Some packages are missing. Install them with:")
        print("pip install numpy pandas scikit-learn lightgbm matplotlib seaborn pyyaml")
        return
    
    # Test structure
    structure_ok = test_project_structure()
    
    if not structure_ok:
        print("\n⚠️ Project structure is incomplete")
        return
    
    # Run functionality test
    test_ok = quick_test()
    
    if test_ok:
        print("\n" + "="*60)
        print("🎉 INSTALLATION SUCCESSFUL!")
        print("="*60)
        print("\n📚 Next steps:")
        print("1. Run full pipeline: python run.py --mode full --use-sample")
        print("2. Or launch demo: python app/main.py")
        print("3. Read SETUP.md for detailed instructions")
        print("\n" + "="*60)
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")


if __name__ == "__main__":
    main()
