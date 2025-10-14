import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')
try:
    from backend.server import app
    print('✅ Server imported successfully')
except Exception as e:
    print(f'❌ Server import failed: {e}')
    import traceback
    traceback.print_exc()
