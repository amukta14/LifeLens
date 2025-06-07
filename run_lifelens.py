#!/usr/bin/env python3
"""
LifeLens Startup Script
Simple script to run the LifeLens backend API server
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the LifeLens backend server"""
    print("🔬 Starting LifeLens: Predictive Health and Survival Insight System")
    print("=" * 60)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        print("Please make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_python = None
    possible_venv_paths = [
        Path("venv/bin/python"),
        Path("env/bin/python"), 
        Path(".venv/bin/python"),
        Path("venv/Scripts/python.exe"),  # Windows
        Path("env/Scripts/python.exe"),   # Windows
        Path(".venv/Scripts/python.exe")  # Windows
    ]
    
    for venv_path in possible_venv_paths:
        if venv_path.exists():
            venv_python = str(venv_path)
            print(f"✅ Found virtual environment: {venv_python}")
            break
    
    if not venv_python:
        venv_python = sys.executable
        print(f"⚠️  No virtual environment found, using system Python: {venv_python}")
        print("💡 Tip: Create a virtual environment for better dependency management:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   pip install -r backend/requirements.txt")
        print()
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(backend_dir)
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    print("🚀 Starting FastAPI server...")
    print("📊 API Documentation will be available at: http://localhost:8001/docs")
    print("🔍 Health check endpoint: http://localhost:8001/health")
    print("📈 Metrics endpoint: http://localhost:8001/metrics")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run the server
        subprocess.run([
            venv_python, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure all dependencies are installed:")
        print("   pip install -r backend/requirements.txt")
        print("2. Check if port 8000 is already in use")
        print("3. Review the error messages above")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 