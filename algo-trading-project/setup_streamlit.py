#!/usr/bin/env python
"""
Quick Setup Script for Streamlit Dashboard
Run this to install dependencies and verify setup
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Streamlit and dependencies"""
    print("=" * 60)
    print("Installing Streamlit Dependencies...")
    print("=" * 60)
    
    requirements_file = Path(__file__).parent / "requirements-streamlit.txt"
    
    if not requirements_file.exists():
        print("❌ requirements-streamlit.txt not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify all required packages are installed"""
    print("\n" + "=" * 60)
    print("Verifying Installation...")
    print("=" * 60)
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'xgboost',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def check_project_structure():
    """Verify project structure"""
    print("\n" + "=" * 60)
    print("Checking Project Structure...")
    print("=" * 60)
    
    required_dirs = [
        'src',
        'src/data_collection',
        'src/preprocessing',
        'src/utils',
        'data',
        'notebooks'
    ]
    
    required_files = [
        'app.py',
        '.streamlit/config.toml'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = Path(__file__).parent / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NOT FOUND")
            all_good = False
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_good = False
    
    if all_good:
        print("\n✅ Project structure is correct!")
    else:
        print("\n⚠️  Some project files are missing. Please check your setup.")
    
    return all_good

def main():
    """Main setup routine"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  AlgoTrading Bot - Streamlit Dashboard Setup".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n⚠️  Setup incomplete. Please install dependencies manually.")
        return False
    
    # Step 2: Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages are missing. Please try installing again.")
        return False
    
    # Step 3: Check project structure
    check_project_structure()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("Setup Complete! ✅")
    print("=" * 60)
    print("\nTo start the Streamlit dashboard, run:\n")
    print("  On Windows:")
    print("    • Double-click: start_streamlit.bat")
    print("    • Or run: streamlit run app.py\n")
    print("  On Mac/Linux:")
    print("    • streamlit run app.py\n")
    print("Then open your browser to: http://localhost:8501\n")
    print("For more information, see STREAMLIT_README.md\n")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
