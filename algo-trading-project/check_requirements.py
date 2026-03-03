"""
Check all required packages are installed.
Run this before starting training.
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Required packages
REQUIRED_PACKAGES = {
    "pandas": "Data manipulation",
    "numpy": "Numerical computing",
    "scikit-learn": "ML algorithms",
    "joblib": "Model serialization",
    "pathlib": "Path handling",
}

# Optional packages
OPTIONAL_PACKAGES = {
    "xgboost": "XGBoost models (optional, fallback to RandomForest)",
    "pandas_ta": "Technical analysis (optional, can add manually)",
}


def check_package(package_name: str, is_optional: bool = False) -> bool:
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def main():
    """Check all required and optional packages."""
    
    logger.info("="*70)
    logger.info("PACKAGE REQUIREMENTS CHECK")
    logger.info("="*70)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"\nPython Version: {python_version}")
    
    if sys.version_info.major < 3 or sys.version_info.minor < 8:
        logger.error("❌ Python 3.8+ required")
        return False
    else:
        logger.info("✅ Python version OK")
    
    # Check required packages
    logger.info(f"\n{'Required Packages':<50} {'Status':<10}")
    logger.info("-" * 70)
    
    all_required_ok = True
    
    for package, description in REQUIRED_PACKAGES.items():
        if check_package(package):
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                logger.info(f"{package:<50} ✅ ({version})")
            except:
                logger.info(f"{package:<50} ✅")
        else:
            logger.error(f"{package:<50} ❌ MISSING")
            all_required_ok = False
    
    # Check optional packages
    logger.info(f"\n{'Optional Packages':<50} {'Status':<10}")
    logger.info("-" * 70)
    
    for package, description in OPTIONAL_PACKAGES.items():
        if check_package(package):
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                logger.info(f"{package:<50} ✅ ({version})")
                logger.info(f"  {description}")
            except:
                logger.info(f"{package:<50} ✅")
                logger.info(f"  {description}")
        else:
            logger.warning(f"{package:<50} ⚠️  NOT FOUND")
            logger.info(f"  {description}")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    
    if all_required_ok:
        logger.info("\n✅ All required packages are installed!")
        logger.info("\n🚀 You're ready to run:")
        logger.info("   python validate_data.py      (Check data setup)")
        logger.info("   python run_pipeline.py        (Train & backtest)")
        return True
    else:
        logger.error("\n❌ Some required packages are missing!")
        logger.info("\nInstall missing packages with:")
        logger.info("   pip install pandas numpy scikit-learn joblib")
        logger.info("\nOptional:")
        logger.info("   pip install xgboost pandas_ta")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
