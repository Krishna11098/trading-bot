"""
Train machine learning models for price direction prediction.
Uses Kaggle data with 2015-2023 training and 2024 testing.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost may not be installed in all environments
    XGBClassifier = None

from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_TICKERS,
    INDICATORS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    N_ESTIMATORS,
)
from src.utils.data_split import split_data_by_date, get_date_range
from src.data_collection.load_kaggle_data import load_kaggle_data
from src.preprocessing.clean_data import clean_ohlcv_data

logger = get_logger(__name__)


def prepare_features_with_date_split(
    data: pd.DataFrame,
    train_start: str = "2015-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
    target_column: str = "target",
):
    """
    Prepare features and target for time-series model training with date-based split.
    
    Args:
        data: DataFrame with datetime index containing OHLCV and indicator data
        train_start: Training period start date
        train_end: Training period end date
        test_start: Testing period start date
        test_end: Testing period end date
        target_column: Name of target column to create/use
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols)
    """
    df = data.dropna().copy()

    # Create target if not present (1: price goes up, 0: price goes down)
    if target_column not in df.columns:
        df[target_column] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)

    # Split data by date
    train_data, test_data = split_data_by_date(
        df, 
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    # Extract features (numeric columns, exclude target)
    feature_cols = (
        train_data.select_dtypes(include=[np.number])
        .columns.difference([target_column])
        .tolist()
    )

    X_train = train_data[feature_cols]
    y_train = train_data[target_column]
    
    X_test = test_data[feature_cols]
    y_test = test_data[target_column]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        f"Prepared data with date-based split | "
        f"Train: {X_train.shape} ({X_train.index[0]} to {X_train.index[-1]}) | "
        f"Test: {X_test.shape} ({X_test.index[0]} to {X_test.index[-1]})"
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, train_data.index, test_data.index


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = N_ESTIMATORS,
):
    """
    Train Random Forest classifier.
    """
    logger.info("Training Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest training completed")
    return model


def train_stacked_model(
    X_train,
    y_train,
    rf_estimators: int = N_ESTIMATORS,
    xgb_estimators: int = 100,
):
    """
    Train a stacked classifier combining RandomForest and XGBoost with a
    LogisticRegression meta-estimator. Falls back to RandomForest-only if
    XGBoost is not available.
    """
    logger.info("Training stacked model (RandomForest + XGBoost)...")

    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    estimators = [("rf", rf)]

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=xgb_estimators,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        estimators.append(("xgb", xgb))
    else:
        logger.warning("XGBoost not available; falling back to RandomForest only in stacking.")

    final_est = LogisticRegression(max_iter=1000)

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=False,
        n_jobs=-1,
    )

    stacker.fit(X_train, y_train)

    logger.info("Stacked model training completed")
    return stacker


def save_model(model, scaler, feature_cols, ticker: str):
    """
    Save trained model artifacts.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / f"{ticker}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{ticker}_scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / f"{ticker}_features.pkl")

    logger.info(f"Saved model artifacts for {ticker}")


def train_improved_stacked_model(
    X_train,
    y_train,
    rf_estimators: int = 200,
    xgb_estimators: int = 200,
    gb_estimators: int = 100,
):
    """
    Train an improved stacked classifier combining RandomForest, XGBoost, and 
    GradientBoosting with optimized hyperparameters for better accuracy.
    """
    logger.info("Training improved stacked model (RF + XGB + GB)...")

    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    estimators = [("rf", rf)]

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        estimators.append(("xgb", xgb))
    else:
        logger.warning("XGBoost not available; using RF and GB only.")

    gb = GradientBoostingClassifier(
        n_estimators=gb_estimators,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    estimators.append(("gb", gb))

    final_est = LogisticRegression(max_iter=1000)

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=False,
        n_jobs=-1,
    )

    stacker.fit(X_train, y_train)

    logger.info("Improved stacked model training completed")
    return stacker


def tune_xgboost_hyperparameters(X_train, y_train):
    """
    Use GridSearchCV to find optimal XGBoost hyperparameters.
    Returns the best estimator and its CV score.
    """
    if XGBClassifier is None:
        logger.error("XGBoost not installed, skipping hyperparameter tuning")
        return None, None

    logger.info("Starting XGBoost hyperparameter tuning...")

    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150, 200],
        'subsample': [0.7, 0.8, 0.9],
    }

    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    xgb_grid.fit(X_train, y_train)
    
    logger.info(f"Best XGBoost parameters: {xgb_grid.best_params_}")
    logger.info(f"Best CV score: {xgb_grid.best_score_:.4f}")

    return xgb_grid.best_estimator_, xgb_grid.best_score_


def train_tuned_stacked_model(X_train, y_train, X_test=None, y_test=None):
    """
    Train a stacked model using tuned XGBoost hyperparameters.
    Optionally returns accuracy comparison if test data is provided.
    """
    logger.info("Training stacked model with tuned hyperparameters...")

    # Get tuned XGBoost
    best_xgb, _ = tune_xgboost_hyperparameters(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    estimators = [("rf", rf)]

    if best_xgb is not None:
        estimators.append(("xgb", best_xgb))

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    estimators.append(("gb", gb))

    final_est = LogisticRegression(max_iter=1000)

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=False,
        n_jobs=-1,
    )

    stacker.fit(X_train, y_train)

    logger.info("Tuned stacked model training completed")

    # Return accuracy metrics if test data provided
    if X_test is not None and y_test is not None:
        y_pred = stacker.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Tuned Model Test Accuracy: {accuracy:.4f}")
        return stacker, accuracy
    
    return stacker, None



def train_all_tickers():
    """
    Train models for all configured tickers using Kaggle data.
    Train on 2015-2023 data, test on 2024 data.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for ticker in DEFAULT_TICKERS:
        logger.info(f"🚀 Training model for {ticker}")
        
        try:
            # Load raw Kaggle data
            raw_data = load_kaggle_data(ticker)
            
            # Clean data
            cleaned_data = clean_ohlcv_data(raw_data)

            # Skip ticker if cleaning produced an empty dataframe
            if cleaned_data.empty:
                logger.warning(f"Cleaned data for {ticker} is empty — skipping training")
                results[ticker] = {"status": "skipped", "reason": "empty_cleaned_data"}
                continue
            
            # Add technical indicators (simplified - add basic features)
            # Note: For full features, you would use feature_engineering.py
            cleaned_data = add_basic_features(cleaned_data)
            
            # Prepare features with date-based split (2015-2023 train, 2024 test)
            X_train, X_test, y_train, y_test, scaler, features, train_idx, test_idx = prepare_features_with_date_split(
                cleaned_data
            )
            
            if len(X_train) == 0:
                logger.warning(f"No training data available for {ticker}, skipping")
                continue
            
            if len(X_test) == 0:
                logger.warning(f"No test data available for {ticker}, skipping")
                continue
            
            # Train stacked model
            try:
                model, accuracy = train_tuned_stacked_model(X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f"Stacked training failed for {ticker}, falling back to RandomForest: {e}")
                model, accuracy = train_random_forest_with_accuracy(X_train, y_train, X_test, y_test)
            
            # Save model and metadata
            save_model(model, scaler, features, ticker)
            
            results[ticker] = {
                "status": "success",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "test_accuracy": accuracy,
            }
            
            logger.info(f"✅ Successfully trained model for {ticker}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train model for {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
    
    logger.info(f"\nTraining Summary:\n{results}")
    return results


def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators if not already present.
    """
    df = data.copy()
    
    if "Close" in df.columns:
        # Moving averages
        if "SMA_20" not in df.columns:
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
        if "SMA_50" not in df.columns:
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
        
        # Returns
        if "returns" not in df.columns:
            df["returns"] = df["Close"].pct_change()
        
        # Volatility
        if "volatility_20" not in df.columns:
            df["volatility_20"] = df["returns"].rolling(window=20).std()
    
    return df


def train_random_forest_with_accuracy(X_train, y_train, X_test=None, y_test=None):
    """
    Train Random Forest and optionally return accuracy if test data is provided.
    """
    logger.info("Training Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest training completed")
    
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Random Forest Test Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    return model, None


if __name__ == "__main__":
    train_all_tickers()
