"""
Evaluate trained classification models for multiple tickers.
"""
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import DEFAULT_TICKERS, INDICATORS_DIR
from utils import load_model

logger = get_logger(__name__)


def evaluate_classification(y_true, y_pred):
    """
    Compute standard classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, ticker, save_dir=None):
    """
    Plot confusion matrix for a single ticker.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{ticker} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{ticker}_confusion_matrix.png"
        plt.savefig(path, bbox_inches="tight")
        logger.info(f"Saved confusion matrix → {path}")
    else:
        plt.show()

    plt.close()


def evaluate_all_tickers():
    """
    Evaluate all trained models (one per ticker).
    """
    summary = []

    for ticker in DEFAULT_TICKERS:
        logger.info(f"🔍 Evaluating model for {ticker}")

        try:
            model, scaler, features = load_model(ticker)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        data_path = INDICATORS_DIR / f"{ticker}_features.csv"
        if not data_path.exists():
            logger.warning(f"Missing data file for {ticker}")
            continue

        data = pd.read_csv(
            data_path,
            index_col=0,
            parse_dates=True
        )

        if "target" not in data.columns:
            logger.warning(f"No target column for {ticker}")
            continue

        X = data[features]
        y_true = data["target"]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        metrics = evaluate_classification(y_true, y_pred)

        logger.info(
            f"{ticker} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"Prec: {metrics['precision']:.3f} | "
            f"Rec: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1']:.3f}"
        )

        plot_confusion_matrix(
            y_true,
            y_pred,
            ticker,
            save_dir=Path("reports/confusion_matrices")
        )

        metrics["ticker"] = ticker
        summary.append(metrics)

    return pd.DataFrame(summary)


if __name__ == "__main__":
    summary_df = evaluate_all_tickers()

    if not summary_df.empty:
        print("\n📊 Model Evaluation Summary")
        print(summary_df.set_index("ticker").round(3))
