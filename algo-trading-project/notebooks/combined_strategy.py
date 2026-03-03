#!/usr/bin/env python
# coding: utf-8

# In[1]:


# combined_strategy.py
# --------------------------------------------------
# Core imports
# --------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# --------------------------------------------------
# Live / intraday imports
# --------------------------------------------------

import yfinance as yf
from datetime import datetime, timedelta
import pytz


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Internal project imports
# --------------------------------------------------

from src.data_collection.load_kaggle_data import load_kaggle_data
from src.preprocessing.clean_data import clean_ohlcv_data
from src.utils.data_split import split_data_by_date
from src.utils.config import (
    DEFAULT_TICKERS,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)

# --------------------------------------------------
# ML imports
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier
import joblib


# In[2]:


def add_scalping_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rule-based scalping signals using:
    - RSI
    - SMA trend
    - MACD momentum
    - Near-high strength filter

    Output:
        strategy_signal:
            +1  -> Buy
            -1  -> Sell
             0  -> Neutral
    """
    df = data.copy()

    # ---------- RSI ----------
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    # ---------- Trend ----------
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()

    # ---------- MACD ----------
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # ---------- Buy Conditions ----------
    buy_uptrend = (df["Close"] > sma_20) & (sma_20 > sma_50)
    buy_rsi = (rsi > 30) & (rsi < 50)
    buy_macd = (macd > 0) & (macd_hist > 0)

    close_20_high = df["Close"].rolling(20).max()
    buy_strength = df["Close"] > 0.95 * close_20_high

    buy_signal = (
        (buy_uptrend & buy_rsi) |
        (buy_uptrend & buy_macd) |
        (buy_uptrend & buy_strength)
    )

    # ---------- Sell Conditions ----------
    sell_downtrend = (df["Close"] < sma_20) & (sma_20 < sma_50)
    sell_rsi = (rsi < 70) & (rsi > 50)
    sell_macd = (macd < 0) & (macd_hist < 0)

    sell_signal = (
        (sell_downtrend & sell_rsi) |
        (sell_downtrend & sell_macd)
    )

    # ---------- Final Signal ----------
    signal = pd.Series(0, index=df.index, dtype=np.int8)
    signal[buy_signal & ~sell_signal] = 1
    signal[sell_signal & ~buy_signal] = -1

    df["strategy_signal"] = signal
    return df

def add_basic_features(
    data: pd.DataFrame,
    horizon: int = 3,
    cost: float = 0.0003
) -> pd.DataFrame:
    """
    Adds base ML features:
    - Returns & volatility
    - Trend strength
    - Candle structure
    - RSI
    - Volume normalization
    - Binary classification target
    """
    df = data.copy()

    # ---------- Returns ----------
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # ---------- Trend ----------
    sma_10 = df["Close"].rolling(10).mean()
    sma_20 = df["Close"].rolling(20).mean()

    df["trend_10"] = (df["Close"] - sma_10) / (sma_10 + 1e-8)
    df["trend_20"] = (df["Close"] - sma_20) / (sma_20 + 1e-8)
    df["trend_diff"] = (sma_10 - sma_20) / (sma_20 + 1e-8)

    # ---------- Candle Structure ----------
    df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)
    df["body_pct"] = (df["Close"] - df["Open"]) / (df["Close"] + 1e-8)
    df["body_abs"] = df["body_pct"].abs()

    # ---------- Volatility ----------
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["vol_ratio"] = df["volatility_10"] / (
        df["volatility_10"].rolling(50).mean() + 1e-8
    )
    df["high_vol"] = (df["vol_ratio"] > 1.0).astype(np.int8)

    # ---------- RSI ----------
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI"] = (100 - (100 / (1 + rs))) / 100.0

    # ---------- Volume ----------
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol_sma = df["Volume"].rolling(20).mean()
        df["Volume_norm"] = np.log1p(df["Volume"] / (vol_sma + 1e-8))
    else:
        df["Volume_norm"] = 0.0

    # ---------- Target ----------
    future_return = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    df["target"] = (future_return > cost).astype(np.int8)

    df.dropna(inplace=True)
    return df



# In[3]:


def prepare_train_test_data(
    ticker: str = None,
    verbose: bool = False
):
    """
    Loads data for a ticker, cleans it, splits into train/test,
    and applies scalping signals + basic features.

    Returns:
        train_df : pd.DataFrame
        test_df  : pd.DataFrame
    """

    # ---------- Select ticker ----------
    if ticker is None:
        ticker = DEFAULT_TICKERS[0]

    if verbose:
        print("\n" + "=" * 80)
        print(f"ANALYZING {ticker}")
        print("=" * 80)

    # ---------- Load & clean ----------
    raw_data = load_kaggle_data(ticker)
    cleaned_data = clean_ohlcv_data(raw_data)

    # ---------- Split ----------
    train_data, test_data = split_data_by_date(cleaned_data)

    if verbose:
        print(f"Train data: {train_data.shape}")
        print(f"Test data:  {test_data.shape}")

    # ---------- Feature pipeline ----------
    train_df = add_basic_features(
        add_scalping_signals(train_data)
    )

    test_df = add_basic_features(
        add_scalping_signals(test_data)
    )

    if verbose:
        print(f"Train with features: {train_df.shape}")
        print(f"Test with features:  {test_df.shape}")

    return train_df, test_df


from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # ---------- Capital ----------
    initial_capital: float = 1_000_000
    risk_free_rate: float = 0.0

    # ---------- Trade Horizon ----------
    horizon: int = 9

    # ---------- Entry ----------
    entry_q: float = 0.35
    entry_threshold_min: float = 0.35

    # ---------- Position Sizing ----------
    max_position: float = 0.96
    min_position: float = 0.25
    size_exponent: float = 0.22

    # ---------- Risk Management ----------
    base_stop_loss: float = 0.024
    base_take_profit: float = 0.032
    trail_stop: float = 0.0025

    cost_per_trade: float = 0.000001

    # ---------- Trade Rules ----------
    cooldown: int = 0
    min_trades_per_day: int = 1

    require_uptrend: bool = False
    trend_strength_min: float = 0.0

    max_volatility: float = 2.0


# In[5]:


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced technical features for ML + strategy logic:
    - Multi-horizon momentum
    - ATR & volatility regime
    - Trend strength score
    - Price position & alignment signals
    """
    d = df.copy()

    # ---------- Momentum ----------
    d["momentum_5"] = (d["Close"] - d["Close"].shift(5)) / d["Close"].shift(5)
    d["momentum_10"] = (d["Close"] - d["Close"].shift(10)) / d["Close"].shift(10)
    d["momentum_20"] = (d["Close"] - d["Close"].shift(20)) / d["Close"].shift(20)

    d["momentum_accel"] = d["momentum_5"] - d["momentum_10"]

    # ---------- True Range & ATR ----------
    prev_close = d["Close"].shift(1)

    d["tr"] = np.maximum(
        d["High"] - d["Low"],
        np.maximum(
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs()
        )
    )

    d["atr"] = d["tr"].rolling(14).mean()
    d["atr_pct"] = d["atr"] / (d["Close"] + 1e-8)

    # ---------- Trend Strength ----------
    sma_10 = d["Close"].rolling(10).mean()
    sma_20 = d["Close"].rolling(20).mean()
    sma_50 = d["Close"].rolling(50).mean()

    d["trend_strength"] = (
        (d["Close"] > sma_20).astype(np.int8) * 0.4 +
        (sma_20 > sma_50).astype(np.int8) * 0.3 +
        (d["momentum_5"] > 0).astype(np.int8) * 0.3
    )

    # ---------- Volatility Regime ----------
    d["vol_20"] = d["Close"].pct_change().rolling(20).std()
    d["vol_regime"] = (
        d["vol_20"] > d["vol_20"].rolling(50).mean()
    ).astype(np.int8)

    # ---------- Price Position ----------
    d["high_20"] = d["High"].rolling(20).max()
    d["low_20"] = d["Low"].rolling(20).min()

    d["price_position"] = (
        d["Close"] - d["low_20"]
    ) / (d["high_20"] - d["low_20"] + 1e-8)

    # ---------- Composite Signals ----------
    d["momentum_trend_signal"] = d["momentum_5"] * d["trend_strength"]
    d["price_momentum_align"] = d["price_position"] * d["momentum_10"]

    return d



def prepare_backtest_dataset(
    ticker: str,
    verbose: bool = False
):
    """
    Prepares full feature dataset for backtesting:
    - Loads data
    - Cleans & splits
    - Builds full feature pipeline
    - Extracts test set only

    Returns:
        test_df       : pd.DataFrame
        feature_cols  : list[str]
        X_test        : pd.DataFrame
        y_test        : pd.Series
        prices        : np.ndarray
        atr_values    : np.ndarray
    """

    if verbose:
        print(f"\nBacktesting: {ticker}")

    # ---------- Load & clean ----------
    raw = load_kaggle_data(ticker)
    cleaned = clean_ohlcv_data(raw)

    train_data, test_data = split_data_by_date(cleaned)

    # ---------- Build full feature space ----------
    full_data = pd.concat([train_data, test_data], axis=0)

    full_features = add_advanced_features(
        add_basic_features(
            add_scalping_signals(full_data)
        )
    )

    # ---------- Extract test portion ----------
    test_df = full_features.loc[test_data.index]

    # ---------- Feature selection ----------
    feature_cols = [
        c for c in test_df.columns
        if c not in [
            "target",
            "Open", "High", "Low", "Close", "Volume",
            "strategy_signal",
            "tr", "high_20", "low_20"
        ]
    ]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    prices = test_df["Close"].values
    atr_values = test_df["atr"].values

    if verbose:
        print(f"Advanced features added: {len(feature_cols)} features total")

    return (
        test_df,
        feature_cols,
        X_test,
        y_test,
        prices,
        atr_values,
    )



# In[7]:

from lightgbm import LGBMClassifier


def train_ensemble_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = False,
):
    """
    Trains an ensemble ML model (XGBoost + LightGBM) and
    produces ensemble probabilities on the test set.

    Returns:
        ml_prob        : np.ndarray
        xgb_model      : trained XGBClassifier
        lgb_model      : trained LGBMClassifier
        scaler         : fitted StandardScaler
        metrics        : dict with AUC scores
    """

    # ---------- Split ----------
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # ---------- Scale ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print("\nTraining Ensemble Model (XGBoost + LightGBM)...")

    # ---------- XGBoost ----------
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    # ---------- LightGBM ----------
    lgb_model = LGBMClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.75,
        num_leaves=31,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective="binary",
        metric="auc",
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train_scaled, y_train)
    lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]

    # ---------- Ensemble ----------
    ml_prob = (xgb_proba * 0.55) + (lgb_proba * 0.45)

    # ---------- Metrics ----------
    metrics = {
        "xgb_auc": roc_auc_score(y_test, xgb_proba),
        "lgb_auc": roc_auc_score(y_test, lgb_proba),
        "ensemble_auc": roc_auc_score(y_test, ml_prob),
    }

    if verbose:
        print("Ensemble Model Ready")
        print(f"  XGBoost AUC:  {metrics['xgb_auc']:.4f}")
        print(f"  LightGBM AUC: {metrics['lgb_auc']:.4f}")
        print(f"  Ensemble AUC:{metrics['ensemble_auc']:.4f}")

    return ml_prob, xgb_model, lgb_model, scaler, metrics


def position_size(prob: float, threshold: float) -> float:
    """
    Computes normalized position size based on model confidence.

    Args:
        prob      : model probability (0–1)
        threshold : entry threshold (0–1)

    Returns:
        size in [0, 1]
    """
    size = (prob - threshold) / (1.0 - threshold)
    return float(np.clip(size, 0.0, 1.0))

def run_backtest(
    test_df: pd.DataFrame,
    prices: np.ndarray,
    atr_values: np.ndarray,
    ml_prob: np.ndarray,
    cfg: StrategyConfig,
    verbose: bool = False,
):
    """
    Runs the main backtest loop using ML probabilities and strategy rules.

    Returns:
        results: dict with capital, trades, equity curve, stats
    """

    capital = cfg.initial_capital
    equity_curve = []
    trades = []
    recent_returns = []

    # ---------- Entry Threshold ----------
    entry_threshold = np.quantile(ml_prob, cfg.entry_q)

    if verbose:
        print("\nML Ensemble Probability Distribution:")
        print(f"  Min: {ml_prob.min():.4f}")
        print(f"  25%: {np.percentile(ml_prob, 25):.4f}")
        print(f"  Median: {np.median(ml_prob):.4f}")
        print(f"  75%: {np.percentile(ml_prob, 75):.4f}")
        print(f"  95%: {np.percentile(ml_prob, 95):.4f}")
        print(f"  Max: {ml_prob.max():.4f}")
        print(f"\nEntry Threshold (Q={cfg.entry_q}): {entry_threshold:.4f}")
        print(f"Potential signals: {(ml_prob >= entry_threshold).sum()} / {len(ml_prob)}")

    i = 0
    n = len(prices)
    last_exit = -cfg.cooldown

    # ---------- Main Loop ----------
    while i < n - cfg.horizon:

        prob = ml_prob[i]
        current_price = prices[i]
        current_atr = atr_values[i]

        # Warm-up
        if i < 50:
            equity_curve.append(capital)
            i += 1
            continue

        # Cooldown
        if i - last_exit < cfg.cooldown:
            equity_curve.append(capital)
            i += 1
            continue

        # ML filter
        if prob < entry_threshold:
            equity_curve.append(capital)
            i += 1
            continue

        # Trend filter
        if cfg.require_uptrend:
            if test_df["trend_strength"].iloc[i] < 0.6:
                equity_curve.append(capital)
                i += 1
                continue

        # Momentum filter
        if test_df["momentum_5"].iloc[i] < cfg.trend_strength_min:
            equity_curve.append(capital)
            i += 1
            continue

        # Volatility regime filter
        vol_regime = test_df["vol_regime"].iloc[i]
        if vol_regime == 1 and prob < 0.55:
            equity_curve.append(capital)
            i += 1
            continue

        # ---------- Adaptive sizing ----------
        if recent_returns:
            recent_wins = sum(r > 0 for r in recent_returns[-20:])
            recent_win_rate = recent_wins / min(20, len(recent_returns))
            size_multiplier = 0.8 + (recent_win_rate * 0.4)
        else:
            size_multiplier = 1.0

        confidence_edge = prob - entry_threshold
        max_edge = 1.0 - entry_threshold
        normalized = confidence_edge / max_edge if max_edge > 0 else 0.5

        position_fraction = (
            cfg.min_position
            + (cfg.max_position - cfg.min_position)
            * (normalized ** cfg.size_exponent)
        )

        position_fraction = np.clip(
            position_fraction * size_multiplier,
            cfg.min_position,
            cfg.max_position,
        )

        position_value = capital * position_fraction
        entry_price = current_price
        max_price = current_price

        # ---------- Stops ----------
        vol_adjusted_stop = cfg.base_stop_loss + (current_atr / current_price) * 0.5
        vol_adjusted_stop = np.clip(
            vol_adjusted_stop,
            cfg.base_stop_loss,
            cfg.base_stop_loss * 1.5,
        )

        vol_adjusted_profit = cfg.base_take_profit * (0.8 + vol_regime * 0.4)

        exit_price = prices[i + cfg.horizon]
        exit_idx = i + cfg.horizon
        exit_reason = "TIME"

        # ---------- Trade management ----------
        for j in range(1, cfg.horizon + 1):
            price = prices[i + j]
            max_price = max(max_price, price)

            if price <= entry_price * (1 - vol_adjusted_stop):
                exit_price = entry_price * (1 - vol_adjusted_stop)
                exit_idx = i + j
                exit_reason = "STOP"
                break

            if price >= entry_price * (1 + vol_adjusted_profit):
                exit_price = entry_price * (1 + vol_adjusted_profit)
                exit_idx = i + j
                exit_reason = "PROFIT"
                break

            trail_level = max_price * (1 - cfg.trail_stop)
            if price < trail_level:
                exit_price = trail_level
                exit_idx = i + j
                exit_reason = "TRAIL"
                break

        # ---------- P&L ----------
        ret = (exit_price - entry_price) / entry_price
        net_ret = ret - (cfg.cost_per_trade * 2)

        pnl = position_value * net_ret
        capital += pnl

        if capital < 0:
            capital = cfg.initial_capital * 0.001

        recent_returns.append(net_ret)

        trades.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "prob": prob,
            "size": position_fraction,
            "return": net_ret,
            "pnl": pnl,
            "capital": capital,
            "exit_reason": exit_reason,
            "vol": test_df["volatility_10"].iloc[i],
            "trend_strength": test_df["trend_strength"].iloc[i],
        })

        equity_curve.append(capital)
        last_exit = exit_idx
        i = exit_idx + cfg.cooldown

    # ---------- Results ----------
    results = {
        "initial_capital": cfg.initial_capital,
        "final_capital": capital,
        "equity_curve": equity_curve,
        "trades": trades,
        "total_trades": len(trades),
        "entry_threshold": entry_threshold,
    }

    if verbose:
        print(f"\nAdvanced Backtest complete: {len(trades)} trades")
        print(f"Capital: {cfg.initial_capital:,.0f} → {capital:,.0f}")

    return results



def train_backtest_xgb_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = False,
):
    """
    Trains a single XGBoost model for backtesting
    with class imbalance handling.

    Returns:
        ml_prob     : np.ndarray
        model       : trained XGBClassifier
        scaler      : fitted StandardScaler
    """

    # ---------- Split ----------
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]

    # ---------- Scale ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Class imbalance ----------
    positives = y_train.sum()
    negatives = len(y_train) - positives
    scale_pos_weight = negatives / max(positives, 1)

    # ---------- Model ----------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train_scaled, y_train)

    # ---------- Inference ----------
    ml_prob = model.predict_proba(X_test_scaled)[:, 1]

    if verbose:
        print("Backtest XGBoost model trained")
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"  Prob range: {ml_prob.min():.4f} → {ml_prob.max():.4f}")

    return ml_prob, model, scaler




def compute_backtest_metrics(
    equity_curve: list | np.ndarray,
    trades: list[dict],
):
    """
    Computes performance metrics from equity curve and trades.

    Returns:
        metrics: dict with return, drawdown, sharpe, PF, win stats, etc.
    """

    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()

    # ---------- Core metrics ----------
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    net_profit = equity.iloc[-1] - equity.iloc[0]

    max_drawdown = ((equity / equity.cummax()) - 1).min()

    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60)
    else:
        sharpe = 0.0

    # ---------- Trade stats ----------
    winning_trades = [t["pnl"] for t in trades if t["pnl"] > 0]
    losing_trades = [t["pnl"] for t in trades if t["pnl"] < 0]

    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 1e-8

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    metrics = {
        "initial_capital": float(equity.iloc[0]),
        "final_capital": float(equity.iloc[-1]),
        "net_profit": float(net_profit),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "total_trades": total_trades,
        "equity_curve": equity.values,
    }

    return metrics




def fetch_live_intraday_data(
    ticker_primary="^NSEBANK",
    ticker_fallback="NIFTYBANK.NS",
    days: int = 7,
    interval: str = "1m",
    fallback_df: pd.DataFrame | None = None,
):
    """
    Fetches last N days of intraday data from yfinance.
    Falls back to provided dataframe if download fails.
    """
    IST = pytz.timezone("Asia/Kolkata")

    try:
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=days)

        try:
            data = yf.download(
                ticker_primary,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            ticker_used = ticker_primary
        except Exception:
            data = yf.download(
                ticker_fallback,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            ticker_used = ticker_fallback

        # Timezone handling
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        return data, ticker_used, False

    except Exception:
        if fallback_df is None:
            raise RuntimeError("Live data fetch failed and no fallback provided")

        return (
            fallback_df[["Open", "High", "Low", "Close", "Volume"]].copy(),
            "FALLBACK",
            True,
        )
def prepare_live_features_and_probs(
    live_data: pd.DataFrame,
    feature_cols: list[str],
    model,
    scaler,
):
    """
    Applies feature pipeline to live data and
    generates ML probabilities.
    """
    live_features = add_advanced_features(
        add_basic_features(
            add_scalping_signals(live_data)
        )
    ).dropna()

    X_live = live_features[feature_cols].copy()
    X_live_scaled = scaler.transform(X_live)

    ml_prob_live = model.predict_proba(X_live_scaled)[:, 1]

    prices = live_features["Close"].values
    atr = live_features["atr"].values

    return live_features, ml_prob_live, prices, atr
def run_live_paper_trading(
    live_features: pd.DataFrame,
    prices: np.ndarray,
    atr: np.ndarray,
    ml_prob: np.ndarray,
    cfg: StrategyConfig,
):
    """
    Runs live paper trading simulation on intraday data.
    """

    capital = cfg.initial_capital
    peak_capital = capital

    positions = []
    trades = []
    equity_curve = []

    entry_threshold = np.percentile(ml_prob, 75)

    for i in range(len(live_features)):
        current_price = prices[i]
        current_time = live_features.index[i]
        prob = ml_prob[i]
        current_atr = atr[i]

        if i < 30:
            equity_curve.append(capital)
            continue

        # ----- Manage open positions -----
        positions_to_close = []

        for idx, pos in enumerate(positions):
            price_move = (current_price - pos["entry_price"]) / pos["entry_price"]

            vol_stop = cfg.base_stop_loss * (1 + (current_atr / current_price) * 0.5)
            vol_stop = np.clip(vol_stop, cfg.base_stop_loss * 0.8, cfg.base_stop_loss * 1.2)

            exit_price = current_price
            exit_reason = None

            if price_move >= 0.004:
                exit_reason = "QUICK_PROFIT"
                exit_price = pos["entry_price"] * 1.004

            elif price_move >= 0.008:
                exit_reason = "MEDIUM_PROFIT"
                exit_price = pos["entry_price"] * 1.008

            elif price_move >= 0.015:
                exit_reason = "BIG_PROFIT"
                exit_price = pos["entry_price"] * 1.015

            elif price_move > 0.004 and current_price < pos["max_price"] * (1 - 0.003):
                exit_reason = "TRAIL"
                exit_price = pos["max_price"] * (1 - 0.003)

            elif price_move <= -vol_stop:
                exit_reason = "STOP"

            elif i - pos["entry_idx"] >= 20:
                exit_reason = "TIME"

            if exit_reason:
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                net_ret = ret - (cfg.cost_per_trade * 2)
                pnl = pos["position_value"] * net_ret

                capital += pnl
                peak_capital = max(peak_capital, capital)

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": current_time,
                    "pnl": pnl,
                    "return": net_ret,
                    "exit_reason": exit_reason,
                })

                positions_to_close.append(idx)

        positions = [p for j, p in enumerate(positions) if j not in positions_to_close]

        # ----- Entry logic -----
        if prob >= entry_threshold and i < len(live_features) - 20:
            momentum_5 = live_features["momentum_5"].iloc[i]

            if momentum_5 >= cfg.trend_strength_min:
                position_fraction = np.clip(0.15 + (prob - entry_threshold), 0.15, 0.50)
                position_value = capital * position_fraction

                positions.append({
                    "entry_idx": i,
                    "entry_price": current_price,
                    "entry_time": current_time,
                    "position_value": position_value,
                    "position_fraction": position_fraction,
                    "max_price": current_price,
                })

        for p in positions:
            p["max_price"] = max(p["max_price"], current_price)

        equity_curve.append(capital)

    return {
        "initial_capital": cfg.initial_capital,
        "final_capital": capital,
        "peak_capital": peak_capital,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def fetch_today_intraday_data(
    ticker="^NSEBANK",
    interval="1m",
    fallback_df: pd.DataFrame | None = None,
):
    IST = pytz.timezone("Asia/Kolkata")

    try:
        trade_date = datetime.now(IST).date()
        start_time = IST.localize(datetime.combine(trade_date, datetime.min.time()))
        end_time = IST.localize(datetime.combine(trade_date, datetime.max.time()))

        data = yf.download(
            ticker,
            start=start_time,
            end=end_time,
            interval=interval,
            progress=False,
        )

        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return data, False

    except Exception:
        if fallback_df is None:
            raise RuntimeError("Failed to fetch today data and no fallback provided")

        fallback = fallback_df[["Open", "High", "Low", "Close", "Volume"]].iloc[-390:].copy()
        fallback.index = pd.date_range(
            start=datetime.now() - timedelta(hours=6),
            periods=len(fallback),
            freq="1min",
        )
        return fallback, True

def run_today_trading(
    features: pd.DataFrame,
    prices: np.ndarray,
    atr: np.ndarray,
    ml_prob: np.ndarray,
    initial_capital: float,
    cfg: dict,
):
    capital = initial_capital
    peak_capital = capital

    positions = []
    trades = []
    equity_curve = []

    entry_threshold = np.percentile(ml_prob, cfg["ENTRY_PCTL"])

    for i in range(len(features)):
        price = prices[i]
        time = features.index[i]
        prob = ml_prob[i]
        current_atr = atr[i]

        if i < cfg["WARMUP"]:
            equity_curve.append(capital)
            continue

        # ---- exits ----
        closed = []
        for idx, pos in enumerate(positions):
            move = (price - pos["entry_price"]) / pos["entry_price"]

            vol_stop = cfg["STOP_LOSS"] * (1 + (current_atr / price) * 0.3)
            vol_stop = np.clip(vol_stop, cfg["STOP_LOSS"] * 0.8, cfg["STOP_LOSS"] * 1.2)

            exit_reason = None
            exit_price = price

            if move >= cfg["TP1"]:
                exit_reason = "QUICK_PROFIT"
                exit_price = pos["entry_price"] * (1 + cfg["TP1"])
            elif move >= cfg["TP2"]:
                exit_reason = "MEDIUM_PROFIT"
                exit_price = pos["entry_price"] * (1 + cfg["TP2"])
            elif move >= cfg["TP3"]:
                exit_reason = "BIG_PROFIT"
                exit_price = pos["entry_price"] * (1 + cfg["TP3"])
            elif move > cfg["TP1"] and price < pos["max_price"] * (1 - cfg["TRAIL"]):
                exit_reason = "TRAIL"
                exit_price = pos["max_price"] * (1 - cfg["TRAIL"])
            elif move <= -vol_stop:
                exit_reason = "STOP"
            elif i - pos["entry_idx"] >= cfg["HORIZON"]:
                exit_reason = "TIME"

            if exit_reason:
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                net_ret = ret - (cfg["COST"] * 2)
                pnl = pos["value"] * net_ret

                capital += pnl
                peak_capital = max(peak_capital, capital)

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": time,
                    "pnl": pnl,
                    "return": net_ret,
                    "exit_reason": exit_reason,
                })
                closed.append(idx)

        positions = [p for j, p in enumerate(positions) if j not in closed]

        # ---- entries ----
        if prob >= entry_threshold and i < len(features) - cfg["HORIZON"]:
            momentum_5 = features["momentum_5"].iloc[i]
            if momentum_5 >= cfg["MIN_MOMENTUM"]:
                frac = np.clip(
                    cfg["MIN_POS"] +
                    (cfg["MAX_POS"] - cfg["MIN_POS"]) * ((prob - entry_threshold) ** 1.3),
                    cfg["MIN_POS"],
                    cfg["MAX_POS"],
                )

                positions.append({
                    "entry_idx": i,
                    "entry_price": price,
                    "entry_time": time,
                    "value": capital * frac,
                    "max_price": price,
                })

        for p in positions:
            p["max_price"] = max(p["max_price"], price)

        equity_curve.append(capital)

    return {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "peak_capital": peak_capital,
        "equity_curve": equity_curve,
        "trades": trades,
    }
