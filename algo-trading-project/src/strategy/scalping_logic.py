"""
Improved scalping signal generation (vote-based).
"""

import pandas as pd
import numpy as np
from src.utils.config import RSI_OVERSOLD, RSI_OVERBOUGHT

def calculate_scalping_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scalp_score"] = 0

    # ---------- RSI ----------
    if "RSI" in df.columns:
        df.loc[df["RSI"] <= RSI_OVERSOLD, "scalp_score"] += 1
        df.loc[df["RSI"] >= RSI_OVERBOUGHT, "scalp_score"] -= 1

    # ---------- Bollinger Bands ----------
    if {"BBL_20_2.0", "BBU_20_2.0"}.issubset(df.columns):
        df.loc[df["Close"] <= df["BBL_20_2.0"], "scalp_score"] += 1
        df.loc[df["Close"] >= df["BBU_20_2.0"], "scalp_score"] -= 1

    # ---------- MACD ----------
    if {"MACD_12_26_9", "MACDs_12_26_9"}.issubset(df.columns):
        bullish = (
            (df["MACD_12_26_9"] > df["MACDs_12_26_9"]) &
            (df["MACD_12_26_9"].shift(1) <= df["MACDs_12_26_9"].shift(1))
        )
        bearish = (
            (df["MACD_12_26_9"] < df["MACDs_12_26_9"]) &
            (df["MACD_12_26_9"].shift(1) >= df["MACDs_12_26_9"].shift(1))
        )

        df.loc[bullish, "scalp_score"] += 1
        df.loc[bearish, "scalp_score"] -= 1

    # ---------- Final signal ----------
    df["scalp_signal"] = np.where(
        df["scalp_score"] >= 2, 1,
        np.where(df["scalp_score"] <= -2, -1, 0)
    )

    return df
