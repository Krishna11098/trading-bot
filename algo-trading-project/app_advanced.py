#!/usr/bin/env python
"""
Advanced Streamlit Dashboard - Multi-Ticker Comparison
Extended version with additional features for comparison analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Setup
st.set_page_config(
    page_title="AlgoTrading Bot - Advanced Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
try:
    from src.data_collection.load_kaggle_data import load_kaggle_data
    from src.preprocessing.clean_data import clean_ohlcv_data
    from src.utils.config import DEFAULT_TICKERS
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

@st.cache_data
def load_data(ticker):
    """Load and cache data for a ticker"""
    try:
        data = load_kaggle_data(ticker)
        return clean_ohlcv_data(data)
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

def add_scalping_signals(data):
    """Add rule-based scalping signals"""
    df = data.copy()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # Trend
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()
    
    # MACD
    ema_12 = df["Close"].ewm(span=12).mean()
    ema_26 = df["Close"].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    
    # Signals
    buy_uptrend = (df["Close"] > sma_20) & (sma_20 > sma_50)
    buy_rsi = (rsi > 30) & (rsi < 50)
    buy_macd = (macd > 0) & (macd_hist > 0)
    buy_strong = buy_uptrend & buy_rsi & buy_macd
    
    sell_downtrend = (df["Close"] < sma_20) & (sma_20 < sma_50)
    sell_rsi = (rsi > 50) & (rsi < 70)
    sell_macd = (macd < 0) & (macd_hist < 0)
    sell_strong = sell_downtrend & sell_rsi & sell_macd
    
    strategy_signal = np.where(buy_strong, 1, np.where(sell_strong, -1, 0))
    
    df['RSI'] = rsi
    df['SMA_20'] = sma_20
    df['SMA_50'] = sma_50
    df['MACD'] = macd
    df['Signal'] = strategy_signal
    
    return df

# Title
st.title("📊 Advanced AlgoTrading Dashboard")
st.markdown("Multi-ticker comparison and portfolio analysis")

# Sidebar
with st.sidebar:
    st.header("Dashboard Settings")
    
    page = st.radio(
        "Select View",
        ["Single Ticker", "Multi-Ticker Comparison", "Portfolio Overview", "Performance Report"]
    )
    
    selected_tickers = st.multiselect(
        "Select Tickers",
        DEFAULT_TICKERS,
        default=[DEFAULT_TICKERS[0]] if DEFAULT_TICKERS else [],
        help="Choose one or more tickers to analyze"
    )

# --------------------------------------------------
# PAGE 1: Single Ticker
# --------------------------------------------------
if page == "Single Ticker":
    if not selected_tickers:
        st.warning("Please select at least one ticker from the sidebar")
    else:
        ticker = selected_tickers[0]
        st.header(f"Detailed Analysis - {ticker}")
        
        try:
            data = load_data(ticker)
            if data is not None:
                data = add_scalping_signals(data)
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                latest_price = data['Close'].iloc[-1]
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]) * 100
                rsi = data['RSI'].iloc[-1]
                signal = data['Signal'].iloc[-1]
                
                with col1:
                    st.metric("Current Price", f"${latest_price:.2f}", f"{change:+.2f}%")
                with col2:
                    st.metric("RSI", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
                with col3:
                    signal_text = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ NEUTRAL"
                    st.metric("Signal", signal_text)
                with col4:
                    st.metric("Volatility", f"{data['Close'].pct_change().std():.2%}")
                
                st.divider()
                
                # Chart
                col_charts = st.columns(1)[0]
                with col_charts:
                    # Price chart with signals
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price', line=dict(color='black')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', dash='dash')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))
                    
                    buy = data[data['Signal'] == 1]
                    sell = data[data['Signal'] == -1]
                    if not buy.empty:
                        fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='Buy', marker=dict(size=8, color='green')))
                    if not sell.empty:
                        fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='Sell', marker=dict(size=8, color='red')))
                    
                    fig.update_layout(title=f'{ticker} Price Chart', height=500, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
        except:
            pass        
# --------------------------------------------------
# PAGE 2: Multi-Ticker Comparison
# --------------------------------------------------
elif page == "Multi-Ticker Comparison":
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 tickers to compare")
    else:
        st.header("Multi-Ticker Comparison")
        
        # Normalize prices for comparison
        comparison_data = {}
        for ticker in selected_tickers:
            try:
                data = load_data(ticker)
                if data is not None:
                    comparison_data[ticker] = data['Close']
            except:
                pass
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_normalized = df_comparison / df_comparison.iloc[0] * 100
            
            # Normalized Price Comparison
            fig = go.Figure()
            for ticker in df_normalized.columns:
                fig.add_trace(go.Scatter(x=df_normalized.index, y=df_normalized[ticker], mode='lines', name=ticker))
            
            fig.update_layout(
                title='Normalized Price Comparison (Base = 100)',
                xaxis_title='Date',
                yaxis_title='Indexed Price',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics Comparison
            st.subheader("Performance Statistics")
            stats_data = []
            for ticker in selected_tickers:
                try:
                    data = load_data(ticker)
                    if data is not None:
                        returns = data['Close'].pct_change()
                        stats_data.append({
                            'Ticker': ticker,
                            'Return (%)': f"{(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100:.2f}",
                            'Volatility (%)': f"{returns.std() * 100:.2f}",
                            'Sharpe Ratio': f"{returns.mean() / returns.std() * np.sqrt(252):.2f}" if returns.std() > 0 else 'N/A'
                        })
                except:
                    pass
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# --------------------------------------------------
# PAGE 3: Portfolio Overview
# --------------------------------------------------
elif page == "Portfolio Overview":
    st.header("Portfolio Overview")
    
    # Allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Allocation")
        weights = {ticker: 1/len(selected_tickers) * 100 for ticker in selected_tickers}
        fig = px.pie(values=list(weights.values()), names=list(weights.keys()), title="Equal Weight Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Current Signals Summary")
        signals_data = []
        for ticker in selected_tickers:
            try:
                data = load_data(ticker)
                if data is not None:
                    data = add_scalping_signals(data)
                    signal = data['Signal'].iloc[-1]
                    signals_data.append({
                        'Ticker': ticker,
                        'Signal': '🟢 BUY' if signal == 1 else '🔴 SELL' if signal == -1 else '⚪ NEUTRAL',
                        'Price': f"${data['Close'].iloc[-1]:.2f}",
                        'RSI': f"{data['RSI'].iloc[-1]:.1f}"
                    })
            except:
                pass
        
        if signals_data:
            st.dataframe(pd.DataFrame(signals_data), use_container_width=True, hide_index=True)

# --------------------------------------------------
# PAGE 4: Performance Report
# --------------------------------------------------
elif page == "Performance Report":
    st.header("Performance Report")
    
    report_data = []
    for ticker in selected_tickers:
        try:
            data = load_data(ticker)
            if data is not None:
                data = add_scalping_signals(data)
                returns = data['Close'].pct_change()
                
                report_data.append({
                    'Ticker': ticker,
                    'Latest Price': f"${data['Close'].iloc[-1]:.2f}",
                    'Min (52w)': f"${data['Close'].tail(252).min():.2f}",
                    'Max (52w)': f"${data['Close'].tail(252).max():.2f}",
                    'Avg Volume': f"{data['Volume'].mean():.0f}",
                    'Total Return': f"{(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100:.2f}%",
                    'Annual Return': f"{returns.mean() * 252 * 100:.2f}%"
                })
        except:
            pass
    
    if report_data:
        st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

st.divider()
st.markdown("**Advanced Dashboard** | Last Updated: Jan 30, 2026")
