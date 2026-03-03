#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Web Application for Algo Trading Bot - Combined Strategy
Dashboard for monitoring ensemble ML + scalping strategy backtesting and live trading
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
import yfinance as yf
import pytz

warnings.filterwarnings('ignore')

# --------------------------------------------------
# Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Combined Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
INITIAL_CAPITAL = 1_000_000
TREND_STRENGTH_MIN = 0.5

# Add project to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
try:
    from src.data_collection.load_kaggle_data import load_kaggle_data
    from src.preprocessing.clean_data import clean_ohlcv_data
    from src.utils.data_split import split_data_by_date
    from src.utils.config import DEFAULT_TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
    import joblib
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# --------------------------------------------------
# Styling & Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary: #667eea;
        --primary-dark: #764ba2;
        --accent: #00d9ff;
        --accent-bright: #00f0ff;
        --success: #00ff88;
        --warning: #ffa500;
        --danger: #ff5555;
        --bg-primary: #0a0f1f;
        --bg-secondary: #1a2a5e;
        --bg-tertiary: #142257;
        --text-primary: #ffffff;
        --text-secondary: #e0e8f0;
        --text-tertiary: #a8b4c4;
        --border: rgba(0, 217, 255, 0.15);
    }
    
    /* ===== MAIN LAYOUT ===== */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
        background-attachment: fixed;
    }
    
    body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--text-primary);
        font-size: 15px;
        line-height: 1.6;
        letter-spacing: 0.3px;
    }
    
    .main {
        background: transparent;
        padding: 24px;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.8px;
        line-height: 1.2;
        color: var(--text-primary);
        margin: 24px 0 14px 0;
    }
    
    h1 {
        font-size: 42px;
        color: var(--accent-bright);
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
        margin-bottom: 32px;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    h2 {
        font-size: 32px;
        color: var(--accent-bright);
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        margin-bottom: 22px;
        padding-bottom: 12px;
        border-bottom: 3px solid var(--accent);
        font-weight: 800;
    }
    
    h3 {
        font-size: 24px;
        color: var(--accent);
        margin-bottom: 16px;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(0, 217, 255, 0.2);
    }
    
    h4 {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
    }
    
    p, .stMarkdown, .stWrite {
        font-size: 15px;
        color: var(--text-primary);
        line-height: 1.8;
        letter-spacing: 0.3px;
        font-weight: 400;
    }
    
    /* List styling */
    ul, ol {
        color: var(--text-primary);
        line-height: 2;
    }
    
    li {
        color: var(--text-primary);
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    li::marker {
        color: var(--accent);
        font-weight: bold;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #132345 0%, #0a0f1f 100%);
        border-right: 2px solid var(--border);
        padding-top: 24px;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 3px solid var(--border);
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        font-weight: 700;
        color: var(--text-secondary);
        border-radius: 10px 10px 0 0;
        background: transparent;
        border: none;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 16px;
        letter-spacing: 0.4px;
        padding: 4px 12px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-bright);
        border-bottom: 4px solid var(--accent-bright);
        font-weight: 800;
        background: rgba(0, 217, 255, 0.08);
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.4);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent);
        background: rgba(0, 217, 255, 0.05);
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card, [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.35), inset 0 1px 2px rgba(255, 255, 255, 0.25);
        font-weight: 600;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .metric-card:hover, [data-testid="metric-container"]:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.45), inset 0 1px 2px rgba(255, 255, 255, 0.25);
    }
    
    /* ===== CONTENT CONTAINERS ===== */
    [data-testid="stVerticalBlock"] > [style*="flex-direction"] {
        gap: 24px;
    }
    
    .card, [data-testid="stExpander"] {
        background: linear-gradient(135deg, rgba(26, 42, 94, 0.6) 0%, rgba(20, 34, 87, 0.6) 100%);
        border: 2px solid var(--border);
        border-radius: 14px;
        padding: 28px;
        margin: 14px 0;
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    
    .card:hover, [data-testid="stExpander"]:hover {
        background: linear-gradient(135deg, rgba(26, 42, 94, 0.8) 0%, rgba(20, 34, 87, 0.8) 100%);
        border-color: var(--accent);
        box-shadow: 0 12px 40px rgba(0, 217, 255, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* ===== STATUS BOXES ===== */
    .success-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 100, 0.1) 100%);
        border-left: 6px solid var(--success);
        border: 2px solid rgba(0, 255, 136, 0.3);
        padding: 18px;
        border-radius: 12px;
        margin: 14px 0;
        color: #ffffff;
        backdrop-filter: blur(12px);
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.7;
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.15);
        transition: all 0.3s ease;
    }
    
    .success-box:hover {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.25) 0%, rgba(0, 200, 100, 0.15) 100%);
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.25);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%);
        border-left: 6px solid var(--warning);
        border: 2px solid rgba(255, 165, 0, 0.3);
        padding: 18px;
        border-radius: 12px;
        margin: 14px 0;
        color: #ffffff;
        backdrop-filter: blur(12px);
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.7;
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .warning-box:hover {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.25) 0%, rgba(255, 140, 0, 0.15) 100%);
        box-shadow: 0 10px 30px rgba(255, 165, 0, 0.25);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(255, 85, 85, 0.15) 0%, rgba(255, 50, 50, 0.1) 100%);
        border-left: 6px solid var(--danger);
        border: 2px solid rgba(255, 85, 85, 0.3);
        padding: 18px;
        border-radius: 12px;
        margin: 14px 0;
        color: #ffffff;
        backdrop-filter: blur(12px);
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.7;
        box-shadow: 0 6px 20px rgba(255, 85, 85, 0.15);
        transition: all 0.3s ease;
    }
    
    .error-box:hover {
        background: linear-gradient(135deg, rgba(255, 85, 85, 0.25) 0%, rgba(255, 50, 50, 0.15) 100%);
        box-shadow: 0 10px 30px rgba(255, 85, 85, 0.25);
    }
    
    /* ===== INPUTS ===== */
    .stTextInput input, 
    .stSelectbox select, 
    .stNumberInput input, 
    .stTextArea textarea,
    .stSlider {
        background-color: rgba(20, 34, 60, 0.8) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
        padding: 12px 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:hover,
    .stSelectbox select:hover,
    .stNumberInput input:hover,
    .stTextArea textarea:hover {
        border-color: var(--accent) !important;
        background-color: rgba(20, 34, 60, 0.95) !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.15) !important;
    }
    
    .stTextInput input:focus,
    .stSelectbox select:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent-bright) !important;
        background-color: rgba(20, 34, 60, 1) !important;
        box-shadow: 0 0 0 4px rgba(0, 217, 255, 0.15) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder, 
    .stTextArea textarea::placeholder {
        color: var(--text-tertiary) !important;
        font-weight: 500 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 800;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 15px;
        letter-spacing: 0.5px;
        padding: 14px 28px !important;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #7b8ff0 0%, #8355b0 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* ===== CODE BLOCKS ===== */
    .stCodeBlock {
        background-color: rgba(10, 15, 31, 0.95) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        overflow-x: auto !important;
    }
    
    .stCodeBlock > code,
    .stCodeBlock pre {
        font-family: 'Courier New', 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        line-height: 2.0 !important;
        color: #e0f7ff !important;
        white-space: pre !important;
        word-wrap: normal !important;
        overflow-wrap: normal !important;
        display: block !important;
        padding: 16px !important;
    }
    
    code {
        background-color: rgba(10, 15, 31, 0.9) !important;
        border-radius: 6px !important;
        font-family: 'Courier New', 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        color: #e0f7ff !important;
        padding: 2px 6px !important;
    }
    
    pre {
        background-color: rgba(10, 15, 31, 0.95) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-family: 'Courier New', 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        line-height: 2.0 !important;
        color: #e0f7ff !important;
        white-space: pre !important;
        word-wrap: normal !important;
        overflow-x: auto !important;
        margin: 12px 0 !important;
    }
    
    pre code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        margin: 28px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--accent) 50%, transparent 100%);
    }
    
    /* ===== LABELS ===== */
    label, .stLabel {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 800;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        display: block;
    }
    
    /* ===== VALUES & METRICS ===== */
    .value-text, .metric-value {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 40px;
        font-weight: 900;
        color: var(--accent-bright);
        letter-spacing: -1.5px;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
    }
    
    /* ===== DATA TABLES ===== */
    .stDataFrame {
        background: transparent !important;
    }
    
    [data-testid="stDataFrame"] {
        background: transparent !important;
    }
    
    .stDataFrame table {
        background: linear-gradient(135deg, rgba(26, 42, 94, 0.5) 0%, rgba(20, 34, 87, 0.5) 100%) !important;
        border-collapse: collapse !important;
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    .stDataFrame thead {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
    }
    
    .stDataFrame thead th {
        color: var(--accent-bright) !important;
        font-weight: 800 !important;
        padding: 16px !important;
        border-bottom: 3px solid var(--accent) !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame tbody td {
        color: var(--text-primary) !important;
        padding: 14px 16px !important;
        border-bottom: 1px solid var(--border) !important;
        font-size: 14px !important;
        font-weight: 500;
    }
    
    .stDataFrame tbody tr:hover {
        background: rgba(0, 217, 255, 0.1) !important;
    }
    
    /* ===== CHARTS & GRAPHS ===== */
    .plotly {
        background: transparent !important;
    }
    
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px rgba(0, 217, 255, 0.3); }
        50% { text-shadow: 0 0 20px rgba(0, 217, 255, 0.6); }
    }
    
    .card, .metric-card, .success-box, .warning-box, .error-box {
        animation: fadeIn 0.5s ease-out;
    }
    
    h1, h2 {
        animation: glow 3s ease-in-out infinite;
    }
    
    /* ===== SELECTBOX DROPDOWN ===== */
    [data-baseweb="select"] {
        background: rgba(20, 34, 60, 0.8) !important;
    }
    
    .stSelectbox {
        font-size: 15px;
    }
    
    /* ===== EXPANDER ===== */
    [data-testid="stExpander"] > div:first-child {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        color: var(--accent-bright) !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 16px !important;
    }
    
    /* ===== SPACING UTILITIES ===== */
    .spacer-small { margin: 10px 0; }
    .spacer-medium { margin: 20px 0; }
    .spacer-large { margin: 30px 0; }
    
    /* ===== SCROLLBAR STYLING ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 34, 60, 0.4);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 10px;
        border: 2px solid rgba(20, 34, 60, 0.4);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7b8ff0 0%, #8355b0 100%);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

@st.cache_data(ttl=30)  # Cache for 30 seconds only (2x per minute) to get fresh data
def get_latest_price(ticker):
    """Fetch the latest current price for a ticker from yfinance"""
    try:
        IST = pytz.timezone("Asia/Kolkata")
        
        # Map ticker names to yfinance symbols
        ticker_map = {
            "NIFTY BANK": "^NSEBANK",
            "NIFTY": "^NSEI",
            "NIFTY COMMODITIES": "NIFTYCOMMDX.NS",
            "NIFTY CONSUMPTION": "NIFTYCONSUMER.NS",
            "NIFTY FIN SERVICE": "NIFTYFINANCE.NS",
            "NIFTY INDIA MFG": "NIFTYMFG.NS",
            "INDIA VIX": "^INDIAVIX"
        }
        
        yf_ticker = ticker_map.get(ticker, ticker)
        
        # Fetch latest data - 1d is sufficient to get the latest price
        data = yf.download(
            yf_ticker,
            period="1d",
            progress=False,
            timeout=10
        )
        
        if data.empty:
            return None
        
        # Get the latest close price
        latest_price = data['Close'].iloc[-1]
        return float(latest_price)
        
    except Exception as e:
        return None

# --------------------------------------------------
# Helper Functions - Combined Strategy
# --------------------------------------------------

@st.cache_data
def load_data(ticker):
    """Load and cache data for a ticker"""
    try:
        data = load_kaggle_data(ticker)
        return clean_ohlcv_data(data)
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

def fetch_today_realtime_data(ticker, days=1):
    """Fetch today's real-time intraday data from yfinance with improved reliability"""
    try:
        IST = pytz.timezone("Asia/Kolkata")
        
        # Map ticker names to yfinance symbols
        ticker_map = {
            "NIFTY BANK": "^NSEBANK",
            "NIFTY": "^NSEI",
            "NIFTY COMMODITIES": "NIFTYCOMMDX.NS",
            "NIFTY CONSUMPTION": "NIFTYCONSUMER.NS",
            "NIFTY FIN SERVICE": "NIFTYFINANCE.NS",
            "NIFTY INDIA MFG": "NIFTYMFG.NS",
            "INDIA VIX": "^INDIAVIX"
        }
        
        yf_ticker = ticker_map.get(ticker, ticker)
        
        # Get current time in IST
        now_ist = datetime.now(IST)
        
        # Check if today is a holiday or weekend, and if so, find the last trading day
        current_day = now_ist.weekday()  # 0=Monday, 6=Sunday
        
        # Indian stock market holidays (sample list - add known holidays)
        indian_holidays = {
            (1, 26),   # Republic Day
            (3, 8),    # Maha Shivaratri (varies, but placeholder)
            (3, 29),   # Good Friday
            (4, 17),   # Ram Navami (varies)
            (4, 21),   # Mahavir Jayanti
            (5, 23),   # Buddha Purnima (varies)
            (8, 15),   # Independence Day
            (8, 27),   # Janmashtami (varies)
            (9, 16),   # Milad-un-Nabi (varies)
            (10, 2),   # Gandhi Jayanti
            (10, 12),  # Dussehra (varies)
            (10, 31),  # Diwali (varies, usually Oct-Nov)
            (11, 1),   # Diwali (varies)
            (11, 15),  # Guru Nanak Jayanti
            (12, 25),  # Christmas
        }
        
        target_date = now_ist
        days_back = 0
        max_lookback = 7  # Don't look back more than 7 days
        
        # Keep going back until we find a trading day (weekday and not a holiday)
        while days_back < max_lookback:
            if target_date.weekday() < 5:  # Monday=0 to Friday=4
                # Check if it's a known holiday
                date_tuple = (target_date.month, target_date.day)
                if date_tuple not in indian_holidays:
                    # This should be a trading day
                    break
            # Go back one day
            target_date = target_date - timedelta(days=1)
            days_back += 1
        
        # Fetch last N days of 1-minute data
        end_date = target_date + timedelta(days=1)
        start_date = target_date - timedelta(days=days)
        
        # Try 1-minute interval first
        try:
            live_data = yf.download(
                yf_ticker,
                start=start_date,
                end=end_date,
                interval="1m",
                progress=False,
                timeout=10
            )
        except Exception as intraday_error:
            # Fallback to 5-minute interval if 1-minute fails
            try:
                live_data = yf.download(
                    yf_ticker,
                    start=start_date,
                    end=end_date,
                    interval="5m",
                    progress=False,
                    timeout=10
                )
            except Exception as five_min_error:
                # Final fallback to hourly data + latest daily candle
                live_data = yf.download(
                    yf_ticker,
                    start=start_date,
                    end=end_date,
                    interval="1h",
                    progress=False,
                    timeout=10
                )
        
        if live_data.empty:
            return None
        
        # Convert timezone to IST if needed
        if live_data.index.tz is None:
            live_data.index = live_data.index.tz_localize("UTC").tz_convert(IST)
        else:
            live_data.index = live_data.index.tz_convert(IST)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(live_data.columns, pd.MultiIndex):
            live_data.columns = [col[0] for col in live_data.columns]
        
        # Standardize column names
        live_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        live_data = live_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # Ensure we have current data
        if len(live_data) > 0:
            last_time = live_data.index[-1]
            time_diff = (datetime.now(IST) - last_time.replace(tzinfo=IST)).total_seconds() / 60
            # If last data point is more than 30 minutes old, add current day's OHLC
            if time_diff > 30:
                try:
                    today_data = yf.download(
                        yf_ticker,
                        start=end_date.date(),
                        end=end_date + timedelta(days=1),
                        interval="1d",
                        progress=False,
                        timeout=10
                    )
                    if not today_data.empty:
                        today_data.index = pd.to_datetime(today_data.index).tz_localize("UTC").tz_convert(IST)
                        if isinstance(today_data.columns, pd.MultiIndex):
                            today_data.columns = [col[0] for col in today_data.columns]
                        today_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        # Use the latest daily candle if it's more recent
                        if today_data.index[-1] > live_data.index[-1]:
                            live_data = pd.concat([live_data, today_data[[col for col in today_data.columns if col in live_data.columns]]])
                except:
                    pass
        
        return live_data
        
    except Exception as e:
        st.error(f"Error fetching real-time data for {ticker}: {e}")
        return None

def add_scalping_signals(data):
    """Add rule-based scalping signals using RSI, SMA, and MACD"""
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

    # Near-high strength
    close_20_high = df["Close"].rolling(20).max()
    buy_strength = df["Close"] > 0.95 * close_20_high

    # Buy signals
    buy_uptrend = (df["Close"] > sma_20) & (sma_20 > sma_50)
    buy_rsi = (rsi > 30) & (rsi < 50)
    buy_macd = (macd > 0) & (macd_hist > 0)
    buy_signal = (buy_uptrend & buy_rsi) | (buy_uptrend & buy_macd) | (buy_uptrend & buy_strength)

    # Sell signals
    sell_downtrend = (df["Close"] < sma_20) & (sma_20 < sma_50)
    sell_rsi = (rsi < 70) & (rsi > 50)
    sell_macd = (macd < 0) & (macd_hist < 0)
    sell_signal = (sell_downtrend & sell_rsi) | (sell_downtrend & sell_macd)

    signal = pd.Series(0, index=df.index)
    signal[buy_signal & ~sell_signal] = 1
    signal[sell_signal & ~buy_signal] = -1
    signal[~buy_signal & ~sell_signal] = 0

    df["strategy_signal"] = signal
    df["RSI"] = rsi
    df["SMA_20"] = sma_20
    df["SMA_50"] = sma_50
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal

    return df

def add_basic_features(data, horizon=3, cost=0.0003):
    """Add basic ML features"""
    df = data.copy()

    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    sma_10 = df["Close"].rolling(10).mean()
    sma_20 = df["Close"].rolling(20).mean()

    df["trend_10"] = (df["Close"] - sma_10) / (sma_10 + 1e-8)
    df["trend_20"] = (df["Close"] - sma_20) / (sma_20 + 1e-8)
    df["trend_diff"] = (sma_10 - sma_20) / (sma_20 + 1e-8)

    df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)
    df["body_pct"] = (df["Close"] - df["Open"]) / (df["Close"] + 1e-8)
    df["body_abs"] = df["body_pct"].abs()

    df["volatility_10"] = df["returns"].rolling(10).std()
    df["vol_ratio"] = df["volatility_10"] / (df["volatility_10"].rolling(50).mean() + 1e-8)
    df["high_vol"] = (df["vol_ratio"] > 1.0).astype(int)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI"] = (100 - (100 / (1 + rs))) / 100.0

    if "Volume" in df.columns and float(df["Volume"].sum()) > 0:
        vol_sma = df["Volume"].rolling(20).mean()
        df["Volume_norm"] = np.log1p(df["Volume"] / (vol_sma + 1e-8))
    else:
        df["Volume_norm"] = 0.0

    future_return = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    df["target"] = (future_return > cost).astype(int)

    df.dropna(inplace=True)
    return df

def add_advanced_features(df):
    """Add advanced features from your notebook"""
    d = df.copy()
    
    d['momentum_5'] = (d['Close'] - d['Close'].shift(5)) / d['Close'].shift(5)
    d['momentum_10'] = (d['Close'] - d['Close'].shift(10)) / d['Close'].shift(10)
    d['momentum_20'] = (d['Close'] - d['Close'].shift(20)) / d['Close'].shift(20)
    d['momentum_accel'] = d['momentum_5'] - d['momentum_10']
    
    d['tr'] = np.maximum(
        d['High'] - d['Low'],
        np.maximum(
            abs(d['High'] - d['Close'].shift(1)),
            abs(d['Low'] - d['Close'].shift(1))
        )
    )
    d['atr'] = d['tr'].rolling(14).mean()
    d['atr_pct'] = d['atr'] / d['Close']
    
    sma_10 = d['Close'].rolling(10).mean()
    sma_20 = d['Close'].rolling(20).mean()
    sma_50 = d['Close'].rolling(50).mean()
    
    d['trend_strength'] = (
        ((d['Close'] > sma_20).astype(int) * 0.4) +
        ((sma_20 > sma_50).astype(int) * 0.3) +
        ((d['momentum_5'] > 0).astype(int) * 0.3)
    )
    
    d['vol_20'] = d['Close'].pct_change().rolling(20).std()
    d['vol_regime'] = (d['vol_20'] > d['vol_20'].rolling(50).mean()).astype(int)
    
    d['high_20'] = d['High'].rolling(20).max()
    d['low_20'] = d['Low'].rolling(20).min()
    d['price_position'] = (d['Close'] - d['low_20']) / (d['high_20'] - d['low_20'] + 1e-8)
    
    d['momentum_trend_signal'] = d['momentum_5'] * d['trend_strength']
    d['price_momentum_align'] = d['price_position'] * d['momentum_10']
    
    return d

@st.cache_resource
def train_ensemble_model(X_train_scaled, y_train):
    """Train XGBoost + LightGBM ensemble"""
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
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    
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
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    
    return xgb_model, lgb_model

def get_ensemble_probability(xgb_model, lgb_model, X_scaled):
    """Get ensemble ML probability"""
    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_scaled)[:, 1]
    return xgb_proba * 0.55 + lgb_proba * 0.45

# --------------------------------------------------
# Main App
# --------------------------------------------------

# --------------------------------------------------
# Main App Header
# --------------------------------------------------

st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 32px;">🎯 Combined ML + Scalping Strategy</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">XGBoost + LightGBM Ensemble with Advanced Risk Management</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #667eea; text-align: center;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    st.divider()
    
    selected_ticker = st.selectbox(
        "📊 Select Ticker",
        DEFAULT_TICKERS,
        help="Choose a stock ticker to analyze"
    )
    
    st.info("💡 Tip: Switch tickers to compare different market segments", icon="ℹ️")

# --------------------------------------------------
# Main Tabs with Enhanced Styling
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "🤖 ML Predictions", 
    "📊 Live Trading", 
    "📅 Daily Simulation",
    "🔴 Real-Time Market (9:15 AM - 3:15 PM)"
])

st.divider()

# --------------------------------------------------
# Data Loading & Model Training
# --------------------------------------------------

@st.cache_resource
def prepare_strategy_data(ticker):
    """Prepare all data and train model"""
    try:
        # Load data
        raw_data = load_data(ticker)
        if raw_data is None:
            return None
        
        cleaned_data = clean_ohlcv_data(raw_data)
        train_data, test_data = split_data_by_date(cleaned_data)
        
        # Add signals and features to training data
        train_with_signals = add_scalping_signals(train_data)
        train_with_basic = add_basic_features(train_with_signals)
        train_with_advanced = add_advanced_features(train_with_basic)
        train_with_advanced = train_with_advanced.dropna()
        
        # Add signals and features to test data
        test_with_signals = add_scalping_signals(test_data)
        test_with_basic = add_basic_features(test_with_signals)
        test_with_advanced = add_advanced_features(test_with_basic)
        test_with_advanced = test_with_advanced.dropna()
        
        # Get feature columns
        feature_cols = [
            c for c in train_with_advanced.columns
            if c not in ["target", "Open", "High", "Low", "Close", "Volume", "strategy_signal", "tr", "high_20", "low_20"]
        ]
        
        # Prepare training data
        X_train = train_with_advanced[feature_cols]
        y_train = train_with_advanced["target"]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train ensemble
        xgb_model, lgb_model = train_ensemble_model(X_train_scaled, y_train)
        
        # Prepare test data
        X_test = test_with_advanced[feature_cols]
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'train_data': train_with_advanced,
            'test_data': test_with_advanced,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'xgb_model': xgb_model,
            'lgb_model': lgb_model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

# Load data
with st.spinner(f"Loading data for {selected_ticker}..."):
    strategy_data = prepare_strategy_data(selected_ticker)

if strategy_data is None:
    st.error("Failed to load data")
    st.stop()

# Get ML probabilities
ml_proba = get_ensemble_probability(
    strategy_data['xgb_model'],
    strategy_data['lgb_model'],
    strategy_data['X_test_scaled']
)

test_df = strategy_data['test_data'].copy()
test_df['ml_prob'] = ml_proba
prices = test_df["Close"].values
atr_values = test_df["atr"].values

# --------------------------------------------------
# FETCH TODAY'S REAL-TIME DATA FOR OVERVIEW
# --------------------------------------------------
# Force update real-time data every page refresh (no caching)
today_realtime_data = fetch_today_realtime_data(selected_ticker, days=3)  # Fetch 3 days for robustness

is_realtime = False
use_fallback = False

if today_realtime_data is not None and len(today_realtime_data) > 0:
    try:
        # Use the last trading day's data (which could be today, Friday, or earlier if holidays)
        IST = pytz.timezone("Asia/Kolkata")
        last_trading_date = today_realtime_data.index.max().date()
        today_data = today_realtime_data[today_realtime_data.index.date == last_trading_date].copy()
        
        # If we don't have enough data for the last trading day, include previous days for feature calculation
        if len(today_data) < 20 and len(today_realtime_data) >= 20:
            today_data = today_realtime_data.tail(max(20, len(today_data))).copy()
        
        if len(today_data) >= 5:
            # Process real-time data with features
            with st.spinner("Processing real-time market data with ML features..."):
                today_with_signals = add_scalping_signals(today_data.copy())
                today_with_basic = add_basic_features(today_with_signals.copy())
                today_with_advanced = add_advanced_features(today_with_basic.copy())
                today_with_advanced = today_with_advanced.dropna()
                
                if len(today_with_advanced) > 0:
                    # Get ML probabilities for today's data
                    X_today = today_with_advanced[strategy_data['feature_cols']].copy()
                    X_today_scaled = strategy_data['scaler'].transform(X_today)
                    today_ml_proba = get_ensemble_probability(
                        strategy_data['xgb_model'],
                        strategy_data['lgb_model'],
                        X_today_scaled
                    )
                    
                    # Use today's data for Overview
                    today_df = today_with_advanced.copy()
                    today_df['ml_prob'] = today_ml_proba
                    overview_prices = today_df["Close"].values
                    overview_atr_values = today_df["atr"].values
                    overview_df = today_df
                    is_realtime = True
        else:
            # Not enough data for today
            use_fallback = True
    except Exception as e:
        st.error(f"Error processing real-time data: {e}")
        use_fallback = True
else:
    # Could not fetch real-time data
    use_fallback = True

# Fallback to backtest data only if real-time data couldn't be processed
if use_fallback:
    overview_prices = prices
    overview_atr_values = atr_values
    overview_df = test_df
    is_realtime = False

# --------------------------------------------------
# TAB 1: Overview
# --------------------------------------------------

with tab1:
    st.markdown("<h2 style='color: #667eea;'>📈 Market Overview & Strategy Status</h2>", unsafe_allow_html=True)
    
    if is_realtime:
        st.success("✅ Real-time Data Active - Fresh market data from today")
    else:
        st.error("❌ FALLBACK MODE - Using historical backtest data (2024) - Real-time data unavailable. Please check market hours or try refreshing.")
    
    st.header(f"Strategy Overview - {selected_ticker}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Get the latest price - prioritize fresh real-time price
    latest_price = get_latest_price(selected_ticker)
    if latest_price is None:
        # Fallback to overview data if real-time fetch fails
        latest_price = overview_prices[-1]
    
    latest_ml_prob = overview_df['ml_prob'].iloc[-1]
    latest_rsi = overview_df['RSI'].iloc[-1]
    latest_signal = overview_df['strategy_signal'].iloc[-1]
    latest_atr_pct = overview_df['atr_pct'].iloc[-1]
    
    st.subheader("📊 Real-Time Market Data")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        price_change = ((latest_price - overview_prices[-20])/overview_prices[-20]*100) if len(overview_prices) > 20 else 0
        st.metric("💰 Price", f"₹{latest_price:,.2f}")
    
    with col2:
        st.metric("🤖 ML Conf", f"{latest_ml_prob*100:.1f}%")
    
    with col3:
        rsi_val = latest_rsi*100
        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        st.metric("📈 RSI", f"{rsi_val:.1f}")
    
    with col4:
        signal_text = "🟢 BUY" if latest_signal == 1 else "🔴 SELL" if latest_signal == -1 else "⚪ NEUTRAL"
        st.metric("📍 Signal", signal_text)
    
    with col5:
        st.metric("⚡ ATR %", f"{latest_atr_pct*100:.3f}%")
    
    st.divider()
    
    # Strategy info
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Ensemble Model")
        st.info("""
        **XGBoost + LightGBM Hybrid (55% + 45%)**
        - 250 estimators each
        - Advanced feature engineering
        - AUC-optimized
        - 25+ features per candle
        """)
    
    with col_right:
        st.subheader("🎯 Trading Logic")
        st.info(f"""
        **Combined Strategy**
        - ML Probability entry threshold: {np.percentile(overview_df['ml_prob'].values, 75):.2%}
        - Dynamic ATR-based stops
        - Trailing stops with 3-tier profits
        - Multiple position management
        """)

# --------------------------------------------------
# TAB 2: ML Predictions
# --------------------------------------------------

with tab2:
    st.markdown("<h2 style='color: #667eea;'>🤖 ML Model Analysis & Insights</h2>", unsafe_allow_html=True)
    
    # ML Probability distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Probability Distribution")
        fig_prob_dist = px.histogram(
            x=overview_df['ml_prob'].values,
            nbins=50,
            title="Ensemble ML Probability Distribution",
            labels={'x': 'Probability', 'y': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        fig_prob_dist.update_layout(showlegend=False, hovermode='x unified')
        st.plotly_chart(fig_prob_dist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Key Statistics")
        ml_probs_array = overview_df['ml_prob'].values
        stats_data = {
            'Metric': ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'Std Dev'],
            'Value': [
                f"{ml_probs_array.min():.4f}",
                f"{np.percentile(ml_probs_array, 25):.4f}",
                f"{np.median(ml_probs_array):.4f}",
                f"{np.percentile(ml_probs_array, 75):.4f}",
                f"{ml_probs_array.max():.4f}",
                f"{ml_probs_array.mean():.4f}",
                f"{ml_probs_array.std():.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Price vs ML Probability
    fig_corr = go.Figure()
    
    fig_corr.add_trace(go.Scatter(
        x=np.arange(len(prices)),
        y=prices / prices[0],
        mode='lines',
        name='Price (Normalized)',
        yaxis='y1',
        line=dict(color='black', width=2)
    ))
    
    fig_corr.add_trace(go.Scatter(
        x=np.arange(len(ml_proba)),
        y=ml_proba,
        mode='lines',
        name='ML Probability',
        yaxis='y2',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig_corr.update_layout(
        title='Price vs ML Probability Over Time',
        xaxis_title='Candle Index',
        yaxis=dict(title='Normalized Price', side='left'),
        yaxis2=dict(title='ML Probability', side='right', overlaying='y'),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# TAB 3: Live Trading
# --------------------------------------------------

with tab3:
    st.markdown("<h2 style='color: #667eea;'>📊 Live Paper Trading Dashboard</h2>", unsafe_allow_html=True)
    
    # Critical data freshness indicator
    if is_realtime:
        st.success("✅ LIVE TRADING MODE - Real-time data is active and fresh")
    else:
        st.error("❌ FALLBACK MODE - Using historical 2024 backtest data. Real-time market data not available. Refresh the page to retry.", icon="🚨")
    
    st.info("🔴 Real-time simulation with ML ensemble predictions and dynamic risk management", icon="ℹ️")
    
    # Data source indicator
    col_source1, col_source2, col_source3 = st.columns(3)
    with col_source1:
        is_live_status = "✅ LIVE" if is_realtime else "⏱️ HISTORICAL"
        st.metric("Data Source", is_live_status)
    
    with col_source2:
        if len(overview_df) > 0:
            last_candle_time = overview_df.index[-1]
            IST = pytz.timezone("Asia/Kolkata")
            if last_candle_time.tzinfo is None:
                last_candle_time = IST.localize(last_candle_time)
            time_diff = datetime.now(IST) - last_candle_time.replace(tzinfo=IST)
            minutes_ago = int(time_diff.total_seconds() / 60)
            st.metric("Latest Candle Age", f"{minutes_ago} min ago")
    
    with col_source3:
        st.metric("Total Candles", f"{len(overview_df):,}")
    
    st.divider()
    
    # Recent signals
    st.subheader("📈 Recent Trading Signals (Last 20 Candles)")
    
    recent_df = pd.DataFrame({
        'Time': overview_df.index[-20:],
        'Price': overview_prices[-20:],
        'ML Prob': overview_df['ml_prob'].iloc[-20:].values,
        'RSI': overview_df['RSI'].iloc[-20:].values * 100,
        'ATR %': overview_df['atr_pct'].iloc[-20:].values * 100,
        'Signal': overview_df['strategy_signal'].iloc[-20:].values
    })
    
    # Format the Time column to show readable timestamps
    recent_df['Time'] = recent_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    recent_df['Signal'] = recent_df['Signal'].map({1: '🟢 BUY', -1: '🔴 SELL', 0: '⚪ NEUTRAL'})
    recent_df = recent_df.round(4)
    
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Current status with improved styling
    st.subheader("⚡ Real-Time Status Indicators")
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        confidence = overview_df['ml_prob'].iloc[-1] * 100
        strength = "💪 Strong" if overview_df['ml_prob'].iloc[-1] > 0.6 else "⚠️ Weak"
        st.metric("ML Confidence", f"{confidence:.1f}%", strength)
    
    with col_status2:
        signals_buy = (overview_df['strategy_signal'] == 1).sum()
        signals_sell = (overview_df['strategy_signal'] == -1).sum()
        st.metric("📊 BUY vs SELL", f"🟢 {signals_buy} | 🔴 {signals_sell}")
    
    with col_status3:
        st.metric("📉 Data Points", f"{len(overview_df):,} candles")
    
    st.divider()
    
    # Detailed Data Verification
    with st.expander("🔍 Data Verification & Debug Info"):
        col_debug1, col_debug2 = st.columns(2)
        
        with col_debug1:
            st.subheader("Latest Candle Details")
            if len(overview_df) > 0:
                last_candle = overview_df.iloc[-1]
                last_time = overview_df.index[-1]
                
                debug_info = f"""
                **Time:** {last_time}
                **Close Price:** ₹{last_candle['Close']:,.2f}
                **High:** ₹{last_candle['High']:,.2f}
                **Low:** ₹{last_candle['Low']:,.2f}
                **Open:** ₹{last_candle['Open']:,.2f}
                **Volume:** {last_candle['Volume']:,.0f}
                **ML Probability:** {last_candle['ml_prob']*100:.2f}%
                **RSI:** {last_candle['RSI']*100:.2f}
                **ATR:** {last_candle['atr']:.2f}
                """
                st.markdown(debug_info)
        
        with col_debug2:
            st.subheader("Data Source Information")
            source_info = f"""
            **Data Type:** {'Real-time' if is_realtime else 'Historical Backtest'}
            **Total Candles in Dataset:** {len(overview_df):,}
            **Date Range:** {overview_df.index[0]} to {overview_df.index[-1]}
            **Days Covered:** {(overview_df.index[-1] - overview_df.index[0]).days} days
            **Ticker:** {selected_ticker}
            **Strategy Signal:** {'🟢 BUY' if overview_df['strategy_signal'].iloc[-1] == 1 else '🔴 SELL' if overview_df['strategy_signal'].iloc[-1] == -1 else '⚪ NEUTRAL'}
            """
            st.markdown(source_info)
    
    st.divider()
    
    # Candlestick chart for last 20 candles
    st.subheader("📊 Last 20 Candles - Candlestick Chart")
    
    chart_data = overview_df.iloc[-20:].copy()
    chart_data['Index'] = range(len(chart_data))
    
    fig = go.Figure(data=[go.Candlestick(
        x=chart_data['Index'],
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title=f"Last 20 Candles - {selected_ticker}",
        yaxis_title='Price (₹)',
        xaxis_title='Candle Index',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("<h2 style='color: #667eea;'>📅 Daily Live Simulation</h2>", unsafe_allow_html=True)
    
    st.info("🔴 Select a date to simulate live trading for that specific day using real yfinance data", icon="⏰")
    
    # Define trading cost
    COST_PER_TRADE = 0.000001
    
    IST = pytz.timezone("Asia/Kolkata")
    
    # Date selector
    col_date1, col_date2 = st.columns([2, 3])
    
    with col_date1:
        days_back = st.slider("Days back from today:", 0, 30, 0, key="daily_sim_days")
    
    with col_date2:
        selected_sim_date = datetime.now(IST).date() - timedelta(days=days_back)
        st.metric("Simulation Date", selected_sim_date.strftime("%A, %B %d, %Y"))
    
    # Fetch and simulate for selected date
    if st.button("📊 Run Daily Simulation", key="run_daily_sim"):
        
        with st.spinner("Fetching yfinance data..."):
            try:
                # Fetch data for the selected date
                start_time = IST.localize(datetime.combine(selected_sim_date, datetime.min.time()))
                end_time = IST.localize(datetime.combine(selected_sim_date, datetime.max.time()))
                
                try:
                    sim_data = yf.download(
                        "^NSEBANK",
                        start=start_time,
                        end=end_time,
                        interval="1m",
                        progress=False
                    )
                    ticker_used = "^NSEBANK"
                except:
                    sim_data = yf.download(
                        "NIFTYBANK.NS",
                        start=start_time,
                        end=end_time,
                        interval="1m",
                        progress=False
                    )
                    ticker_used = "NIFTYBANK.NS"
                
                # Convert timezone
                if sim_data.index.tz is None:
                    sim_data.index = sim_data.index.tz_localize("UTC").tz_convert(IST)
                else:
                    sim_data.index = sim_data.index.tz_convert(IST)
                
                # Fix columns
                if isinstance(sim_data.columns, pd.MultiIndex):
                    sim_data.columns = [col[0] for col in sim_data.columns]
                
                sim_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                sim_data = sim_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                
                if len(sim_data) == 0:
                    st.warning("⚠️ No data available for this date. It might be a holiday or weekend.")
                    st.stop()
                
                st.success(f"✅ Fetched {len(sim_data)} candles from yfinance")
                
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")
                st.stop()
        
        # Apply feature engineering
        with st.spinner("Processing features..."):
            try:
                sim_with_signals = add_scalping_signals(sim_data.copy())
                sim_with_features = add_basic_features(sim_with_signals.copy())
                sim_features = add_advanced_features(sim_with_features.copy())
                sim_features = sim_features.dropna()
                
                X_sim = sim_features[strategy_data['feature_cols']].copy()
                X_sim_scaled = strategy_data['scaler'].transform(X_sim)
                ml_prob_sim = get_ensemble_probability(
                    strategy_data['xgb_model'],
                    strategy_data['lgb_model'],
                    X_sim_scaled
                )
                
                sim_prices = sim_features["Close"].values
                sim_atr = sim_features["atr"].values
                
                st.success(f"✅ Features applied: {len(strategy_data['feature_cols'])} features, {len(sim_features)} candles")
                
            except Exception as e:
                st.error(f"❌ Error in feature engineering: {e}")
                st.stop()
        
        # Run daily backtest simulation
        with st.spinner("Running daily trading simulation..."):
            
            SIM_ENTRY_THRESHOLD = np.percentile(ml_prob_sim, 55)
            SIM_STOP_LOSS = 0.010
            SIM_TAKE_PROFIT = 0.025
            SIM_TRAIL_STOP = 0.002
            SIM_HORIZON = 15
            SIM_MIN_POS = 0.05
            SIM_MAX_POS = 0.15
            
            sim_capital = INITIAL_CAPITAL
            sim_peak_capital = sim_capital
            sim_positions = []
            sim_trades = []
            sim_equity = [sim_capital]
            sim_logs = []
            
            # Header
            sim_logs.append("=" * 80)
            sim_logs.append(f" LIVE MARKET SIMULATION - {selected_sim_date.strftime('%Y-%m-%d').upper()}")
            sim_logs.append("=" * 80)
            sim_logs.append(f"\n Fetched {len(sim_data)} candles")
            sim_logs.append(f"   Time Range: {sim_data.index[0]} to {sim_data.index[-1]}")
            sim_logs.append(f"   Price Range: ₹{sim_data['Low'].min():.2f} - ₹{sim_data['High'].max():.2f}")
            sim_logs.append(f"\n Applying feature engineering to data...")
            sim_logs.append(f" Features applied successfully")
            sim_logs.append(f"   ML Probability range: {ml_prob_sim.min():.4f} to {ml_prob_sim.max():.4f}")
            sim_logs.append(f"   Mean confidence: {ml_prob_sim.mean():.4f}")
            sim_logs.append(f"   Data points: {len(sim_features)} candles")
            sim_logs.append("")
            
            # Daily simulation loop
            for i in range(len(sim_prices)):
                current_price = sim_prices[i]
                current_time = sim_features.index[i]
                current_ml_prob = ml_prob_sim[i]
                current_atr = sim_atr[i]
                
                if i < 20:
                    sim_equity.append(sim_capital)
                    continue
                
                # Close positions
                positions_to_close = []
                
                for pos_idx, pos in enumerate(sim_positions):
                    price_move = (current_price - pos['entry_price']) / pos['entry_price']
                    vol_adj_stop = SIM_STOP_LOSS * (1 + (current_atr / current_price) * 0.3)
                    vol_adj_stop = np.clip(vol_adj_stop, SIM_STOP_LOSS * 0.8, SIM_STOP_LOSS * 1.2)
                    
                    exit_hit = False
                    exit_reason = None
                    exit_price = current_price
                    
                    if price_move >= SIM_TAKE_PROFIT:
                        exit_hit = True
                        exit_reason = "PROFIT"
                        exit_price = pos['entry_price'] * (1 + SIM_TAKE_PROFIT)
                    elif price_move > SIM_TAKE_PROFIT * 0.5 and current_price < pos['max_price'] * (1 - SIM_TRAIL_STOP):
                        exit_hit = True
                        exit_reason = "TRAIL"
                        exit_price = pos['max_price'] * (1 - SIM_TRAIL_STOP)
                    elif price_move <= -vol_adj_stop:
                        exit_hit = True
                        exit_reason = "STOP"
                        exit_price = current_price
                    elif i - pos['entry_idx'] >= SIM_HORIZON:
                        exit_hit = True
                        exit_reason = "TIME"
                        exit_price = current_price
                    
                    if exit_hit:
                        ret = (exit_price - pos['entry_price']) / pos['entry_price']
                        net_ret = ret - (COST_PER_TRADE * 2)
                        pnl = pos['position_value'] * net_ret
                        sim_capital += pnl
                        sim_peak_capital = max(sim_peak_capital, sim_capital)
                        
                        emoji = "🎯" if pnl > 0 else "🛑"
                        pnl_str = f"+₹{pnl:,.0f}" if pnl > 0 else f"-₹{abs(pnl):,.0f}"
                        ret_str = f"+{net_ret*100:.2f}%" if net_ret > 0 else f"{net_ret*100:.2f}%"
                        
                        sim_logs.append(f"{emoji} EXIT | {current_time.strftime('%H:%M:%S')} | ₹{exit_price:,.2f} | P&L: {pnl_str} ({ret_str}) | [{exit_reason}] | Capital: ₹{sim_capital:,.0f}")
                        
                        sim_trades.append({
                            'entry': pos['entry_price'],
                            'exit': exit_price,
                            'prob': pos['ml_prob'],
                            'return': net_ret,
                            'pnl': pnl,
                            'reason': exit_reason
                        })
                        
                        positions_to_close.append(pos_idx)
                
                sim_positions = [pos for idx, pos in enumerate(sim_positions) if idx not in positions_to_close]
                
                # Entry logic
                if current_ml_prob >= SIM_ENTRY_THRESHOLD and i < len(sim_prices) - SIM_HORIZON:
                    momentum_5 = sim_features["momentum_5"].iloc[i] if 'momentum_5' in sim_features.columns else 0
                    
                    total_exposure = sum(p['position_fraction'] for p in sim_positions)
                    max_allowed = 1.5
                    
                    if total_exposure < max_allowed:
                        conf_edge = current_ml_prob - SIM_ENTRY_THRESHOLD
                        max_edge = 1.0 - SIM_ENTRY_THRESHOLD if SIM_ENTRY_THRESHOLD < 1.0 else 0.5
                        normalized = conf_edge / max_edge if max_edge > 0 else 0.4
                        
                        pos_frac = SIM_MIN_POS + (SIM_MAX_POS - SIM_MIN_POS) * (normalized ** 1.3)
                        pos_frac = np.clip(pos_frac, SIM_MIN_POS, SIM_MAX_POS)
                        pos_frac *= (1.0 - total_exposure / max_allowed * 0.3)
                        pos_frac = max(pos_frac, SIM_MIN_POS * 0.5)
                        
                        pos_value = sim_capital * pos_frac
                        
                        sim_positions.append({
                            'entry_idx': i,
                            'entry_price': current_price,
                            'position_value': pos_value,
                            'position_fraction': pos_frac,
                            'ml_prob': current_ml_prob,
                            'max_price': current_price
                        })
                        
                        prob_pct = current_ml_prob * 100
                        size_pct = pos_frac * 100
                        sim_logs.append(f"🟢 ENTRY #{len(sim_positions)} | {current_time.strftime('%H:%M:%S')} | ₹{current_price:,.2f} | Conf: {prob_pct:.1f}% | Size: {size_pct:.0f}% | Capital: ₹{sim_capital:,.0f}")
                
                for pos in sim_positions:
                    pos['max_price'] = max(pos['max_price'], current_price)
                
                sim_equity.append(sim_capital)
            
            # Close remaining positions
            for pos in sim_positions:
                exit_price = sim_prices[-1]
                ret = (exit_price - pos['entry_price']) / pos['entry_price']
                net_ret = ret - (COST_PER_TRADE * 2)
                pnl = pos['position_value'] * net_ret
                sim_capital += pnl
                sim_peak_capital = max(sim_peak_capital, sim_capital)
                
                sim_trades.append({
                    'entry': pos['entry_price'],
                    'exit': exit_price,
                    'prob': pos['ml_prob'],
                    'return': net_ret,
                    'pnl': pnl,
                    'reason': 'CLOSE'
                })
            
            # Display logs
            sim_logs.append("")
            sim_logs.append("=" * 80)
            sim_logs.append("")
            
            logs_text = "\n".join(sim_logs)
            st.markdown(f"```\n{logs_text}\n```")
            
            # Summary statistics
            sim_return = (sim_capital / INITIAL_CAPITAL) - 1
            sim_winning = sum(1 for t in sim_trades if t['pnl'] > 0)
            sim_losing = len(sim_trades) - sim_winning
            
            sim_logs_summary = []
            sim_logs_summary.append(" CAPITAL SUMMARY:")
            sim_logs_summary.append(f"   Starting Capital:   ₹{INITIAL_CAPITAL:,.0f}")
            sim_logs_summary.append(f"   Ending Capital:     ₹{sim_capital:,.0f}")
            sim_logs_summary.append(f"   Peak Capital:       ₹{sim_peak_capital:,.0f}")
            sim_logs_summary.append(f"   Net P&L:            ₹{sim_capital - INITIAL_CAPITAL:+,.0f}")
            sim_logs_summary.append("")
            sim_logs_summary.append(" PERFORMANCE METRICS:")
            sim_logs_summary.append(f"   Daily Return:       {sim_return*100:+.3f}%")
            
            sim_equity_arr = np.array(sim_equity)
            max_dd = ((sim_equity_arr / np.maximum.accumulate(sim_equity_arr)) - 1).min()
            sim_logs_summary.append(f"   Max Drawdown:       {max_dd*100:.2f}%")
            
            sim_logs_summary.append(f"   Trades:             {len(sim_trades)}")
            if len(sim_trades) > 0:
                sim_logs_summary.append(f"   Winning:            {sim_winning} ({sim_winning/len(sim_trades)*100:.1f}%)")
                sim_logs_summary.append(f"   Losing:             {sim_losing} ({sim_losing/len(sim_trades)*100:.1f}%)")
            
            sim_logs_summary.append("")
            sim_logs_summary.append(" EXIT DISTRIBUTION:")
            exit_reasons = {}
            for t in sim_trades:
                reason = t['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            for reason in sorted(exit_reasons.keys()):
                count = exit_reasons[reason]
                pct = count / len(sim_trades) * 100 if sim_trades else 0
                sim_logs_summary.append(f"   {reason:15} : {count:3} trades ({pct:5.1f}%)")
            
            sim_logs_summary.append("=" * 80)
            
            if sim_return > 0.05:
                verdict = f" EXCELLENT DAY! +{sim_return*100:.3f}%"
            elif sim_return > 0:
                verdict = f" PROFITABLE DAY! +{sim_return*100:.3f}%"
            elif sim_return == 0:
                verdict = " BREAKEVEN TODAY"
            else:
                verdict = f"  LOSS TODAY: {sim_return*100:.3f}%"
            
            sim_logs_summary.append(verdict)
            sim_logs_summary.append("=" * 80)
            
            summary_text = "\n".join(sim_logs_summary)
            st.markdown(f"```\n{summary_text}\n```")
    
    # ------ 7-DAY SIMULATION SECTION ------
    st.divider()
    st.markdown("### 📈 7-Day Paper Trading Simulation")
    st.info("📈 Run a 7-day backtest simulation using the last 7 days of NIFTY BANK data from yfinance", icon="📊")
    
    col_7d_1, col_7d_2 = st.columns([2, 3])
    with col_7d_1:
        st.metric("Period", "Last 7 Days")
    with col_7d_2:
        end_7d = datetime.now(IST).date()
        start_7d = end_7d - timedelta(days=7)
        st.metric("Date Range", f"{start_7d.strftime('%Y-%m-%d')} to {end_7d.strftime('%Y-%m-%d')}")
    
    if st.button("📈 Run 7-Day Simulation", key="run_7day_sim"):
        with st.spinner("Fetching 7 days of NIFTY BANK data from yfinance..."):
            try:
                end_dt = datetime.now(IST)
                start_dt = end_dt - timedelta(days=7)
                
                try:
                    data_7d = yf.download("^NSEBANK", start=start_dt, end=end_dt, interval="1m", progress=False)
                except:
                    data_7d = yf.download("NIFTYBANK.NS", start=start_dt, end=end_dt, interval="1m", progress=False)
                
                # Fix timezone
                if data_7d.index.tz is None:
                    data_7d.index = data_7d.index.tz_localize("UTC").tz_convert(IST)
                else:
                    data_7d.index = data_7d.index.tz_convert(IST)
                
                # Fix columns
                if isinstance(data_7d.columns, pd.MultiIndex):
                    data_7d.columns = [col[0] for col in data_7d.columns]
                
                data_7d.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                data_7d = data_7d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                
                if len(data_7d) == 0:
                    st.warning("⚠️ No data available for last 7 days.")
                    st.stop()
                
                st.success(f"✅ Fetched {len(data_7d)} candles from {data_7d.index[0].strftime('%Y-%m-%d %H:%M')} to {data_7d.index[-1].strftime('%Y-%m-%d %H:%M')}")
                
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")
                st.stop()
        
        with st.spinner("Processing features..."):
            try:
                data_7d_signals = add_scalping_signals(data_7d.copy())
                data_7d_features = add_basic_features(data_7d_signals.copy())
                data_7d_adv = add_advanced_features(data_7d_features.copy())
                data_7d_adv = data_7d_adv.dropna()
                
                X_7d = data_7d_adv[strategy_data['feature_cols']].copy()
                X_7d_scaled = strategy_data['scaler'].transform(X_7d)
                ml_prob_7d = get_ensemble_probability(
                    strategy_data['xgb_model'],
                    strategy_data['lgb_model'],
                    X_7d_scaled
                )
                
                prices_7d = data_7d_adv["Close"].values
                atr_7d = data_7d_adv["atr"].values
                
                st.success(f"✅ Features applied: {len(data_7d_adv)} candles processed")
                
            except Exception as e:
                st.error(f"❌ Error in feature engineering: {e}")
                st.stop()
        
        with st.spinner("Running 7-day trading simulation..."):
            # Parameters
            ENTRY_THRESHOLD_7D = np.percentile(ml_prob_7d, 75)
            STOP_LOSS_7D = 0.015
            TP1_7D = 0.004
            TP2_7D = 0.008
            TP3_7D = 0.015
            TRAIL_7D = 0.003
            HORIZON_7D = 20
            MIN_POS_7D = 0.15
            MAX_POS_7D = 0.50
            
            cap_7d = INITIAL_CAPITAL
            peak_7d = cap_7d
            pos_7d = []
            trades_7d = []
            equity_7d = [cap_7d]
            logs_7d = []
            
            logs_7d.append("=" * 80)
            logs_7d.append(" LIVE PAPER TRADING - LAST 7 DAYS")
            logs_7d.append("=" * 80)
            logs_7d.append(f"\n Fetched {len(data_7d)} candles")
            logs_7d.append(f"   Time Range: {data_7d.index[0]} to {data_7d.index[-1]}")
            logs_7d.append(f"   Price Range: ₹{data_7d['Low'].min():.2f} - ₹{data_7d['High'].max():.2f}")
            logs_7d.append(f"\n ML Probability range: {ml_prob_7d.min():.4f} to {ml_prob_7d.max():.4f}")
            logs_7d.append(f"   Mean confidence: {ml_prob_7d.mean():.4f}")
            logs_7d.append(f"   Data points: {len(data_7d_adv)} candles")
            logs_7d.append("")
            
            # Simulation loop
            for i in range(len(prices_7d)):
                price = prices_7d[i]
                tm = data_7d_adv.index[i]
                prob = ml_prob_7d[i]
                atr = atr_7d[i]
                
                if i < 20:
                    equity_7d.append(cap_7d)
                    continue
                
                # Process exits
                to_close = []
                for pidx, p in enumerate(pos_7d):
                    pm = (price - p['ep']) / p['ep']
                    vstop = STOP_LOSS_7D * (1 + (atr / price) * 0.3)
                    vstop = np.clip(vstop, STOP_LOSS_7D * 0.8, STOP_LOSS_7D * 1.2)
                    
                    exit_flag = False
                    reason = None
                    xp = price
                    
                    if pm >= TP1_7D:
                        exit_flag, reason, xp = True, "QUICK_PROFIT", p['ep'] * (1 + TP1_7D)
                    elif pm >= TP2_7D:
                        exit_flag, reason, xp = True, "MEDIUM_PROFIT", p['ep'] * (1 + TP2_7D)
                    elif pm >= TP3_7D:
                        exit_flag, reason, xp = True, "BIG_PROFIT", p['ep'] * (1 + TP3_7D)
                    elif pm > TP1_7D and price < p['mp'] * (1 - TRAIL_7D):
                        exit_flag, reason, xp = True, "TRAIL", p['mp'] * (1 - TRAIL_7D)
                    elif pm <= -vstop:
                        exit_flag, reason, xp = True, "STOP", price
                    elif i - p['ei'] >= HORIZON_7D:
                        exit_flag, reason, xp = True, "TIME", price
                    
                    if exit_flag:
                        ret = (xp - p['ep']) / p['ep']
                        nret = ret - (COST_PER_TRADE * 2)
                        pnl = p['pv'] * nret
                        cap_7d += pnl
                        peak_7d = max(peak_7d, cap_7d)
                        
                        emoji = "🎯" if pnl > 0 else "🛑"
                        logs_7d.append(f"{emoji} EXIT | {tm.strftime('%Y-%m-%d %H:%M:%S')} | ₹{xp:,.2f} | PL: {pnl:+,.0f} ({nret*100:+.2f}%) | [{reason}]")
                        
                        trades_7d.append({'ep': p['ep'], 'xp': xp, 'ret': nret, 'pnl': pnl, 'rsn': reason})
                        to_close.append(pidx)
                
                pos_7d = [p for idx, p in enumerate(pos_7d) if idx not in to_close]
                
                # Process entries
                if prob >= ENTRY_THRESHOLD_7D and i < len(prices_7d) - HORIZON_7D:
                    exp = sum(p['pf'] for p in pos_7d)
                    mexp = 1.5
                    
                    if exp < mexp:
                        edge = prob - ENTRY_THRESHOLD_7D
                        medge = 1.0 - ENTRY_THRESHOLD_7D if ENTRY_THRESHOLD_7D < 1.0 else 0.5
                        norm = edge / medge if medge > 0 else 0.4
                        pf = MIN_POS_7D + (MAX_POS_7D - MIN_POS_7D) * (norm ** 1.3)
                        pf = np.clip(pf, MIN_POS_7D, MAX_POS_7D)
                        pf *= (1.0 - exp / mexp * 0.3)
                        pf = max(pf, MIN_POS_7D * 0.5)
                        pv = cap_7d * pf
                        
                        pos_7d.append({'ei': i, 'ep': price, 'pv': pv, 'pf': pf, 'pb': prob, 'mp': price})
                        logs_7d.append(f"🟢 ENTRY #{len(pos_7d)} | {tm.strftime('%Y-%m-%d %H:%M:%S')} | ₹{price:,.2f} | Conf: {prob*100:.1f}% | Size: {pf*100:.0f}%")
                
                for p in pos_7d:
                    p['mp'] = max(p['mp'], price)
                
                equity_7d.append(cap_7d)
            
            # Close remaining
            for p in pos_7d:
                xp = prices_7d[-1]
                ret = (xp - p['ep']) / p['ep']
                nret = ret - (COST_PER_TRADE * 2)
                pnl = p['pv'] * nret
                cap_7d += pnl
                peak_7d = max(peak_7d, cap_7d)
                trades_7d.append({'ep': p['ep'], 'xp': xp, 'ret': nret, 'pnl': pnl, 'rsn': 'CLOSE'})
            
            # Results
            logs_7d.append("")
            logs_7d.append("=" * 80)
            logs_result = "\n".join(logs_7d)
            st.code(logs_result, language="text")
            
            ret_7d = (cap_7d / INITIAL_CAPITAL) - 1
            wins = sum(1 for t in trades_7d if t['pnl'] > 0)
            loss = len(trades_7d) - wins
            
            sum_7d = []
            sum_7d.append("=" * 80)
            sum_7d.append("📊 LIVE PAPER TRADING RESULTS (LAST 7 DAYS)")
            sum_7d.append("=" * 80)
            sum_7d.append(f"\n CAPITAL SUMMARY:")
            sum_7d.append(f"   Initial: ₹{INITIAL_CAPITAL:,.0f}")
            sum_7d.append(f"   Final: ₹{cap_7d:,.0f}")
            sum_7d.append(f"   Peak: ₹{peak_7d:,.0f}")
            sum_7d.append(f"   Net P&L: ₹{cap_7d - INITIAL_CAPITAL:+,.0f}")
            sum_7d.append(f"\n PERFORMANCE:")
            sum_7d.append(f"   Return: {ret_7d*100:+.2f}%")
            
            eq_arr = np.array(equity_7d)
            mdd = ((eq_arr / np.maximum.accumulate(eq_arr)) - 1).min()
            sum_7d.append(f"   Max DD: {mdd*100:.2f}%")
            
            rets = np.diff(eq_arr) / eq_arr[:-1]
            rets = rets[rets != 0]
            if len(rets) > 0 and rets.std() > 0:
                sharpe = (rets.mean() / rets.std()) * np.sqrt(252 * 6.5 * 60)
            else:
                sharpe = 0
            sum_7d.append(f"   Sharpe: {sharpe:.2f}")
            
            sum_7d.append(f"\n TRADING STATS:")
            sum_7d.append(f"   Total: {len(trades_7d)} trades")
            if len(trades_7d) > 0:
                sum_7d.append(f"   Wins: {wins} ({wins/len(trades_7d)*100:.1f}%)")
                sum_7d.append(f"   Loss: {loss} ({loss/len(trades_7d)*100:.1f}%)")
            
            sum_7d.append(f"\n EXIT BREAKDOWN:")
            reasons = {}
            for t in trades_7d:
                r = t['rsn']
                reasons[r] = reasons.get(r, 0) + 1
            for r in sorted(reasons.keys()):
                c = reasons[r]
                sum_7d.append(f"   {r:15} : {c:3} trades ({c/len(trades_7d)*100:5.1f}%)" if trades_7d else f"   {r:15} : {c:3} trades")
            
            sum_7d.append("=" * 80)
            
            if ret_7d > 0.10:
                verd = f" EXCELLENT! +{ret_7d*100:.2f}% return in 7 days!"
            elif ret_7d > 0.05:
                verd = f" VERY GOOD! +{ret_7d*100:.2f}% return in 7 days"
            elif ret_7d > 0:
                verd = f" PROFITABLE! +{ret_7d*100:.2f}% return in 7 days"
            elif ret_7d < 0:
                verd = f"  LOSS: {ret_7d*100:.2f}% on {len(trades_7d)} trades"
            else:
                verd = " NEUTRAL. 0.00% in 7 days"
            
            sum_7d.append(verd)
            sum_7d.append("=" * 80)
            
            st.code("\n".join(sum_7d), language="text")

# --------------------------------------------------
# TAB 5: Real-Time Market Trading (9:15 AM - 3:15 PM IST)
# --------------------------------------------------

with tab5:
    st.markdown("<h2 style='color: #667eea;'>🔴 Real-Time Live Market Trading (9:15 AM - 3:15 PM IST)</h2>", unsafe_allow_html=True)
    st.info("🔴 Live paper trading synchronized with NSE market hours. Fetches real-time data every minute.", icon="⏰")
    
    # IST timezone
    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if market is open
    is_market_open = market_start <= now_ist <= market_end and now_ist.weekday() < 5  # Mon-Fri
    
    col_market_status1, col_market_status2, col_market_status3 = st.columns(3)
    
    with col_market_status1:
        status_indicator = "🟢 OPEN" if is_market_open else "🔴 CLOSED"
        st.metric("Market Status", status_indicator)
    
    with col_market_status2:
        st.metric("Current Time (IST)", now_ist.strftime("%H:%M:%S"))
    
    with col_market_status3:
        if is_market_open:
            time_remaining = market_end - now_ist
            minutes_left = int(time_remaining.total_seconds() / 60)
            st.metric("Time Left", f"{minutes_left} min")
        else:
            st.metric("Next Open", market_start.strftime("%H:%M"))
    
    st.divider()
    
    # Real-time trading configuration
    st.subheader("⚙️ Real-Time Trading Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown("**Entry Configuration**")
        rt_entry_quantile = st.slider("Entry Threshold (Percentile)", 40, 90, 70, key="rt_entry_q")
        rt_min_confidence = st.slider("Minimum ML Confidence", 0.50, 0.75, 0.55, 0.01, key="rt_min_conf")
    
    with col_config2:
        st.markdown("**Exit Configuration**")
        rt_stop_loss = st.slider("Stop Loss %", 0.5, 3.0, 1.0, 0.1, key="rt_stop") / 100
        rt_take_profit = st.slider("Take Profit %", 0.2, 3.0, 0.8, 0.1, key="rt_tp") / 100
        rt_trail_stop = st.slider("Trailing Stop %", 0.1, 1.0, 0.2, 0.05, key="rt_trail") / 100
    
    col_config3, col_config4 = st.columns(2)
    
    with col_config3:
        st.markdown("**Position Sizing**")
        rt_min_position = st.slider("Min Position Size %", 1, 20, 5, key="rt_min_pos") / 100
        rt_max_position = st.slider("Max Position Size %", 10, 50, 15, key="rt_max_pos") / 100
    
    with col_config4:
        st.markdown("**Time Management**")
        rt_holding_period = st.slider("Holding Period (bars)", 5, 60, 15, key="rt_hold")
        rt_max_positions = st.slider("Max Open Positions", 1, 5, 3, key="rt_max_open")
    
    st.divider()
    
    # Start real-time simulation button
    if st.button("▶️ START REAL-TIME TRADING SIMULATION", key="start_rt_trading"):
        
        if not is_market_open:
            st.warning(f"⏰ Market is currently CLOSED. Trading resumes at {market_start.strftime('%H:%M IST')} on weekdays.")
            st.info("You can still run a simulation with historical data.")
        
        with st.spinner("🔄 Initializing real-time trading engine..."):
            
            # Fetch initial data
            try:
                end_date = datetime.now(IST)
                start_date = end_date - timedelta(hours=6)  # Last 6 hours of data
                
                try:
                    rt_data = yf.download(
                        "^NSEBANK",
                        start=start_date,
                        end=end_date,
                        interval="1m",
                        progress=False
                    )
                    rt_ticker = "^NSEBANK"
                except:
                    rt_data = yf.download(
                        "NIFTYBANK.NS",
                        start=start_date,
                        end=end_date,
                        interval="1m",
                        progress=False
                    )
                    rt_ticker = "NIFTYBANK.NS"
                
                # Convert timezone
                if rt_data.index.tz is None:
                    rt_data.index = rt_data.index.tz_localize("UTC").tz_convert(IST)
                else:
                    rt_data.index = rt_data.index.tz_convert(IST)
                
                # Fix columns
                if isinstance(rt_data.columns, pd.MultiIndex):
                    rt_data.columns = [col[0] for col in rt_data.columns]
                
                rt_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                rt_data = rt_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                
                if len(rt_data) < 50:
                    st.warning("⚠️ Not enough data available. Please try again when market has more activity.")
                    st.stop()
                
                st.success(f"✅ Fetched {len(rt_data)} candles from real-time data")
                
            except Exception as e:
                st.error(f"❌ Error fetching real-time data: {e}")
                st.stop()
        
        # Apply features
        with st.spinner("🔄 Processing features..."):
            try:
                rt_with_signals = add_scalping_signals(rt_data.copy())
                rt_with_features = add_basic_features(rt_with_signals.copy())
                rt_features = add_advanced_features(rt_with_features.copy())
                rt_features = rt_features.dropna()
                
                X_rt = rt_features[strategy_data['feature_cols']].copy()
                X_rt_scaled = strategy_data['scaler'].transform(X_rt)
                ml_prob_rt = get_ensemble_probability(
                    strategy_data['xgb_model'],
                    strategy_data['lgb_model'],
                    X_rt_scaled
                )
                
                rt_prices = rt_features["Close"].values
                rt_atr = rt_features["atr"].values
                
                st.success(f"✅ Features processed: {len(strategy_data['feature_cols'])} features")
                
            except Exception as e:
                st.error(f"❌ Error in feature pipeline: {e}")
                st.stop()
        
        # Real-time trading simulation
        with st.spinner("🔄 Running real-time trading simulation..."):
            
            RT_ENTRY_THRESHOLD = np.percentile(ml_prob_rt, rt_entry_quantile)
            RT_STOP_LOSS = rt_stop_loss
            RT_TAKE_PROFIT = rt_take_profit
            RT_TRAIL_STOP = rt_trail_stop
            RT_HOLDING_PERIOD = rt_holding_period
            RT_MIN_POS = rt_min_position
            RT_MAX_POS = rt_max_position
            RT_MAX_OPEN = rt_max_positions
            
            rt_capital = INITIAL_CAPITAL
            rt_peak_capital = rt_capital
            rt_positions = []
            rt_trades = []
            rt_equity_curve = [rt_capital]
            rt_logs = []
            
            # Simulation header
            rt_logs.append("=" * 90)
            rt_logs.append(f" REAL-TIME LIVE TRADING SIMULATION - {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
            rt_logs.append("=" * 90)
            rt_logs.append(f"\n✅ Data Summary:")
            rt_logs.append(f"   Candles: {len(rt_data)} (1-minute)")
            rt_logs.append(f"   Time Range: {rt_data.index[0]} → {rt_data.index[-1]}")
            rt_logs.append(f"   Price Range: ₹{rt_data['Low'].min():.2f} - ₹{rt_data['High'].max():.2f}")
            rt_logs.append(f"\n⚙️  Configuration:")
            rt_logs.append(f"   Entry Threshold: {RT_ENTRY_THRESHOLD:.4f} ({rt_entry_quantile}th percentile)")
            rt_logs.append(f"   Stop Loss: {RT_STOP_LOSS*100:.2f}%")
            rt_logs.append(f"   Take Profit: {RT_TAKE_PROFIT*100:.2f}%")
            rt_logs.append(f"   Trailing Stop: {RT_TRAIL_STOP*100:.3f}%")
            rt_logs.append(f"   Holding Period: {RT_HOLDING_PERIOD} bars")
            rt_logs.append(f"   Max Open Positions: {RT_MAX_OPEN}")
            rt_logs.append("")
            rt_logs.append(" 🟢 TRADES LOG:")
            rt_logs.append("-" * 90)
            
            # Main trading loop
            for i in range(len(rt_prices)):
                current_price = rt_prices[i]
                current_time = rt_features.index[i]
                current_ml_prob = ml_prob_rt[i]
                current_atr = rt_atr[i]
                
                # Warmup period
                if i < 20:
                    rt_equity_curve.append(rt_capital)
                    continue
                
                # Process existing positions
                positions_to_close = []
                
                for pos_idx, pos in enumerate(rt_positions):
                    price_move = (current_price - pos['entry_price']) / pos['entry_price']
                    vol_adj_stop = RT_STOP_LOSS * (1 + (current_atr / current_price) * 0.5)
                    vol_adj_stop = np.clip(vol_adj_stop, RT_STOP_LOSS * 0.8, RT_STOP_LOSS * 1.2)
                    
                    exit_hit = False
                    exit_reason = None
                    exit_price = current_price
                    
                    # Check profit targets
                    if price_move >= RT_TAKE_PROFIT:
                        exit_hit = True
                        exit_reason = "PROFIT_TARGET"
                        exit_price = pos['entry_price'] * (1 + RT_TAKE_PROFIT)
                    
                    # Trailing stop after partial profit
                    elif price_move > RT_TAKE_PROFIT * 0.5 and current_price < pos['max_price'] * (1 - RT_TRAIL_STOP):
                        exit_hit = True
                        exit_reason = "TRAILING_STOP"
                        exit_price = pos['max_price'] * (1 - RT_TRAIL_STOP)
                    
                    # Stop loss
                    elif price_move <= -vol_adj_stop:
                        exit_hit = True
                        exit_reason = "STOP_LOSS"
                        exit_price = current_price
                    
                    # Time-based exit
                    elif i - pos['entry_idx'] >= RT_HOLDING_PERIOD:
                        exit_hit = True
                        exit_reason = "TIME_EXIT"
                        exit_price = current_price
                    
                    if exit_hit:
                        ret = (exit_price - pos['entry_price']) / pos['entry_price']
                        net_ret = ret - (COST_PER_TRADE * 2)
                        pnl = pos['position_value'] * net_ret
                        
                        rt_capital += pnl
                        rt_peak_capital = max(rt_peak_capital, rt_capital)
                        
                        emoji = "🎯" if pnl > 0 else "🛑"
                        pnl_str = f"+₹{pnl:,.0f}" if pnl > 0 else f"-₹{abs(pnl):,.0f}"
                        ret_str = f"+{net_ret*100:.2f}%" if net_ret > 0 else f"{net_ret*100:.2f}%"
                        
                        rt_logs.append(f"{emoji} EXIT | {current_time.strftime('%H:%M:%S')} | ₹{exit_price:,.2f} | P&L: {pnl_str} ({ret_str}) | [{exit_reason}] | Capital: ₹{rt_capital:,.0f}")
                        
                        rt_trades.append({
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'prob': pos['ml_prob'],
                            'return': net_ret,
                            'pnl': pnl,
                            'reason': exit_reason,
                            'duration': i - pos['entry_idx']
                        })
                        
                        positions_to_close.append(pos_idx)
                
                rt_positions = [pos for idx, pos in enumerate(rt_positions) if idx not in positions_to_close]
                
                # Entry logic
                if current_ml_prob >= RT_ENTRY_THRESHOLD and current_ml_prob >= rt_min_confidence:
                    if i < len(rt_prices) - RT_HOLDING_PERIOD:
                        if len(rt_positions) < RT_MAX_OPEN:
                            total_exposure = sum(p['position_fraction'] for p in rt_positions)
                            
                            if total_exposure < (RT_MAX_OPEN * RT_MAX_POS):
                                conf_edge = current_ml_prob - RT_ENTRY_THRESHOLD
                                max_edge = 1.0 - RT_ENTRY_THRESHOLD
                                normalized = conf_edge / max_edge if max_edge > 0 else 0.4
                                
                                pos_frac = RT_MIN_POS + (RT_MAX_POS - RT_MIN_POS) * (normalized ** 1.3)
                                pos_frac = np.clip(pos_frac, RT_MIN_POS, RT_MAX_POS)
                                pos_frac *= (1.0 - total_exposure / (RT_MAX_OPEN * RT_MAX_POS) * 0.3)
                                pos_frac = max(pos_frac, RT_MIN_POS * 0.5)
                                
                                pos_value = rt_capital * pos_frac
                                
                                rt_positions.append({
                                    'entry_idx': i,
                                    'entry_price': current_price,
                                    'position_value': pos_value,
                                    'position_fraction': pos_frac,
                                    'ml_prob': current_ml_prob,
                                    'max_price': current_price
                                })
                                
                                prob_pct = current_ml_prob * 100
                                size_pct = pos_frac * 100
                                rt_logs.append(f"🟢 ENTRY #{len(rt_positions)} | {current_time.strftime('%H:%M:%S')} | ₹{current_price:,.2f} | Conf: {prob_pct:.1f}% | Size: {size_pct:.0f}% | Capital: ₹{rt_capital:,.0f}")
                
                # Update max prices for trailing stops
                for pos in rt_positions:
                    pos['max_price'] = max(pos['max_price'], current_price)
                
                rt_equity_curve.append(rt_capital)
            
            # Close remaining positions at session end
            for pos in rt_positions:
                exit_price = rt_prices[-1]
                ret = (exit_price - pos['entry_price']) / pos['entry_price']
                net_ret = ret - (COST_PER_TRADE * 2)
                pnl = pos['position_value'] * net_ret
                
                rt_capital += pnl
                rt_peak_capital = max(rt_peak_capital, rt_capital)
                
                rt_trades.append({
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'prob': pos['ml_prob'],
                    'return': net_ret,
                    'pnl': pnl,
                    'reason': 'SESSION_END',
                    'duration': 0
                })
            
            rt_logs.append("-" * 90)
            rt_logs.append("")
            
            # Results summary
            rt_return = (rt_capital / INITIAL_CAPITAL) - 1
            rt_equity_arr = np.array(rt_equity_curve)
            rt_max_dd = ((rt_equity_arr / np.maximum.accumulate(rt_equity_arr)) - 1).min()
            
            rt_winning = sum(1 for t in rt_trades if t['pnl'] > 0)
            rt_losing = len(rt_trades) - rt_winning
            
            rt_logs.append(" 📊 FINAL RESULTS:")
            rt_logs.append("-" * 90)
            rt_logs.append(f"   Starting Capital:   ₹{INITIAL_CAPITAL:,.0f}")
            rt_logs.append(f"   Ending Capital:     ₹{rt_capital:,.0f}")
            rt_logs.append(f"   Peak Capital:       ₹{rt_peak_capital:,.0f}")
            rt_logs.append(f"   Net P&L:            ₹{rt_capital - INITIAL_CAPITAL:+,.0f}")
            rt_logs.append("")
            rt_logs.append(f"   Return:             {rt_return*100:+.3f}%")
            rt_logs.append(f"   Max Drawdown:       {rt_max_dd*100:.2f}%")
            rt_logs.append(f"   Total Trades:       {len(rt_trades)}")
            rt_logs.append(f"   Winners:            {rt_winning} ({rt_winning/max(len(rt_trades),1)*100:.1f}%)")
            rt_logs.append(f"   Losers:             {rt_losing} ({rt_losing/max(len(rt_trades),1)*100:.1f}%)")
            
            if len(rt_trades) > 0:
                gross_profit = sum(t['pnl'] for t in rt_trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in rt_trades if t['pnl'] < 0))
                avg_win = gross_profit / rt_winning if rt_winning > 0 else 0
                avg_loss = gross_loss / rt_losing if rt_losing > 0 else 0
                pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)
                
                rt_logs.append(f"   Profit Factor:      {pf:.2f}" if pf != np.inf else f"   Profit Factor:      ∞")
                rt_logs.append(f"   Avg Win:            ₹{avg_win:,.0f}")
                rt_logs.append(f"   Avg Loss:           ₹{avg_loss:,.0f}")
            
            rt_logs.append("=" * 90)
            
            # Display logs
            logs_text = "\n".join(rt_logs)
            st.markdown(f"```\n{logs_text}\n```")
            
            # Summary cards
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            
            with col_summary1:
                st.metric("Final Capital", f"₹{rt_capital:,.0f}", f"{rt_return*100:+.2f}%")
            
            with col_summary2:
                st.metric("Max Drawdown", f"{rt_max_dd*100:.2f}%", "Risk Managed" if rt_max_dd > -0.10 else "High Risk")
            
            with col_summary3:
                st.metric("Win Rate", f"{rt_winning/max(len(rt_trades),1)*100:.1f}%", f"{len(rt_trades)} trades")
            
            with col_summary4:
                if len(rt_trades) > 0:
                    gross_profit = sum(t['pnl'] for t in rt_trades if t['pnl'] > 0)
                    gross_loss = abs(sum(t['pnl'] for t in rt_trades if t['pnl'] < 0))
                    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
                    st.metric("Profit Factor", f"{pf:.2f}", "Strong" if pf > 1.5 else "Weak" if pf < 1.0 else "Neutral")
                else:
                    st.metric("Profit Factor", "N/A", "No trades")
            
            st.divider()
            
            # Trades breakdown
            if rt_trades:
                st.subheader("📋 Detailed Trade Breakdown")
                trades_df = pd.DataFrame(rt_trades)
                trades_df['Entry'] = trades_df['entry_price'].apply(lambda x: f"₹{x:.2f}")
                trades_df['Exit'] = trades_df['exit_price'].apply(lambda x: f"₹{x:.2f}")
                trades_df['Confidence'] = trades_df['prob'].apply(lambda x: f"{x*100:.1f}%")
                trades_df['Return'] = trades_df['return'].apply(lambda x: f"{x*100:+.2f}%")
                trades_df['P&L'] = trades_df['pnl'].apply(lambda x: f"₹{x:,.0f}" if x >= 0 else f"-₹{abs(x):,.0f}")
                trades_df['Bars'] = trades_df['duration']
                
                display_trades = trades_df[['Entry', 'Exit', 'Confidence', 'Return', 'P&L', 'reason', 'Bars']]
                display_trades.columns = ['Entry Price', 'Exit Price', 'ML Conf', 'Return %', 'P&L', 'Exit Reason', 'Bars Held']
                
                st.dataframe(display_trades, use_container_width=True, hide_index=True)
            
            # Equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=np.arange(len(rt_equity_curve)),
                y=rt_equity_curve,
                mode='lines',
                name='Capital',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            fig_equity.update_layout(
                title='Capital Evolution During Real-Time Trading',
                xaxis_title='Candle Index',
                yaxis_title='Capital (₹)',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_equity, use_container_width=True)

st.divider()
st.markdown("""
<center>
    <p style='color: #888; font-size: 12px;'>
        Combined ML + Scalping Strategy | Ensemble Model | © 2026
    </p>
</center>
""", unsafe_allow_html=True)
