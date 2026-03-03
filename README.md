# Algo Trading Bot

A production‑ready **algorithmic trading bot** designed for **intraday & swing trading** using a hybrid of **rule‑based technical strategies + machine learning models**. This project focuses on **robust backtesting, realistic paper trading, performance analytics, and extensibility**.


##  Table of Contents

- Overview
- Features
- System Architecture
- Tech Stack & Libraries Used
- Story behind ml model selection
- Installation & Setup
- How to Start the Bot
- Strategy Logic Explained
- Machine Learning Models
- Performance Metrics
- Operating Modes
- Logging & Debugging
- Disclaimer

---

##  Overview

This bot is designed to:

* Trade **index instruments (e.g., Bank NIFTY, NIFTY)**
* Combine **multiple alpha sources** into a single weighted decision
* Run **backtests, paper trading, and simulation** using identical logic
* Be **modular**, **auditable**, and **strategy‑agnostic**


##  Features

 Hybrid Rule‑Based (NOT entirely scalping or positional) + ML Strategy
 Intraday & multi‑day support
 Realistic transaction costs
 Walk‑forward friendly architecture
 No repainting indicators
 Config‑driven strategy weights
 Performance & drawdown tracking
 Safe paper trading mode

---

##  System Architecture

```
Data Source (yfinance)
        ↓
Feature Engineering (TA + price action)
        ↓
Rule‑Based Strategies ──┐
                         ├── Signal Fusion Engine → Risk Filter → Order Engine
ML Prediction Engine ───┘
        ↓
Backtest / Paper Trade 
        ↓
Performance Analytics
```

---

##  Tech Stack & Libraries Used

###  Core

* Python 3.10+
* Jupyter / VS Code

###  Data & Indicators

* `pandas`
* `numpy`
* `yfinance`
* `ta-lib` 

###  Machine Learning

* `scikit-learn`
* `xgboost` 
* `lightgbm`


###  Utilities

* `datetime`
* `logging`
* `tqdm`

---

## Story behind model selection
The machine learning model selection process evolved through multiple iterations, driven by empirical performance and the realities of high-frequency (1-minute) market data.
We initially experimented with a Random Forest Classifier as a baseline model due to its simplicity and robustness on tabular data. However, despite extensive feature engineering, the model failed to achieve satisfactory predictive accuracy and struggled to generalize well across different market conditions.
To better capture temporal dependencies, we then explored LSTM (Long Short-Term Memory) networks. While LSTM showed improved performance compared to Random Forest during experimentation, further analysis revealed a critical risk: 1-minute candle data is inherently noisy, and deep sequence models tend to overfit microstructure noise rather than learn durable market signals. This made the approach unstable and less reliable for live or paper trading environments.

Based on these observations, we transitioned to a gradient boosting ensemble, combining:
XGBoost (55%)
LightGBM (45%)

Both models are well known for their strong performance on noisy, non-stationary financial data, offering an effective balance between stability and adaptability. The weighted ensemble reduces model-specific bias and provides more consistent predictions across varying market regimes.
In the final system, the ML ensemble acts as a confirmation and confidence layer, complementing rule-based strategies rather than replacing them.
##  Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/nkmehta216/SnowMore.git
cd algo-trading-bot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Start the Bot

### 🔹 Notebook Exploration
    run the above notebook, everything is present in the same notebook.
```bash
jupyter notebook 06_combined_strategy.ipynb
```

---

##  Strategy Logic Explained

### Rule‑Based Strategies Include:

* RSI mean reversion
* EMA/VWAP trend alignment
* Price momentum confirmation
* Volatility contraction breakout

Each strategy outputs:

```
Signal ∈ {‑1, 0, +1}
Confidence ∈ [0, 1]
```

Signals are **weighted and fused**:

```
Final Signal = Σ(weightᵢ × signalᵢ)
```

---

##  Machine Learning Models

### Model Objective

Predict **next candle direction & strength**.

### Features

* RSI, MACD, ATR
* Rolling returns
* Volatility regime
* Volume delta

### Training Flow

1. Historical data collection
2. Feature scaling
3. Train/test split (time‑aware)
4. Model persistence (`joblib`)

ML output is **never used alone**, only as confirmation.

---

##  Performance Metrics

  ### FOR 1 YEAR BACKTESTING (ON KAGGLE DATA):
  
    ## 🎯 Backtest Results

### 📈 1-Year Backtesting (Kaggle Data)

**Initial Capital:** ₹1,000,000  
**Final Capital:** ₹1,151,022  
**Total Return:** **15.10%**  
**Net Profit:** ₹151,022  

**Max Drawdown:** -2.63%  
**Sharpe Ratio:** 3.51  
**Profit Factor:** 1.17  
**Win Rate:** 49.18%  

---

## 🧪 Paper Trading – 6 January 2026

### 💰 Capital Summary

- **Starting Capital:** ₹1,000,000  
- **Ending Capital:** ₹1,000,344  
- **Peak Capital:** ₹1,000,486  
- **Net P&L:** **₹+344**

### 📊 Performance Metrics

- **Daily Return:** +0.034%  
- **Max Drawdown:** -0.02%   
- **Profit Factor:** 2.10  

### 📉 Trading Statistics

- **Total Trades:** 29  
- **Winning Trades:** 18 (62.1%)  
- **Losing Trades:** 11 (37.9%)  






---

##  Operating Modes

| Mode     | Description                      |
| -------- | -------------------------------- |
| Backtest | Historical simulation            |
| Paper    | Live data, no capital            |

---


##  Logging & Debugging

Logs include:

* Entry/exit reason
* Indicator snapshot
* ML confidence
* P&L per trade



##  Disclaimer

⚠ **Educational purpose only**

 We tried our best to make model learn trading , but there can be some stage where model may give negative returns on one particular day (usually when market is down) , but we managed to make the loss less than - 1 % per day (if loss is there) and in long run we can confirm it will give positive returns.

---

###  If you found this useful, consider starring the repo

Happy Trading !
