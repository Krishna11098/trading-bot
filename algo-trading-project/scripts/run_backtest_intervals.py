"""Run backtests for multiple intervals (1d, 5m, 1m) for a given ticker.
Usage: python scripts/run_backtest_intervals.py [TICKER]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.backtest import SimpleBacktester
from src.strategy.combined_strategy import generate_combined_strategy
from src.utils.config import DEFAULT_TICKERS, INDICATORS_DIR


def run_for_ticker(ticker: str, intervals=("", "_5m", "_1m")):
    results = {}
    for suffix in intervals:
        path = INDICATORS_DIR / f"{ticker}{suffix}_features.csv"
        if not path.exists():
            print(f"Skipping {ticker}{suffix}: features not found at {path}")
            continue

        print(f"\nLoaded features: {path}")
        df = __import__('pandas').read_csv(path, index_col=0, parse_dates=True)

        if 'combined_signal' not in df.columns:
            df = generate_combined_strategy(ticker, df, ml_weight=0.6)

        bt = SimpleBacktester()
        metrics = bt.backtest(df, signal_column='combined_signal')
        print(f"Results for {ticker}{suffix}: {metrics}")
        results[suffix or '_1d'] = metrics

    return results


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKERS[0]
    print(f"Running backtests for: {ticker}")
    all_results = run_for_ticker(ticker)
    print('\nSummary:')
    for k, v in all_results.items():
        print(k, v)
