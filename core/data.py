import datetime
import numpy as np
import yfinance as yf
import pandas as pd
from cachetools import TTLCache

# Cache: up to 500 tickers, 12-hour TTL
_price_cache = TTLCache(maxsize=500, ttl=12 * 3600)

# ^IRX is the 13-week Treasury Bill rate (annualized, in percentage points)
RF_TICKER = "^IRX"


def fetch_prices(ticker, benchmark="^GSPC", years=7):
    """Download adjusted close prices for a stock, benchmark, and risk-free rate.

    Args:
        ticker: stock ticker symbol (e.g. "AAPL")
        benchmark: market index ticker (default S&P 500)
        years: how many years of history to fetch

    Returns:
        pd.DataFrame with columns ["stock", "market", "rf"] indexed by date.
        "rf" is the daily risk-free rate (decimal, not percentage).

    Raises:
        ValueError: if ticker data cannot be fetched
    """
    cache_key = (ticker.upper(), benchmark, years, datetime.date.today().isoformat())
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(years * 365.25) + 30)

    tickers_str = f"{ticker} {benchmark} {RF_TICKER}"
    data = yf.download(tickers_str, start=start.isoformat(), end=end.isoformat(),
                       auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    close = data["Close"]

    if isinstance(close, pd.Series):
        raise ValueError(f"No data found for ticker '{ticker}'")

    if ticker.upper() not in close.columns and ticker not in close.columns:
        raise ValueError(f"No data found for ticker '{ticker}'")

    stock_col = ticker.upper() if ticker.upper() in close.columns else ticker
    bench_col = benchmark

    prices = pd.DataFrame({
        "stock": close[stock_col],
        "market": close[bench_col],
    })

    # Risk-free rate: ^IRX gives annualized % (e.g. 4.5 = 4.5%)
    # Convert to daily decimal rate: (1 + rate/100)^(1/252) - 1
    if RF_TICKER in close.columns:
        irx = close[RF_TICKER].reindex(prices.index)
        # Forward-fill missing days, then fill any remaining NaN with 0
        irx = irx.ffill().fillna(0)
        prices["rf"] = (1 + irx / 100) ** (1 / 252) - 1
    else:
        # Fallback: if ^IRX not available, use 0 (same as raw returns)
        prices["rf"] = 0.0

    prices = prices.dropna(subset=["stock", "market"])

    if len(prices) < 60:
        raise ValueError(f"Insufficient data for ticker '{ticker}' (need at least 60 trading days)")

    _price_cache[cache_key] = prices
    return prices


def compute_returns(prices):
    """Convert price DataFrame to daily returns and compute excess returns.

    Returns:
        pd.DataFrame with columns:
            stock_excess: stock return minus risk-free rate
            market_excess: market return minus risk-free rate
            rf: daily risk-free rate
    """
    daily_ret = prices[["stock", "market"]].pct_change()
    rf = prices["rf"]

    result = pd.DataFrame({
        "stock_excess": daily_ret["stock"] - rf,
        "market_excess": daily_ret["market"] - rf,
        "rf": rf,
    }).iloc[1:]  # drop first row (NaN from pct_change)

    return result.dropna()
