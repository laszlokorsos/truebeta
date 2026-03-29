import concurrent.futures
import datetime
import numpy as np
import yfinance as yf
import pandas as pd
from cachetools import TTLCache

from core.config import PRICE_CACHE_MAXSIZE, PRICE_CACHE_TTL, MIN_TRADING_DAYS, YFINANCE_TIMEOUT

_price_cache = TTLCache(maxsize=PRICE_CACHE_MAXSIZE, ttl=PRICE_CACHE_TTL)

# ^IRX is the 13-week Treasury Bill rate (annualized, in percentage points)
RF_TICKER = "^IRX"


def _download_with_timeout(tickers_str, start, end, timeout):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            yf.download, tickers_str, start=start, end=end,
            auto_adjust=True, progress=False,
        )
        return future.result(timeout=timeout)


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
    cache_key = (ticker.upper(), benchmark, years)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(years * 365.25) + 30)

    tickers_str = f"{ticker} {benchmark} {RF_TICKER}"
    try:
        data = _download_with_timeout(tickers_str, start.isoformat(), end.isoformat(), YFINANCE_TIMEOUT)
    except concurrent.futures.TimeoutError:
        raise ValueError(f"Timeout fetching data for ticker '{ticker}'")

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

    if len(prices) < MIN_TRADING_DAYS:
        raise ValueError(f"Insufficient data for ticker '{ticker}' (need at least {MIN_TRADING_DAYS} trading days)")

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
