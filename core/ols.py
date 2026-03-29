import numpy as np
import pandas as pd


def rolling_ols_beta(stock_returns, market_returns, window=252):
    """Estimate beta using a rolling-window OLS regression.

    For each date t, runs OLS on the window [t-window+1, t]:
        r_i = alpha + beta * r_m + epsilon

    Args:
        stock_returns: array of stock daily returns
        market_returns: array of market daily returns
        window: rolling window size in trading days (default 252 = 1 year)

    Returns:
        dict with keys: betas, alphas (arrays, same length as input; NaN before window fills)
    """
    stock_returns = np.asarray(stock_returns, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    n = len(stock_returns)

    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)

    for t in range(window - 1, n):
        y = stock_returns[t - window + 1 : t + 1]
        x = market_returns[t - window + 1 : t + 1]

        # OLS: beta = cov(x,y) / var(x), alpha = mean(y) - beta * mean(x)
        x_mean = x.mean()
        y_mean = y.mean()
        x_demean = x - x_mean
        beta = np.dot(x_demean, y - y_mean) / np.dot(x_demean, x_demean)
        alpha = y_mean - beta * x_mean

        betas[t] = beta
        alphas[t] = alpha

    return {
        "betas": betas,
        "alphas": alphas,
    }
