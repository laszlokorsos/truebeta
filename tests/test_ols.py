import numpy as np
from core.ols import rolling_ols_beta


def test_perfect_beta():
    """If stock = 2 * market, beta should be exactly 2.0."""
    np.random.seed(42)
    market = np.random.normal(0, 0.01, 300)
    stock = 2.0 * market
    result = rolling_ols_beta(stock, market, window=60)
    valid = result["betas"][59:]
    assert np.allclose(valid, 2.0, atol=1e-10)


def test_nan_before_window():
    """Betas before the window fills should be NaN."""
    np.random.seed(42)
    market = np.random.normal(0, 0.01, 100)
    stock = market * 1.5
    result = rolling_ols_beta(stock, market, window=50)
    assert np.all(np.isnan(result["betas"][:49]))
    assert not np.isnan(result["betas"][49])


def test_zero_variance_market():
    """Constant market returns should yield NaN beta, not crash."""
    market = np.zeros(100)
    stock = np.random.normal(0, 0.01, 100)
    result = rolling_ols_beta(stock, market, window=50)
    assert np.all(np.isnan(result["betas"][49:]))
