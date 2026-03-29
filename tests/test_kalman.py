import numpy as np
from core.kalman import kalman_beta


def test_kalman_converges():
    """Kalman beta should converge toward true beta on synthetic data."""
    np.random.seed(42)
    market = np.random.normal(0, 0.01, 500)
    stock = 1.3 * market + np.random.normal(0, 0.002, 500)
    result = kalman_beta(stock, market)
    tail = result["betas"][-100:]
    assert abs(np.mean(tail) - 1.3) < 0.1


def test_kalman_output_shape():
    """Output arrays should match input length."""
    n = 200
    result = kalman_beta(np.zeros(n), np.zeros(n))
    assert len(result["betas"]) == n
    assert len(result["alphas"]) == n
    assert len(result["beta_std"]) == n
