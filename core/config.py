import os

# Environment
DEBUG = os.getenv("TRUEBETA_DEBUG", "false").lower() in ("1", "true", "yes")

# Cache
PRICE_CACHE_MAXSIZE = 500
PRICE_CACHE_TTL = 3600          # 1 hour
SP500_CACHE_TTL = 24 * 3600     # 24 hours

# Trading
TRADING_DAYS_PER_YEAR = 252
MIN_TRADING_DAYS = 60

# yfinance
YFINANCE_TIMEOUT = 30           # seconds

# Rate limiting
RATE_LIMIT_DEFAULT = "30/minute"
RATE_LIMIT_SEARCH = "60/minute"
