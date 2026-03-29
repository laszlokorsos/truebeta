import io
import requests
import pandas as pd
from cachetools import TTLCache

from core.config import SP500_CACHE_TTL

_sp500_cache = TTLCache(maxsize=1, ttl=SP500_CACHE_TTL)
_CACHE_KEY = "sp500"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) TrueBeta/1.0"
}


def get_sp500_tickers():
    """Fetch S&P 500 constituents from Wikipedia.

    Returns:
        list of dicts: [{"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"}, ...]
    """
    if _CACHE_KEY in _sp500_cache:
        return _sp500_cache[_CACHE_KEY]

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    df = tables[0]

    result = []
    for _, row in df.iterrows():
        result.append({
            "ticker": row["Symbol"].replace(".", "-"),  # BRK.B -> BRK-B (Yahoo format)
            "name": row["Security"],
            "sector": row.get("GICS Sector", ""),
        })

    _sp500_cache[_CACHE_KEY] = result
    return result
