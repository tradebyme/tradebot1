import logging
import time
from typing import Dict, Any, List, Optional
import requests
from config import Config

logger = logging.getLogger("OandaAPI2")

class OandaClient:
    """OANDA v20 REST API Client for Strategy 2."""

    def __init__(self):
        self.api_key = Config.API_KEY
        self.account_id = Config.ACCOUNT_ID
        self.base_url = Config.BASE_URL
        
        if not self.api_key or not self.account_id:
            logger.warning("OANDA API credentials not fully configured in config/.env.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        })

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None, retries: int = 3) -> Dict[str, Any]:
        """Execute resilient HTTP request with exponential backoff."""
        url = f"{self.base_url}{endpoint}"
        last_err = None

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.request(method=method, url=url, params=params, json=json_data, timeout=10)
                if resp.status_code in (200, 201):
                    return resp.json()
                elif resp.status_code == 401:
                    raise PermissionError(f"OANDA API Error [401]: Insufficient authorization.")
                else:
                    logger.error(f"OANDA API [{resp.status_code}] on {method} {endpoint}: {resp.text}")
                    last_err = RuntimeError(f"OANDA API error [{resp.status_code}]: {resp.text}")
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning(f"Network error (attempt {attempt}/{retries}): {e}")
                last_err = e
                time.sleep(1.0 * attempt)

        raise last_err or RuntimeError("OANDA API request failed after retries.")

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch live account NAV, balance, and margin."""
        res = self._request("GET", f"/accounts/{self.account_id}/summary")
        return res.get("account", {})

    def get_candles(self, instrument: str, granularity: str = "M15", count: int = 120) -> List[Dict[str, Any]]:
        """Fetch historical candles for technical indicator calculation."""
        params = {"granularity": granularity, "count": count, "price": "M"}
        res = self._request("GET", f"/instruments/{instrument}/candles", params=params)
        return res.get("candles", [])

    def get_pricing(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch live Bid, Ask, and calculate Spread for instruments."""
        params = {"instruments": ",".join(instruments)}
        res = self._request("GET", f"/accounts/{self.account_id}/pricing", params=params)
        prices = {}
        for p in res.get("prices", []):
            inst = p["instrument"]
            bid = float(p["bids"][0]["price"]) if p.get("bids") else 0.0
            ask = float(p["asks"][0]["price"]) if p.get("asks") else 0.0
            pip_mult = 100.0 if "JPY" in inst else 10000.0
            spread = round((ask - bid) * pip_mult, 1)
            prices[inst] = {"bid": bid, "ask": ask, "spread_pips": spread, "time": p.get("time")}
        return prices

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Fetch all currently open positions."""
        res = self._request("GET", f"/accounts/{self.account_id}/openTrades")
        return res.get("trades", [])

    def get_closed_trades(self, count: int = 30) -> List[Dict[str, Any]]:
        """Fetch recently closed trades with realized P&L."""
        params = {"state": "CLOSED", "count": count}
        res = self._request("GET", f"/accounts/{self.account_id}/trades", params=params)
        return res.get("trades", [])

    def create_market_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> Dict[str, Any]:
        """
        Execute Market Order with OANDA server-side Stop Loss and Take Profit attached directly.
        Guarantees protection even if bot restarts.
        """
        precision = 3 if "JPY" in instrument else 5

        order_body = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "timeInForce": "FOK",
                "stopLossOnFill": {
                    "price": f"{stop_loss:.{precision}f}",
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": f"{take_profit:.{precision}f}",
                    "timeInForce": "GTC"
                }
            }
        }

        logger.info(f"Submitting MARKET Order: {instrument} Units={units} SL={stop_loss:.{precision}f} TP={take_profit:.{precision}f}")
        res = self._request("POST", f"/accounts/{self.account_id}/orders", json_data=order_body)
        
        fill = res.get("orderFillTransaction")
        if fill:
            logger.info(f"Order FILLED: Trade ID #{fill.get('id')} @ {fill.get('price')}")
            return fill
        else:
            cancel = res.get("orderCancelTransaction", {})
            reason = cancel.get("reason", "Unknown reason")
            logger.error(f"Order REJECTED/CANCELLED: {reason}")
            raise RuntimeError(f"Order failed: {reason}")

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close an active position immediately."""
        return self._request("PUT", f"/accounts/{self.account_id}/trades/{trade_id}/close")

