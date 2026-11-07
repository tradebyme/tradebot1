import logging
import requests
from typing import Dict, Any, List, Optional
from config import Config

logger = logging.getLogger("SMC_OandaAPI")

class OandaAPI:
    """OANDA v20 REST API Client for Strategy 3."""

    def __init__(self):
        self.api_key = Config.API_KEY
        self.account_id = Config.ACCOUNT_ID
        self.base_url = Config.BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Fetch real-time account financial summary."""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("account", {})
            logger.error(f"Failed to fetch account summary: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
        return None

    def get_candles(self, instrument: str, count: int = 50, granularity: str = "M15") -> List[Dict[str, Any]]:
        """Fetch latest candlestick history."""
        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M"
        }
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("candles", [])
            logger.error(f"Failed to fetch candles for {instrument}: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error fetching candles for {instrument}: {e}")
        return []

    def get_pricing(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Fetch live bid, ask, and spread."""
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                prices = resp.json().get("prices", [])
                if prices:
                    p = prices[0]
                    bids = p.get("bids", [])
                    asks = p.get("asks", [])
                    if bids and asks:
                        bid = float(bids[0]["price"])
                        ask = float(asks[0]["price"])
                        pip_scale = 0.10 if ("XAU" in instrument or "GOLD" in instrument) else (0.01 if "JPY" in instrument else 0.0001)
                        spread_pips = round((ask - bid) / pip_scale, 1)
                        return {
                            "instrument": instrument,
                            "bid": bid,
                            "ask": ask,
                            "spread_pips": spread_pips
                        }
        except Exception as e:
            logger.error(f"Error fetching pricing for {instrument}: {e}")
        return None

    def create_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> Optional[Dict[str, Any]]:
        """Execute market order with broker-side stopLossOnFill and takeProfitOnFill."""
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        decimals = 2 if ("XAU" in instrument or "GOLD" in instrument) else (3 if "JPY" in instrument else 5)

        data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{stop_loss:.{decimals}f}",
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": f"{take_profit:.{decimals}f}",
                    "timeInForce": "GTC"
                }
            }
        }
        try:
            resp = requests.post(url, headers=self.headers, json=data, timeout=12)
            if resp.status_code in (200, 201):
                logger.info(f"Order filled successfully for {instrument}: {units} units.")
                return resp.json()
            logger.error(f"Order placement failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")
        return None

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Fetch all currently open positions."""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("trades", [])
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
        return []

    def get_closed_trades(self, count: int = 15) -> List[Dict[str, Any]]:
        """Fetch recent closed trade history."""
        url = f"{self.base_url}/accounts/{self.account_id}/trades"
        params = {"state": "CLOSED", "count": count}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("trades", [])
        except Exception as e:
            logger.error(f"Error fetching closed trades: {e}")
        return []
