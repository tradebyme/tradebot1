import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from config import Config

logger = logging.getLogger("OandaAPI")

class OandaClient:
    """Robust client wrapper for OANDA v20 REST API."""

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = Config.BASE_URL
        self.account_id = account_id or Config.ACCOUNT_ID
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OandaForexBot/1.0"
        })

        if not self.account_id:
            self._auto_discover_account()

    def _auto_discover_account(self):
        """Auto-discover the primary account ID if not provided in config."""
        logger.info("Account ID not specified, discovering accounts from OANDA...")
        try:
            res = self._request("GET", "/accounts")
            accounts = res.get("accounts", [])
            if not accounts:
                raise ValueError("No accounts found for this OANDA API key.")
            self.account_id = accounts[0]["id"]
            logger.info(f"Discovered primary Account ID: {self.account_id}")
        except Exception as e:
            logger.error(f"Failed to auto-discover account ID: {e}")
            raise

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, max_retries: int = 3) -> Dict[str, Any]:
        """Execute HTTP request with retries and timeout."""
        url = f"{self.base_url}{endpoint}"
        retries = 0
        backoff = 1.0

        while retries <= max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data if data is not None else None,
                    timeout=10
                )

                if response.status_code == 200 or response.status_code == 201:
                    return response.json()

                # Handle Rate Limiting (429) or Server Errors (500, 502, 503, 504)
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"OANDA API returned {response.status_code} for {endpoint}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    retries += 1
                    continue

                # Client error (400, 401, 403, 404)
                error_body = response.text
                try:
                    error_json = response.json()
                    error_message = error_json.get("errorMessage") or error_json.get("message") or error_body
                except Exception:
                    error_message = error_body

                logger.error(f"OANDA API Error [{response.status_code}] on {method} {endpoint}: {error_message}")
                raise RuntimeError(f"OANDA API Error [{response.status_code}]: {error_message}")

            except requests.RequestException as e:
                logger.warning(f"Network error on {method} {url}: {e}. Retrying ({retries}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2
                retries += 1

        raise RuntimeError(f"Failed {method} {endpoint} after {max_retries} retries.")

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch account balance, NAV, unrealized PnL, margin rate."""
        endpoint = f"/accounts/{self.account_id}/summary"
        data = self._request("GET", endpoint)
        return data.get("account", {})

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Fetch currently open trades."""
        endpoint = f"/accounts/{self.account_id}/openTrades"
        data = self._request("GET", endpoint)
        return data.get("trades", [])

    def get_closed_trades(self, count: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent closed trades."""
        endpoint = f"/accounts/{self.account_id}/trades"
        params = {"state": "CLOSED", "count": count}
        data = self._request("GET", endpoint, params=params)
        return data.get("trades", [])

    def get_candles(self, instrument: str, granularity: str = "M15", count: int = 100) -> List[Dict[str, Any]]:
        """Fetch historical candlestick data with Mid, Bid, and Ask prices."""
        endpoint = f"/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": "MBA" # Mid, Bid, Ask
        }
        data = self._request("GET", endpoint, params=params)
        return data.get("candles", [])

    def get_pricing(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch live bid, ask, and spread for a list of instruments."""
        endpoint = f"/accounts/{self.account_id}/pricing"
        params = {"instruments": ",".join(instruments)}
        data = self._request("GET", endpoint, params=params)
        
        result = {}
        for price in data.get("prices", []):
            inst = price["instrument"]
            bid = float(price["closeoutBid"])
            ask = float(price["closeoutAsk"])
            if "XAU" in inst or "GOLD" in inst:
                pip_size = 0.10
            elif "JPY" in inst:
                pip_size = 0.01
            else:
                pip_size = 0.0001
            spread_pips = round((ask - bid) / pip_size, 2)
            result[inst] = {
                "bid": bid,
                "ask": ask,
                "mid": round((bid + ask) / 2, 5),
                "spread_pips": spread_pips,
                "pip_size": pip_size
            }
        return result

    def place_market_order(self, instrument: str, units: int, stop_loss_price: float, take_profit_price: float = 0.0, client_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Place a MARKET order with attached Stop Loss and optional Take Profit.
        units > 0 = BUY (Long), units < 0 = SELL (Short).
        Tags order with client_tag (e.g. APEX_SNIPER) for permanent isolation on OANDA.
        """
        endpoint = f"/accounts/{self.account_id}/orders"
        
        # Determine precision based on instrument specs
        _, _, precision = self.get_pip_specs(instrument)
        sl_str = f"{stop_loss_price:.{precision}f}"

        order_data = {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "timeInForce": "FOK", # Fill Or Kill
            "stopLossOnFill": {
                "price": sl_str,
                "timeInForce": "GTC"
            }
        }

        if take_profit_price is not None and take_profit_price > 0:
            tp_str = f"{take_profit_price:.{precision}f}"
            order_data["takeProfitOnFill"] = {
                "price": tp_str,
                "timeInForce": "GTC"
            }
            tp_display = tp_str
        else:
            tp_display = "None (Trailing ATR)"

        if client_tag:
            tag_id = f"{client_tag[:10]}_{int(time.time())}"
            order_data["clientExtensions"] = {"id": tag_id, "tag": client_tag, "comment": client_tag}
            order_data["tradeClientExtensions"] = {"id": tag_id, "tag": client_tag, "comment": client_tag}

        order_payload = {"order": order_data}

        logger.info(f"Submitting MARKET Order: {instrument} Units={units} SL={sl_str} TP={tp_display} Tag={client_tag}")
        res = self._request("POST", endpoint, data=order_payload)
        
        # Check if filled
        if "orderFillTransaction" in res:
            fill = res["orderFillTransaction"]
            trade_opened = fill.get("tradeOpened", {})
            trade_id = trade_opened.get("tradeID") or fill.get("id")
            price = fill.get("price")
            logger.info(f"Order FILLED: Trade ID #{trade_id} @ {price}")
            return {
                "success": True,
                "trade_id": trade_id,
                "price": price,
                "fill": fill
            }
        elif "orderCancelTransaction" in res:
            reason = res["orderCancelTransaction"].get("reason", "Unknown cancellation")
            logger.warning(f"Order CANCELLED: {reason}")
            return {"success": False, "error": reason}
        
        return {"success": True, "raw": res}

    def get_pip_specs(self, instrument: str) -> Tuple[float, float, int]:
        """
        Returns (pip_size, pip_value_factor, price_precision).
        XAU_USD: pip_size=0.10, precision=2
        JPY pairs: pip_size=0.01, precision=3
        Standard pairs: pip_size=0.0001, precision=5
        """
        if instrument == "XAU_USD":
            return 0.10, 1.0, 2
        elif "JPY" in instrument:
            return 0.01, 100.0, 3
        else:
            return 0.0001, 10000.0, 5

    def modify_trade_stop_loss(self, trade_id: str, new_stop_loss: float, instrument: str) -> Dict[str, Any]:
        """Modify or ratchet the Stop Loss price on an existing active trade on OANDA."""
        endpoint = f"/accounts/{self.account_id}/trades/{trade_id}/orders"
        _, _, precision = self.get_pip_specs(instrument)
        body = {
            "stopLoss": {
                "price": f"{new_stop_loss:.{precision}f}",
                "timeInForce": "GTC"
            }
        }
        return self._request("PUT", endpoint, data=body)

    def close_trade(self, trade_id: str, units: Optional[str] = "ALL") -> Dict[str, Any]:
        """Close an open trade by its trade ID."""
        endpoint = f"/accounts/{self.account_id}/trades/{trade_id}/close"
        payload = {"units": units}
        logger.info(f"Closing trade #{trade_id}...")
        return self._request("PUT", endpoint, data=payload)

    def close_all_trades(self) -> List[Dict[str, Any]]:
        """Emergency function: Close all open trades immediately."""
        open_trades = self.get_open_trades()
        results = []
        for trade in open_trades:
            trade_id = trade["id"]
            try:
                res = self.close_trade(trade_id)
                results.append({"trade_id": trade_id, "success": True, "res": res})
            except Exception as e:
                logger.error(f"Failed to close trade #{trade_id}: {e}")
                results.append({"trade_id": trade_id, "success": False, "error": str(e)})
        return results

