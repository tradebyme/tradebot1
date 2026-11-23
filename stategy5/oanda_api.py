import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from config import Config

logger = logging.getLogger("OandaAPI5")


class OandaClient:
    """
    Robust client wrapper for OANDA v20 REST API.
    Supports Forex (EUR/USD, GBP/USD, USD/JPY) and Commodities (XAU/USD Gold).
    """

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = Config.BASE_URL
        self.account_id = account_id or Config.ACCOUNT_ID
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OandaTrendEMAATRBot/5.0"
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

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        data: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute HTTP request with retries and exponential backoff."""
        url = f"{self.base_url}{endpoint}"
        retries = 0
        backoff = 1.0

        payload = json_data if json_data is not None else data

        while retries <= max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=payload if payload else None,
                    timeout=15.0
                )

                if response.status_code in (200, 201):
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited by OANDA API. Backing off {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    retries += 1
                elif response.status_code >= 500:  # Server side error
                    logger.warning(f"OANDA 5xx error ({response.status_code}). Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    retries += 1
                else:
                    err_msg = response.text
                    logger.error(f"OANDA API error {response.status_code}: {err_msg}")
                    try:
                        return response.json()
                    except Exception:
                        return {"error": err_msg, "status_code": response.status_code}

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(f"Network error ({e}). Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0
                retries += 1

        raise ConnectionError(f"Failed to execute {method} {endpoint} after {max_retries} retries.")

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch live account summary: Balance, NAV, Equity, Margin, Open Trades."""
        endpoint = f"/accounts/{self.account_id}/summary"
        data = self._request("GET", endpoint)
        account = data.get("account", {})
        return {
            "balance": float(account.get("balance", 0.0)),
            "nav": float(account.get("NAV", 0.0)),
            "unrealized_pl": float(account.get("unrealizedPL", 0.0)),
            "margin_used": float(account.get("marginUsed", 0.0)),
            "margin_available": float(account.get("marginAvailable", 0.0)),
            "open_trade_count": int(account.get("openTradeCount", 0)),
            "currency": account.get("currency", "USD")
        }

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Fetch all currently open individual trades."""
        endpoint = f"/accounts/{self.account_id}/openTrades"
        data = self._request("GET", endpoint)
        return data.get("trades", [])

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch open net positions across instruments."""
        endpoint = f"/accounts/{self.account_id}/openPositions"
        data = self._request("GET", endpoint)
        return data.get("positions", [])

    def get_closed_trades(self, count: int = 50) -> List[Dict[str, Any]]:
        """Fetch recently closed trades history."""
        endpoint = f"/accounts/{self.account_id}/trades"
        params = {"state": "CLOSED", "count": count}
        data = self._request("GET", endpoint, params=params)
        return data.get("trades", [])

    @staticmethod
    def get_pip_specs(instrument: str):
        """Returns (pip_size, pip_value_multiplier, precision_digits)."""
        inst = instrument.upper()
        if "XAU" in inst or "GOLD" in inst:
            return 0.10, 1.0, 2
        elif "JPY" in inst:
            return 0.01, 1000.0 / 150.0, 3
        return 0.0001, 10.0, 5

    def get_pricing(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch current Bid, Ask, Mid, and Spread in pips for multiple instruments."""
        if not instruments:
            return {}

        endpoint = f"/accounts/{self.account_id}/pricing"
        params = {"instruments": ",".join(instruments)}
        data = self._request("GET", endpoint, params=params)

        pricing = {}
        for price in data.get("prices", []):
            inst = price["instrument"]
            bids = price.get("bids", [])
            asks = price.get("asks", [])

            if bids and asks:
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                mid = (bid + ask) / 2.0
                pip_size, _, _ = self.get_pip_specs(inst)
                spread_pips = round((ask - bid) / pip_size, 1)

                pricing[inst] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pips": spread_pips,
                    "time": price.get("time")
                }

        return pricing

    def get_candles(self, instrument: str, granularity: str = "H4", count: int = 250) -> List[Dict[str, Any]]:
        """Fetch historical candlestick bars from OANDA (clamped to max 500 per v20 API spec)."""
        clamped_count = max(10, min(count, 500))
        endpoint = f"/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": clamped_count,
            "price": "M"  # Midpoint candles
        }
        data = self._request("GET", endpoint, params=params)
        return data.get("candles", [])

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a Market Order with immediate optional Stop Loss attached directly on OANDA.
        Guarantees that exits remain active on broker engine even if connection drops.
        """
        endpoint = f"/accounts/{self.account_id}/orders"
        _, _, precision = self.get_pip_specs(instrument)

        order_body = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "timeInForce": "FOK"
            }
        }

        if stop_loss is not None and stop_loss > 0:
            order_body["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.{precision}f}",
                "timeInForce": "GTC"
            }

        if take_profit is not None and take_profit > 0:
            order_body["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.{precision}f}",
                "timeInForce": "GTC"
            }

        sl_display = f"{stop_loss:.{precision}f}" if (stop_loss and stop_loss > 0) else "None"
        tp_display = f"{take_profit:.{precision}f}" if (take_profit and take_profit > 0) else "None"

        try:
            logger.info(
                f"Submitting Order: {instrument} Units: {units} | SL: {sl_display} | TP: {tp_display}"
            )
            res = self._request("POST", endpoint, json_data=order_body)

            if "orderFillTransaction" in res:
                fill = res["orderFillTransaction"]
                return {
                    "success": True,
                    "order_id": fill.get("orderID"),
                    "trade_id": fill.get("tradeOpened", {}).get("tradeID") or fill.get("id"),
                    "price": float(fill.get("price", 0.0)),
                    "units": int(fill.get("units", 0)),
                    "pl": float(fill.get("pl", 0.0)),
                    "raw": res
                }
            elif "orderCancelTransaction" in res:
                cancel = res["orderCancelTransaction"]
                reason = cancel.get("reason", "Unknown cancellation")
                logger.error(f"Order cancelled by OANDA: {reason}")
                return {"success": False, "error": f"Order Cancelled: {reason}", "raw": res}
            else:
                return {"success": False, "error": "Order not filled.", "raw": res}

        except Exception as e:
            logger.error(f"Failed to place order on {instrument}: {e}")
            return {"success": False, "error": str(e)}

    def update_trade_stop_loss(self, trade_id: str, new_stop_loss: float, instrument: str = "EUR_USD") -> Dict[str, Any]:
        """Update or ratchet the Stop Loss price of an open trade on OANDA."""
        endpoint = f"/accounts/{self.account_id}/trades/{trade_id}/orders"
        _, _, precision = self.get_pip_specs(instrument)

        payload = {
            "stopLoss": {
                "price": f"{new_stop_loss:.{precision}f}",
                "timeInForce": "GTC"
            }
        }
        try:
            res = self._request("PUT", endpoint, json_data=payload)
            logger.info(f"Updated Stop Loss for trade #{trade_id} ({instrument}) to {new_stop_loss:.{precision}f} on OANDA.")
            return {"success": True, "res": res}
        except Exception as e:
            logger.error(f"Failed to update Stop Loss for trade #{trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close an individual open trade on OANDA."""
        endpoint = f"/accounts/{self.account_id}/trades/{trade_id}/close"
        try:
            res = self._request("PUT", endpoint)
            logger.info(f"Closed trade #{trade_id} on OANDA.")
            return {"success": True, "res": res}
        except Exception as e:
            logger.error(f"Failed to close trade #{trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def close_all_trades(self) -> List[Dict[str, Any]]:
        """Emergency function to close all open trades on OANDA."""
        open_trades = self.get_open_trades()
        results = []
        for t in open_trades:
            tid = t.get("id")
            if tid:
                res = self.close_trade(tid)
                results.append({"trade_id": tid, "result": res})
        return results
