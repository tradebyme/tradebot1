import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from config import Config

logger = logging.getLogger("OandaAPI4")


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
            "User-Agent": "OandaAdaptiveSmootherBot/1.0"
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
        data: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute HTTP request with retries and exponential backoff."""
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

                if response.status_code in (200, 201):
                    return response.json()

                # Handle Rate Limiting (429) or Server Errors (500, 502, 503, 504)
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        f"OANDA API returned {response.status_code} for {endpoint}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    retries += 1
                    continue

                # Client errors (400, 401, 403, 404)
                error_body = response.text
                try:
                    error_json = response.json()
                    error_message = (
                        error_json.get("errorMessage")
                        or error_json.get("message")
                        or error_body
                    )
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

    def get_pricing(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetch current Bid, Ask, Mid, and Spread in pips for multiple instruments.
        """
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
                pip_size = 0.01 if "JPY" in inst else (0.10 if "XAU" in inst else 0.0001)
                spread_pips = round((ask - bid) / pip_size, 1)

                pricing[inst] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pips": spread_pips,
                    "time": price.get("time")
                }

        return pricing

    def get_candles(self, instrument: str, granularity: str = "M15", count: int = 120) -> List[Dict[str, Any]]:
        """Fetch historical candlestick bars from OANDA."""
        endpoint = f"/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
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
        Execute a Market Order with immediate optional Stop Loss and Take Profit attached directly on OANDA broker engine.
        Guarantees that exits remain active even if the bot loses connection.
        """
        endpoint = f"/accounts/{self.account_id}/orders"

        # Determine precision based on instrument
        pip_unit = 0.01 if "JPY" in instrument else (0.10 if "XAU" in instrument else 0.0001)
        precision = 3 if pip_unit >= 0.01 else 5

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
                f"Submitting Market Order: {instrument} Units: {units} | SL: {sl_display} | TP: {tp_display}"
            )
            res = self._request("POST", endpoint, data=order_body)

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

    def update_trade_stop_loss(self, trade_id: str, new_stop_loss: float) -> Dict[str, Any]:
        """Update or ratchet the Stop Loss price of an open trade on OANDA."""
        endpoint = f"/accounts/{self.account_id}/trades/{trade_id}/orders"
        payload = {
            "stopLoss": {
                "price": f"{new_stop_loss:.5f}",
                "timeInForce": "GTC"
            }
        }
        try:
            res = self._request("PUT", endpoint, json_data=payload)
            logger.info(f"Updated Stop Loss for trade #{trade_id} to {new_stop_loss:.5f} on OANDA.")
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

