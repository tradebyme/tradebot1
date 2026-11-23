import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import Config
from oanda_api import OandaClient
from risk_manager import RiskManager
from strategy import TrendEMAATRStrategy

logger = logging.getLogger("TradingBot5")


class TradingBot:
    """
    Strategy 5: Institutional Trend-ATR Portfolio (ITAP) Bot Engine
    Autonomous multi-asset coordinator running in background thread.
    """

    def __init__(self):
        self.client = OandaClient()
        self.risk_manager = RiskManager()
        self.strategy = TrendEMAATRStrategy()

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.known_closed_trade_ids = set()
        self.log_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        # Dynamic Peak Trailing State Tracking
        # Format: {trade_id: {"instrument": str, "direction": str, "entry_price": float, "peak_price": float, "atr": float, "current_sl": float}}
        self.trailing_states: Dict[str, Dict[str, Any]] = {}

        self._seed_closed_trades()
        self.log("INFO", "Institutional Trend-ATR Portfolio Engine (Strategy 5) initialized successfully.")

    def log(self, level: str, message: str):
        """Append log entry for Web UI feed and console logger."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        entry = {"time": timestamp, "level": level, "message": message}

        with self._lock:
            self.log_history.append(entry)
            if len(self.log_history) > 250:
                self.log_history = self.log_history[-250:]

        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)

    def _seed_closed_trades(self):
        """
        Record existing closed trades on startup so container restarts on Render
        never double-count historical trades or lose prop challenge progress.
        """
        try:
            closed = self.client.get_closed_trades(count=60)
            for t in closed:
                tid = t.get("id")
                if tid:
                    self.known_closed_trade_ids.add(tid)
            self.log("INFO", f"Seeded {len(self.known_closed_trade_ids)} historical closed trade IDs from OANDA.")
        except Exception as e:
            self.log("WARNING", f"Could not seed closed trades: {e}")

    def start_background(self):
        """Launch autonomous trading loop in background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.log("INFO", "Autonomous trading loop started in background.")

    def stop_background(self):
        """Pause trading loop."""
        self.is_running = False
        self.log("INFO", "Trading loop stop signal delivered.")

    def _run_loop(self):
        """Core polling loop executing every 20-30 seconds."""
        while self.is_running:
            try:
                self._check_closed_trades()
                self._update_trailing_stops()
                self._scan_for_signals()
            except Exception as e:
                self.log("ERROR", f"Exception in trading loop cycle: {e}")

            time.sleep(25)

    def _check_closed_trades(self):
        """Detect any newly closed trades on OANDA and notify RiskManager."""
        try:
            recent_closed = self.client.get_closed_trades(count=20)
            for t in recent_closed:
                tid = t.get("id")
                if tid and tid not in self.known_closed_trade_ids:
                    self.known_closed_trade_ids.add(tid)
                    realized_pl = float(t.get("realizedPL", 0.0))
                    inst = t.get("instrument", "UNKNOWN")
                    self.log("INFO", f"Trade #{tid} ({inst}) closed on OANDA. Realized P/L: ${realized_pl:+.2f}")
                    self.risk_manager.record_trade_result(tid, inst, realized_pl)
                    # Remove from trailing tracker if present
                    if tid in self.trailing_states:
                        del self.trailing_states[tid]
        except Exception as e:
            self.log("WARNING", f"Error checking closed trades: {e}")

    def _update_trailing_stops(self):
        """
        Tick-by-tick / bar-by-bar Dynamic Peak Trailing Stop:
        Ratchets the Stop Loss at (Peak Price - 3.0 * ATR) as price moves in profit.
        """
        try:
            open_trades = self.client.get_open_trades()
            pricing = self.client.get_pricing(Config.INSTRUMENTS)

            for t in open_trades:
                tid = t.get("id")
                inst = t.get("instrument")
                current_units = int(t.get("currentUnits", 0))
                open_price = float(t.get("price", 0.0))
                stop_loss_order = t.get("stopLossOrder", {})
                current_sl = float(stop_loss_order.get("price", 0.0)) if stop_loss_order else 0.0

                if not inst or inst not in pricing:
                    continue

                curr_mid = pricing[inst]["mid"]
                direction = "LONG" if current_units > 0 else "SHORT"

                # Initialize tracking state if not present
                if tid not in self.trailing_states:
                    candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME_MAPPING.get(inst, "H4"), count=30)
                    df = self.strategy.parse_candles(candles)
                    atr = self.strategy.compute_indicators(df).iloc[-2]["atr"] if len(df) >= 20 else 0.0020

                    self.trailing_states[tid] = {
                        "instrument": inst,
                        "direction": direction,
                        "entry_price": open_price,
                        "peak_price": curr_mid,
                        "atr": atr,
                        "current_sl": current_sl
                    }

                state = self.trailing_states[tid]
                atr = state["atr"]

                # Update peak high / trough low
                if direction == "LONG":
                    if curr_mid > state["peak_price"]:
                        state["peak_price"] = curr_mid

                    profit_move = state["peak_price"] - open_price
                    # Only activate trailing ratchet if moved into profit by at least 1.0x ATR
                    if profit_move >= Config.TRAIL_ACTIVATION_ATR * atr:
                        new_sl = state["peak_price"] - (Config.ATR_TRAIL_MULTIPLIER * atr)
                        # Ratchet upwards only (never move SL down)
                        if new_sl > state["current_sl"] and new_sl > (current_sl + 0.0002):
                            self.log("INFO", f"Ratcheting Trailing SL on {inst} Long #{tid}: {state['current_sl']:.4f} -> {new_sl:.4f}")
                            res = self.client.update_trade_stop_loss(tid, new_sl, instrument=inst)
                            if res.get("success"):
                                state["current_sl"] = new_sl

                elif direction == "SHORT":
                    if curr_mid < state["peak_price"]:
                        state["peak_price"] = curr_mid

                    profit_move = open_price - state["peak_price"]
                    # Only activate trailing ratchet if moved into profit by at least 1.0x ATR
                    if profit_move >= Config.TRAIL_ACTIVATION_ATR * atr:
                        new_sl = state["peak_price"] + (Config.ATR_TRAIL_MULTIPLIER * atr)
                        # Ratchet downwards only (never move SL up)
                        if (state["current_sl"] == 0 or new_sl < state["current_sl"]) and (current_sl == 0 or new_sl < current_sl - 0.0002):
                            self.log("INFO", f"Ratcheting Trailing SL on {inst} Short #{tid}: {state['current_sl']:.4f} -> {new_sl:.4f}")
                            res = self.client.update_trade_stop_loss(tid, new_sl, instrument=inst)
                            if res.get("success"):
                                state["current_sl"] = new_sl

        except Exception as e:
            self.log("WARNING", f"Error updating trailing stops: {e}")

    def _scan_for_signals(self):
        """Scan each portfolio instrument for valid trend crossover setups."""
        try:
            open_trades = self.client.get_open_trades()
            open_instruments = {t.get("instrument") for t in open_trades}

            can_trade, risk_reason = self.risk_manager.can_open_trade(len(open_trades))
            if not can_trade:
                return

            pricing = self.client.get_pricing(Config.INSTRUMENTS)

            for inst in Config.INSTRUMENTS:
                # Do not stack positions on the same instrument
                if inst in open_instruments:
                    continue

                tf = Config.TIMEFRAME_MAPPING.get(inst, Config.DEFAULT_TIMEFRAME)
                raw_candles = self.client.get_candles(inst, granularity=tf, count=225)

                analysis = self.strategy.analyze(inst, raw_candles)
                signal = analysis.get("signal")
                trade_plan = analysis.get("trade_plan")

                if signal in ("BUY", "SELL") and trade_plan:
                    spread = pricing.get(inst, {}).get("spread_pips", 0.0)
                    if spread > Config.MAX_SPREAD_PIPS:
                        self.log("WARNING", f"Skipping {inst} {signal}: Spread ({spread} pips) > Max ({Config.MAX_SPREAD_PIPS})")
                        continue

                    entry_price = trade_plan["entry_price"]
                    stop_loss = trade_plan["stop_loss"]

                    units = self.risk_manager.calculate_position_size(inst, entry_price, stop_loss)
                    signed_units = units if signal == "BUY" else -units

                    self.log("INFO", f"TRIGGER: {signal} on {inst} ({tf}) @ {entry_price:.4f} | SL: {stop_loss:.4f} | Units: {units}")
                    res = self.client.place_market_order(
                        instrument=inst,
                        units=signed_units,
                        stop_loss=stop_loss
                    )

                    if res.get("success"):
                        tid = res.get("trade_id")
                        self.log("INFO", f"Order Executed! OANDA Trade ID: #{tid}")
                        # Immediately seed trailing state
                        if tid:
                            self.trailing_states[str(tid)] = {
                                "instrument": inst,
                                "direction": "LONG" if signal == "BUY" else "SHORT",
                                "entry_price": res.get("price", entry_price),
                                "peak_price": res.get("price", entry_price),
                                "atr": trade_plan["atr"],
                                "current_sl": stop_loss
                            }
                    else:
                        self.log("ERROR", f"Order failed on {inst}: {res.get('error')}")

        except Exception as e:
            self.log("ERROR", f"Error scanning for signals: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Aggregate status payload for Web UI dashboard."""
        open_trades = []
        open_unrealized_pl = 0.0
        try:
            open_trades = self.client.get_open_trades()
            open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)
        except Exception:
            pass

        risk_metrics = self.risk_manager.get_dashboard_metrics(open_unrealized_pl)

        with self._lock:
            logs = list(self.log_history)

        return {
            "is_running": self.is_running,
            "metrics": risk_metrics,
            "open_trades": open_trades,
            "instruments": Config.INSTRUMENTS,
            "timeframe_mapping": Config.TIMEFRAME_MAPPING,
            "logs": logs
        }
