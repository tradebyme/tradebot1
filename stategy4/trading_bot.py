import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import Config
from oanda_api import OandaClient
from risk_manager import RiskManager
from strategy import AdaptiveSuperSmootherStrategy

logger = logging.getLogger("TradingBot4")


class TradingBot:
    """
    Core Trading Bot Engine coordinating Market Data, Adaptive SuperSmoother Strategy,
    and Institutional Prop Firm Risk Management.
    """

    def __init__(self):
        self.client = OandaClient()
        self.risk_manager = RiskManager()
        self.strategy = AdaptiveSuperSmootherStrategy()

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.known_closed_trade_ids = set()
        self.log_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        
        # High-Water Mark Trailing & Re-entry state tracking
        self.trailing_states: Dict[str, Dict[str, Any]] = {}
        self.last_exit_info: Dict[str, Dict[str, Any]] = {}

        # Seed known closed trades so we don't double count historical trades
        self._seed_closed_trades()
        self.log("INFO", "Adaptive SuperSmoother Trading Bot with Dynamic Peak Trailing initialized successfully.")

    def log(self, level: str, message: str):
        """Append log to in-memory history for Web UI feed and system logger."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        entry = {"time": timestamp, "level": level, "message": message}

        with self._lock:
            self.log_history.append(entry)
            if len(self.log_history) > 200:
                self.log_history = self.log_history[-200:]

        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)

    def _seed_closed_trades(self):
        """
        Record existing closed trades on startup and synchronize challenge performance
        from OANDA broker history so that container restarts never lose challenge progress.
        """
        try:
            closed = self.client.get_closed_trades(count=50)
            challenge_start = getattr(Config, "CHALLENGE_START_DATE", "2026-11-09")
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            needs_sync = (self.risk_manager.state.get("total_trades_count", 0) == 0)

            if needs_sync and closed:
                self.log("INFO", f"Synchronizing challenge trades from OANDA broker history (since {challenge_start})...")
                total_pl = 0.0
                total_trades = 0
                total_wins = 0
                total_losses = 0
                daily_pl = 0.0
                daily_trades = 0
                daily_losses = 0
                running_equity = self.risk_manager.virtual_capital
                equity_history = [
                    {"time": f"{challenge_start}T00:00:00+00:00", "equity": running_equity}
                ]

                sorted_closed = sorted(closed, key=lambda x: x.get("closeTime", ""))

                for trade in sorted_closed:
                    close_time = trade.get("closeTime", "")
                    if close_time >= challenge_start:
                        pl = float(trade.get("realizedPL", 0.0))
                        trade_date = close_time[:10]
                        total_pl += pl
                        total_trades += 1
                        if pl > 0:
                            total_wins += 1
                        elif pl < 0:
                            total_losses += 1

                        running_equity += pl
                        equity_history.append({"time": close_time, "equity": round(running_equity, 2)})

                        if trade_date == today_utc:
                            daily_pl += pl
                            daily_trades += 1
                            if pl < 0:
                                daily_losses += 1

                self.risk_manager.state["challenge_start_date"] = challenge_start
                self.risk_manager.state["total_realized_pl"] = round(total_pl, 4)
                self.risk_manager.state["total_trades_count"] = total_trades
                self.risk_manager.state["total_wins"] = total_wins
                self.risk_manager.state["total_losses"] = total_losses
                self.risk_manager.state["daily_realized_pl"] = round(daily_pl, 4)
                self.risk_manager.state["daily_trades_count"] = daily_trades
                self.risk_manager.state["daily_losses_count"] = daily_losses
                self.risk_manager.state["equity_history"] = equity_history
                self.risk_manager._save_state()
                self.log(
                    "INFO",
                    f"Broker Sync Complete: {total_trades} challenge trades loaded. Total Realized PnL: ${total_pl:+.2f}"
                )

            for t in closed:
                self.known_closed_trade_ids.add(t["id"])
            self.log("INFO", f"Seeded {len(self.known_closed_trade_ids)} past closed trades from OANDA.")
        except Exception as e:
            self.log("WARNING", f"Could not sync closed trades: {e}")

    def sync_closed_trades(self):
        """Detect newly closed trades and update RiskManager PnL."""
        try:
            recent_closed = self.client.get_closed_trades(count=15)
            for trade in recent_closed:
                trade_id = trade["id"]
                if trade_id not in self.known_closed_trade_ids:
                    self.known_closed_trade_ids.add(trade_id)
                    realized_pl = float(trade.get("realizedPL", 0.0))
                    inst = trade.get("instrument", "")
                    self.log("INFO", f"Trade #{trade_id} on {inst} CLOSED with PnL: ${realized_pl:+.2f}")
                    self.risk_manager.record_closed_trade(realized_pl)
        except Exception as e:
            self.log("ERROR", f"Error syncing closed trades: {e}")

    def check_opposite_signal_exits(self, open_trades: List[Dict[str, Any]], pricing: Dict[str, Any]):
        """
        Optional early exit: If an open position is held and the Adaptive SuperSmoother
        generates an opposing regime flip, close the existing trade early.
        """
        if not Config.CLOSE_ON_OPPOSITE_SIGNAL or not open_trades:
            return

        for trade in open_trades:
            inst = trade.get("instrument")
            tid = trade.get("id")
            units = int(trade.get("currentUnits", 0))
            is_long = units > 0

            candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=60)
            if not candles:
                continue

            closed_candles = [c for c in candles if c.get("complete", False)]
            if len(closed_candles) < 30:
                continue

            df = self.strategy.parse_candles(closed_candles)
            df = self.strategy.calculate_indicators(df)
            last_sq = int(df["sq"].iloc[-1])

            # If long and regime flipped to bearish (-1)
            if is_long and last_sq == -1:
                self.log("WARNING", f"Opposing Bearish Flip on {inst}! Closing LONG #{tid} early.")
                self.client.close_trade(tid)
            # If short and regime flipped to bullish (1)
            elif (not is_long) and last_sq == 1:
                self.log("WARNING", f"Opposing Bullish Flip on {inst}! Closing SHORT #{tid} early.")
                self.client.close_trade(tid)

    def manage_peak_profit_trailing(self, open_trades: List[Dict[str, Any]], pricing: Dict[str, Any]):
        """
        Active Trade Manager:
        1. Tracks Peak Unrealized Profit (MFE).
        2. Breakeven Lock: Moves Stop Loss to Entry + 0.1R buffer once profit hits >= +1.0R.
        3. Peak Profit Trailing: If profit reaches >= +1.2R, trails stop loss 0.6 ATR behind peak price.
        4. If price retraces and hits trailing floor, closes trade immediately to bank institutional profits.
        """
        if not getattr(Config, "PEAK_TRAIL_ENABLED", True) or not open_trades:
            return

        for trade in open_trades:
            tid = trade.get("id")
            inst = trade.get("instrument")
            entry_p = float(trade.get("price", 0.0))
            units = int(trade.get("currentUnits", 0))
            if units == 0 or entry_p <= 0:
                continue

            is_long = units > 0
            inst_pricing = pricing.get(inst, {})
            current_p = float(inst_pricing.get("bid" if is_long else "ask", entry_p))
            pip_unit = 0.01 if "JPY" in inst else (0.10 if "XAU" in inst else 0.0001)

            # Get ATR for instrument
            candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=30)
            if not candles:
                continue
            closed_candles = [c for c in candles if c.get("complete", False)]
            if len(closed_candles) < 15:
                continue
            df = self.strategy.parse_candles(closed_candles)
            atr_s = self.strategy.calculate_atr(df, length=14)
            curr_atr = float(atr_s.iloc[-1]) if not atr_s.empty and pd.notna(atr_s.iloc[-1]) else (20.0 * pip_unit)

            # Risk distance R
            current_sl_order = trade.get("stopLossOrder")
            current_sl = float(current_sl_order.get("price")) if current_sl_order else None
            r_dist = abs(entry_p - current_sl) if current_sl else (Config.ATR_SL_MULTIPLIER * curr_atr)

            if r_dist <= 0.00005:
                r_dist = Config.ATR_SL_MULTIPLIER * curr_atr

            if tid not in self.trailing_states:
                self.trailing_states[tid] = {
                    "peak_price": current_p,
                    "peak_r": 0.0,
                    "breakeven_done": False,
                    "trailing_active": False,
                    "trailing_floor": current_sl
                }

            t_state = self.trailing_states[tid]

            if is_long:
                if current_p > t_state["peak_price"]:
                    t_state["peak_price"] = current_p
                gain_r = (current_p - entry_p) / r_dist
                peak_r = (t_state["peak_price"] - entry_p) / r_dist
                t_state["peak_r"] = max(t_state["peak_r"], peak_r)

                # 1. Breakeven Lock at +1.0R
                if Config.BREAKEVEN_ACTIVATION_R > 0 and (not t_state["breakeven_done"]) and (gain_r >= Config.BREAKEVEN_ACTIVATION_R):
                    be_price = round(entry_p + (Config.BREAKEVEN_BUFFER_R * r_dist), 5)
                    if current_sl is None or be_price > current_sl:
                        self.log("INFO", f"🎯 BREAKEVEN LOCK: Trade #{tid} ({inst}) reached +{gain_r:.2f}R! Ratcheting SL to {be_price:.5f} (Risk-Free).")
                        res = self.client.update_trade_stop_loss(tid, be_price)
                        if res.get("success"):
                            t_state["breakeven_done"] = True
                            t_state["trailing_floor"] = be_price

                # 2. Peak Trailing Stop at +1.2R
                if peak_r >= Config.PEAK_TRAIL_ACTIVATION_R:
                    t_state["trailing_active"] = True
                    trail_floor = round(t_state["peak_price"] - (Config.PEAK_TRAIL_ATR_MULT * curr_atr), 5)
                    be_floor = round(entry_p + (Config.BREAKEVEN_BUFFER_R * r_dist), 5)
                    effective_floor = max(trail_floor, be_floor)

                    if t_state["trailing_floor"] is None or effective_floor > t_state["trailing_floor"]:
                        t_state["trailing_floor"] = effective_floor
                        self.log("INFO", f"📈 PEAK TRAIL RATCHET: Trade #{tid} ({inst}) Peak={t_state['peak_price']:.5f} (+{peak_r:.2f}R). Ratcheting SL to {effective_floor:.5f}.")
                        self.client.update_trade_stop_loss(tid, effective_floor)

                    if current_p <= effective_floor:
                        self.log("INFO", f"💰 PEAK PROFIT SECURED: Trade #{tid} ({inst}) retraced to {current_p:.5f} <= floor {effective_floor:.5f}. Closing to bank profits!")
                        self.client.close_trade(tid)
                        self.last_exit_info[inst] = {
                            "direction": "BUY",
                            "exit_time": datetime.now(timezone.utc),
                            "reentry_count": self.last_exit_info.get(inst, {}).get("reentry_count", 0) + 1
                        }
                        if tid in self.trailing_states:
                            del self.trailing_states[tid]

            else:  # SHORT (SELL)
                if current_p < t_state["peak_price"]:
                    t_state["peak_price"] = current_p
                gain_r = (entry_p - current_p) / r_dist
                peak_r = (entry_p - t_state["peak_price"]) / r_dist
                t_state["peak_r"] = max(t_state["peak_r"], peak_r)

                # 1. Breakeven Lock at +1.0R
                if Config.BREAKEVEN_ACTIVATION_R > 0 and (not t_state["breakeven_done"]) and (gain_r >= Config.BREAKEVEN_ACTIVATION_R):
                    be_price = round(entry_p - (Config.BREAKEVEN_BUFFER_R * r_dist), 5)
                    if current_sl is None or be_price < current_sl:
                        self.log("INFO", f"🎯 BREAKEVEN LOCK: Trade #{tid} ({inst}) reached +{gain_r:.2f}R! Ratcheting SL to {be_price:.5f} (Risk-Free).")
                        res = self.client.update_trade_stop_loss(tid, be_price)
                        if res.get("success"):
                            t_state["breakeven_done"] = True
                            t_state["trailing_floor"] = be_price

                # 2. Peak Trailing Stop at +1.2R
                if peak_r >= Config.PEAK_TRAIL_ACTIVATION_R:
                    t_state["trailing_active"] = True
                    trail_floor = round(t_state["peak_price"] + (Config.PEAK_TRAIL_ATR_MULT * curr_atr), 5)
                    be_floor = round(entry_p - (Config.BREAKEVEN_BUFFER_R * r_dist), 5)
                    effective_floor = min(trail_floor, be_floor)

                    if t_state["trailing_floor"] is None or effective_floor < t_state["trailing_floor"]:
                        t_state["trailing_floor"] = effective_floor
                        self.log("INFO", f"📈 PEAK TRAIL RATCHET: Trade #{tid} ({inst}) Peak={t_state['peak_price']:.5f} (+{peak_r:.2f}R). Ratcheting SL to {effective_floor:.5f}.")
                        self.client.update_trade_stop_loss(tid, effective_floor)

                    if current_p >= effective_floor:
                        self.log("INFO", f"💰 PEAK PROFIT SECURED: Trade #{tid} ({inst}) retraced to {current_p:.5f} >= floor {effective_floor:.5f}. Closing to bank profits!")
                        self.client.close_trade(tid)
                        self.last_exit_info[inst] = {
                            "direction": "SELL",
                            "exit_time": datetime.now(timezone.utc),
                            "reentry_count": self.last_exit_info.get(inst, {}).get("reentry_count", 0) + 1
                        }
                        if tid in self.trailing_states:
                            del self.trailing_states[tid]

    def scan_once(self) -> Dict[str, Any]:
        """Perform a single market scan, risk check, and order execution pass."""
        try:
            # 1. Fetch account info & open trades
            account = self.client.get_account_summary()
            open_trades = self.client.get_open_trades()

            open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)

            # 2. Sync newly closed trades
            self.sync_closed_trades()

            # 3. Update Risk status
            risk_summary = self.risk_manager.update_status(open_unrealized_pl)

            pricing = self.client.get_pricing(Config.INSTRUMENTS)

            # 4. Check early exit for opposite regime signal
            self.check_opposite_signal_exits(open_trades, pricing)

            # 4b. Manage Peak Profit Trailing & Breakeven Lock on Open Trades
            self.manage_peak_profit_trailing(open_trades, pricing)

            # Re-fetch open trades count after potential early exit
            open_trades = self.client.get_open_trades()
            open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)

            # 5. Check if new trades are permitted
            can_trade, reason = self.risk_manager.can_open_trade(len(open_trades), open_unrealized_pl)

            analysis_results = []

            if can_trade:
                for inst in Config.INSTRUMENTS:
                    inst_pricing = pricing.get(inst, {})
                    candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=120)
                    analysis = self.strategy.analyze(inst, candles, inst_pricing)
                    
                    # Check Smart Continuation Re-Entry if no primary signal
                    if not analysis.get("has_signal") and getattr(Config, "REENTRY_ENABLED", True):
                        last_exit = self.last_exit_info.get(inst)
                        if last_exit:
                            time_diff = (datetime.now(timezone.utc) - last_exit["exit_time"]).total_seconds()
                            max_diff = Config.REENTRY_MAX_BARS * 15 * 60  # e.g. 6 bars * 15 min = 90 min
                            if time_diff <= max_diff and last_exit.get("reentry_count", 0) <= Config.REENTRY_MAX_ATTEMPTS:
                                last_dir = last_exit.get("direction")
                                sq = analysis.get("sq", 0)
                                trend = analysis.get("trend", 0.0)
                                atr = analysis.get("atr", 0.0)
                                bid = inst_pricing.get("bid", 0.0)
                                ask = inst_pricing.get("ask", 0.0)

                                if last_dir == "BUY" and sq == 1 and ask > trend and atr > 0:
                                    entry = float(ask)
                                    sl = round(entry - (Config.ATR_SL_MULTIPLIER * atr), 5)
                                    tp = round(entry + (Config.ATR_TP_MULTIPLIER * atr), 5)
                                    analysis.update({
                                        "has_signal": True,
                                        "direction": "BUY",
                                        "entry_price": entry,
                                        "stop_loss": sl,
                                        "take_profit": tp,
                                        "reason": f"Smart Re-entry: Trend continuation in Bullish Regime SQ=+1 after MFE profit bank."
                                    })
                                elif last_dir == "SELL" and sq == -1 and bid < trend and atr > 0:
                                    entry = float(bid)
                                    sl = round(entry + (Config.ATR_SL_MULTIPLIER * atr), 5)
                                    tp = round(entry - (Config.ATR_TP_MULTIPLIER * atr), 5)
                                    analysis.update({
                                        "has_signal": True,
                                        "direction": "SELL",
                                        "entry_price": entry,
                                        "stop_loss": sl,
                                        "take_profit": tp,
                                        "reason": f"Smart Re-entry: Trend continuation in Bearish Regime SQ=-1 after MFE profit bank."
                                    })

                    analysis_results.append(analysis)

                    if analysis.get("has_signal"):
                        direction = analysis["direction"]
                        entry = analysis["entry_price"]
                        sl = analysis["stop_loss"]
                        tp = analysis["take_profit"]

                        # Calculate lot sizing
                        units = self.risk_manager.calculate_position_size(inst, entry, sl)
                        order_units = units if direction == "BUY" else -units

                        self.log(
                            "INFO",
                            f"SIGNAL TRIGGERED on {inst}: {direction} @ {entry:.5f} | "
                            f"SL: {sl:.5f} | TP: {tp:.5f} | Units: {order_units}"
                        )

                        # Submit Market Order with attached guaranteed SL and TP
                        order_res = self.client.place_market_order(inst, order_units, sl, tp)
                        if order_res.get("success"):
                            trade_id = order_res.get("trade_id")
                            fill_price = order_res.get("price")
                            self.log("INFO", f"ORDER SUCCESS: {inst} #{trade_id} filled @ {fill_price}")
                            break
                        else:
                            self.log("ERROR", f"Order execution failed on {inst}: {order_res.get('error')}")
            else:
                self.log("INFO", f"Scan: New entries halted. {reason}")

            return {
                "account": account,
                "open_trades": open_trades,
                "risk_summary": risk_summary,
                "can_trade": can_trade,
                "reason": reason,
                "analysis_results": analysis_results
            }

        except Exception as e:
            self.log("ERROR", f"Error in scan_once: {e}")
            return {"error": str(e)}

    def run_loop(self):
        """Continuous background trading loop."""
        self.is_running = True
        self.log("INFO", f"Adaptive SuperSmoother Bot loop started (polling every {Config.POLL_INTERVAL_SECONDS}s)...")

        while self.is_running:
            try:
                self.scan_once()
            except Exception as e:
                self.log("ERROR", f"Unhandled exception in bot loop: {e}")

            for _ in range(Config.POLL_INTERVAL_SECONDS):
                if not self.is_running:
                    break
                time.sleep(1)

        self.log("INFO", "Trading Bot loop stopped.")

    def start_background(self):
        """Start the trading bot in a separate background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.info("Bot thread is already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self.run_loop, daemon=True, name="TradingBot4Thread")
        self._thread.start()
        logger.info("Bot background thread launched.")

    def stop_background(self):
        """Stop the background bot thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3)
            logger.info("Bot background thread joined.")

    def emergency_close_all(self) -> List[Dict[str, Any]]:
        """Emergency trigger: Close all open trades on OANDA immediately."""
        self.log("WARNING", "EMERGENCY CLOSE ALL TRIGGERED BY USER!")
        results = self.client.close_all_trades()
        self.sync_closed_trades()
        return results

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate all live data required by the Web Dashboard UI."""
        account = self.client.get_account_summary()
        open_trades = self.client.get_open_trades()
        closed_trades = self.client.get_closed_trades(count=20)

        open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)
        risk_summary = self.risk_manager.get_summary(open_unrealized_pl)

        try:
            live_prices = self.client.get_pricing(Config.INSTRUMENTS)
        except Exception:
            live_prices = {}

        # Enrich open trades with live pricing, pips, and SL/TP details
        enriched_open_trades = []
        if open_trades:
            for t in open_trades:
                inst = t.get("instrument", "")
                open_p = float(t.get("price", 0.0))
                units = int(t.get("currentUnits", 0))
                pip_size = 0.01 if "JPY" in inst else (0.10 if "XAU" in inst else 0.0001)

                inst_price = live_prices.get(inst, {})
                curr_p = inst_price.get("bid" if units > 0 else "ask", open_p)
                pips = round(
                    ((curr_p - open_p) / pip_size) if units > 0 else ((open_p - curr_p) / pip_size),
                    1
                )

                sl_price = t.get("stopLossOrder", {}).get("price") if t.get("stopLossOrder") else None
                tp_price = t.get("takeProfitOrder", {}).get("price") if t.get("takeProfitOrder") else None

                t_state = self.trailing_states.get(str(t.get("id")), {})
                enriched_open_trades.append({
                    "id": t.get("id"),
                    "instrument": inst,
                    "units": units,
                    "direction": "BUY" if units > 0 else "SELL",
                    "open_price": open_p,
                    "current_price": curr_p,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
                    "pips": pips,
                    "open_time": t.get("openTime", ""),
                    "peak_price": t_state.get("peak_price", curr_p),
                    "peak_r": round(t_state.get("peak_r", 0.0), 2),
                    "trailing_floor": t_state.get("trailing_floor", sl_price),
                    "is_trailing": t_state.get("trailing_active", False),
                    "is_breakeven": t_state.get("breakeven_done", False),
                })

        # Enrich closed trades
        enriched_closed_trades = []
        for ct in closed_trades:
            pl = float(ct.get("realizedPL", 0.0))
            units = int(ct.get("initialUnits", 0))
            enriched_closed_trades.append({
                "id": ct.get("id"),
                "instrument": ct.get("instrument", ""),
                "direction": "BUY" if units > 0 else "SELL",
                "units": abs(units),
                "price": float(ct.get("price", 0.0)),
                "average_close_price": float(ct.get("averageClosePrice", 0.0)),
                "realized_pl": pl,
                "close_time": ct.get("closeTime", "")
            })

        # Fetch latest strategy indicators for display
        indicator_states = {}
        for inst in Config.INSTRUMENTS:
            try:
                candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=50)
                closed_candles = [c for c in candles if c.get("complete", False)]
                if closed_candles:
                    df = self.strategy.parse_candles(closed_candles)
                    df = self.strategy.calculate_indicators(df)
                    last = df.iloc[-1]
                    indicator_states[inst] = {
                        "trend": round(float(last.get("trend", 0.0)), 5),
                        "slope": round(float(last.get("slope", 0.0)), 6),
                        "threshold": round(float(last.get("threshold", 0.0)), 6),
                        "sq": int(last.get("sq", 0)),
                        "efficiency": round(float(last.get("efficiency", 0.0)), 2),
                        "regime": "BULLISH" if int(last.get("sq", 0)) == 1 else ("BEARISH" if int(last.get("sq", 0)) == -1 else "NEUTRAL")
                    }
            except Exception:
                pass

        session_active, session_reason = self.strategy.is_session_active()
        market_open, market_reason = self.strategy.is_market_open()

        with self._lock:
            logs = list(self.log_history)

        return {
            "account": {
                "id": self.client.account_id,
                "nav": float(account.get("NAV", 0.0)),
                "balance": float(account.get("balance", 0.0)),
                "unrealized_pl": open_unrealized_pl,
                "margin_used": float(account.get("marginUsed", 0.0)),
                "margin_available": float(account.get("marginAvailable", 0.0)),
                "currency": account.get("currency", "USD")
            },
            "risk": risk_summary,
            "session": {
                "active": session_active,
                "market_open": market_open,
                "message": session_reason if market_open else market_reason
            },
            "pricing": live_prices,
            "indicators": indicator_states,
            "open_trades": enriched_open_trades,
            "closed_trades": enriched_closed_trades,
            "equity_history": self.risk_manager.state.get("equity_history", []),
            "logs": logs,
            "config": {
                "instruments": Config.INSTRUMENTS,
                "timeframe": Config.TIMEFRAME,
                "rr_ratio": Config.RISK_REWARD_RATIO,
                "risk_per_trade_usd": self.risk_manager.risk_per_trade_usd,
                "target_profit_usd": self.risk_manager.target_profit_usd,
                "max_drawdown_usd": self.risk_manager.max_drawdown_usd,
                "daily_circuit_breaker_usd": self.risk_manager.circuit_breaker_usd
            }
        }

