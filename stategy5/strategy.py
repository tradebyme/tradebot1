import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger("Strategy5")


class TrendEMAATRStrategy:
    """
    Strategy 5: Institutional Trend-ATR Portfolio (ITAP) Engine
    
    Architecture:
    1. 200 EMA Trend Filter:
       - Trade strictly in direction of the macro trend (Bullish if Close > 200 EMA, Bearish if Close < 200 EMA).
    2. 20 EMA & 50 EMA Momentum Crossover Trigger:
       - Enters on the candle close when 20 EMA crosses 50 EMA in direction of the 200 EMA.
    3. Anti-Exhaustion Proximity Filter:
       - Distance |Close - 200 EMA| must be < 2.0x ATR to avoid buying at the tail end of exhausted trends.
    4. Volatility-Adjusted Stop & Trailing Engine:
       - Initial Stop Loss: Exactly 2.0x ATR(14) away from entry price.
       - Dynamic Peak Trailing Stop: 3.0x ATR(14) ratcheted tick-by-tick behind highest peak price.
    """

    def __init__(self):
        self.fast_ema = Config.FAST_EMA
        self.slow_ema = Config.SLOW_EMA
        self.trend_ema = Config.TREND_EMA
        self.atr_period = Config.ATR_PERIOD
        self.sl_atr_mult = Config.ATR_SL_MULTIPLIER
        self.trail_atr_mult = Config.ATR_TRAIL_MULTIPLIER
        self.max_dist_atr_mult = Config.MAX_DIST_ATR_MULTIPLIER
        self.last_signal_time: Dict[str, str] = {}

    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if Forex & Gold markets are currently open.
        Forex markets close Friday at ~21:00 UTC and reopen Sunday at ~21:00 UTC.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
        hour = now.hour

        if weekday == 4 and hour >= 21:
            return False, "Market closed for the weekend (Friday post-close)."
        if weekday == 5:
            return False, "Market closed (Saturday)."
        if weekday == 6 and hour < 21:
            return False, "Market closed (Sunday pre-open)."

        return True, "Market is open."

    def is_session_active(self) -> Tuple[bool, str]:
        """Check if session filter is satisfied (optional for H4 swing)."""
        if not Config.SESSION_FILTER_ENABLED:
            return True, "Session filter disabled (Macro H1/H4 swing mode)."

        now = datetime.now(timezone.utc)
        hour = now.hour

        if Config.SESSION_START_HOUR_UTC <= hour < Config.SESSION_END_HOUR_UTC:
            return True, f"Active session ({Config.SESSION_START_HOUR_UTC}:00 - {Config.SESSION_END_HOUR_UTC}:00 UTC)."

        return False, f"Outside session hours ({hour}:00 UTC)."

    def parse_candles(self, raw_candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Parse raw OANDA candle JSON into a clean Pandas DataFrame."""
        records = []
        for c in raw_candles:
            mid = c.get("mid", {})
            records.append({
                "time": c.get("time"),
                "complete": c.get("complete", False),
                "open": float(mid.get("o", 0.0)),
                "high": float(mid.get("h", 0.0)),
                "low": float(mid.get("l", 0.0)),
                "close": float(mid.get("c", 0.0)),
                "volume": int(c.get("volume", 0))
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 20 EMA, 50 EMA, 200 EMA, and 14 ATR using Wilder's smoothing."""
        data = df.copy()

        # Exponential Moving Averages
        data["ema_fast"] = data["close"].ewm(span=self.fast_ema, adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=self.slow_ema, adjust=False).mean()
        data["ema_trend"] = data["close"].ewm(span=self.trend_ema, adjust=False).mean()

        # Average True Range (Wilder's Smoothing)
        high = data["high"]
        low = data["low"]
        close_prev = data["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        data["atr"] = tr.ewm(alpha=1.0 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        return data

    def analyze(self, instrument: str, raw_candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate completed candles and return signal, trade plan, and technical state.
        Guarantees ZERO repainting by operating strictly on candle index -2 (the most recently completed bar).
        """
        df = self.parse_candles(raw_candles)

        if len(df) < self.trend_ema + 15:
            return {
                "signal": "NEUTRAL",
                "reason": f"Insufficient bars ({len(df)}/{self.trend_ema + 15} required)",
                "data": None
            }

        data = self.compute_indicators(df)

        # Inspect the last fully closed candle (iloc[-2]) and its predecessor (iloc[-3])
        # Note: iloc[-1] is the currently forming, uncompleted candle
        curr = data.iloc[-2]
        prev = data.iloc[-3]

        bar_time = str(curr["time"])
        close = curr["close"]
        ema_fast_curr = curr["ema_fast"]
        ema_slow_curr = curr["ema_slow"]
        ema_trend_curr = curr["ema_trend"]
        atr = curr["atr"]

        ema_fast_prev = prev["ema_fast"]
        ema_slow_prev = prev["ema_slow"]

        # Prevent duplicate signal execution on the same bar
        if self.last_signal_time.get(instrument) == bar_time:
            return {
                "signal": "NEUTRAL",
                "reason": f"Signal already evaluated for bar {bar_time}",
                "data": {
                    "close": close,
                    "ema_fast": ema_fast_curr,
                    "ema_slow": ema_slow_curr,
                    "ema_trend": ema_trend_curr,
                    "atr": atr,
                    "bar_time": bar_time
                }
            }

        # Check market open and session
        is_open, open_msg = self.is_market_open()
        if not is_open:
            return {"signal": "NEUTRAL", "reason": open_msg, "data": None}

        is_active, session_msg = self.is_session_active()
        if not is_active:
            return {"signal": "NEUTRAL", "reason": session_msg, "data": None}

        # Crossover Detection
        bullish_crossover = (ema_fast_prev <= ema_slow_prev) and (ema_fast_curr > ema_slow_curr)
        bearish_crossover = (ema_fast_prev >= ema_slow_prev) and (ema_fast_curr < ema_slow_curr)

        # Proximity Calculations
        dist_from_200 = abs(close - ema_trend_curr)
        max_allowed_dist = self.max_dist_atr_mult * atr
        is_near_trend_ema = dist_from_200 < max_allowed_dist

        signal = "NEUTRAL"
        trade_plan = None
        reason = "No valid setup."

        # LONG SETUP:
        # 1. Close > 200 EMA (Macro Bullish)
        # 2. 20 EMA crosses above 50 EMA
        # 3. Distance from 200 EMA < 2.0 * ATR
        if (close > ema_trend_curr) and bullish_crossover:
            if is_near_trend_ema:
                signal = "BUY"
                sl_distance = self.sl_atr_mult * atr
                sl_price = close - sl_distance
                trail_distance = self.trail_atr_mult * atr

                trade_plan = {
                    "direction": "LONG",
                    "entry_price": close,
                    "stop_loss": sl_price,
                    "sl_distance": sl_distance,
                    "trail_distance": trail_distance,
                    "atr": atr,
                    "bar_time": bar_time
                }
                reason = f"Bullish Crossover above 200 EMA (SL: {sl_price:.4f}, Trail: {trail_distance:.4f})"
                self.last_signal_time[instrument] = bar_time
            else:
                reason = f"Bullish Crossover rejected: Overextended from 200 EMA ({dist_from_200:.4f} > {max_allowed_dist:.4f})"

        # SHORT SETUP:
        # 1. Close < 200 EMA (Macro Bearish)
        # 2. 20 EMA crosses below 50 EMA
        # 3. Distance from 200 EMA < 2.0 * ATR
        elif (close < ema_trend_curr) and bearish_crossover:
            if is_near_trend_ema:
                signal = "SELL"
                sl_distance = self.sl_atr_mult * atr
                sl_price = close + sl_distance
                trail_distance = self.trail_atr_mult * atr

                trade_plan = {
                    "direction": "SHORT",
                    "entry_price": close,
                    "stop_loss": sl_price,
                    "sl_distance": sl_distance,
                    "trail_distance": trail_distance,
                    "atr": atr,
                    "bar_time": bar_time
                }
                reason = f"Bearish Crossover below 200 EMA (SL: {sl_price:.4f}, Trail: {trail_distance:.4f})"
                self.last_signal_time[instrument] = bar_time
            else:
                reason = f"Bearish Crossover rejected: Overextended from 200 EMA ({dist_from_200:.4f} > {max_allowed_dist:.4f})"

        return {
            "signal": signal,
            "reason": reason,
            "trade_plan": trade_plan,
            "data": {
                "close": close,
                "ema_fast": ema_fast_curr,
                "ema_slow": ema_slow_curr,
                "ema_trend": ema_trend_curr,
                "atr": atr,
                "dist_from_200": dist_from_200,
                "max_dist": max_allowed_dist,
                "bar_time": bar_time
            }
        }
