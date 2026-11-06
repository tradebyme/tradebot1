import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger("Strategy")

class MomentumPullbackStrategy:
    """
    London/NY Session Momentum Pullback Strategy (M15 Timeframe)
    - Dual EMA (20 & 50) + Macro EMA (100) trend filter
    - Dynamic ATR (14) Volatility Stops (1:2 Risk-Reward)
    - RSI (14) pullback confirmation
    - Session and Weekend filters
    """

    def __init__(self):
        self.instruments = Config.INSTRUMENTS
        self.timeframe = Config.TIMEFRAME
        self.rr_ratio = Config.RISK_REWARD_RATIO # 2.0
        self.last_signal_time = {} # Track last trade candle timestamp per instrument to avoid duplicate entries

    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if Forex markets are currently open.
        Forex closes Friday at ~21:00 UTC and reopens Sunday at ~21:00 UTC.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday() # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
        hour = now.hour

        # Friday after 21:00 UTC
        if weekday == 4 and hour >= 21:
            return False, "Market closed for the weekend (Friday post-close)."
        # All of Saturday
        if weekday == 5:
            return False, "Market closed (Saturday)."
        # Sunday before 21:00 UTC
        if weekday == 6 and hour < 21:
            return False, "Market closed (Sunday pre-open)."

        return True, "Market is open."

    def is_session_active(self) -> Tuple[bool, str]:
        """Check if current time is within high-liquidity London/NY session window."""
        if not Config.SESSION_FILTER_ENABLED:
            return True, "Session filter disabled."

        now = datetime.now(timezone.utc)
        hour = now.hour

        if Config.SESSION_START_HOUR_UTC <= hour < Config.SESSION_END_HOUR_UTC:
            return True, f"Active session ({Config.SESSION_START_HOUR_UTC}:00 - {Config.SESSION_END_HOUR_UTC}:00 UTC)."
        
        return False, f"Outside optimal trading hours ({hour}:00 UTC). Active window is {Config.SESSION_START_HOUR_UTC}:00 - {Config.SESSION_END_HOUR_UTC}:00 UTC."

    def parse_candles(self, raw_candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Parse raw OANDA candles into a clean Pandas DataFrame."""
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
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 20 EMA, 50 EMA, 100 EMA, ATR(14), and RSI(14)."""
        if len(df) < 50:
            return df

        # EMAs
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()

        # ATR 14
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = df["tr"].rolling(window=14).mean()

        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        return df

    def analyze(self, instrument: str, raw_candles: List[Dict[str, Any]], live_pricing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze candles and generate trading signal with exact Stop Loss & Take Profit.
        """
        result = {
            "instrument": instrument,
            "has_signal": False,
            "direction": None, # 'BUY' or 'SELL'
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "atr": 0.0,
            "rsi": 0.0,
            "ema_20": 0.0,
            "ema_50": 0.0,
            "reason": "",
            "candle_time": None
        }

        # 1. Market Open Check
        open_ok, open_msg = self.is_market_open()
        if not open_ok:
            result["reason"] = open_msg
            return result

        # 2. Session Hours Check
        session_ok, session_msg = self.is_session_active()
        if not session_ok:
            result["reason"] = session_msg
            return result

        # 3. Spread Check
        spread_pips = live_pricing.get("spread_pips", 999.0)
        if spread_pips > Config.MAX_SPREAD_PIPS:
            result["reason"] = f"Spread too high ({spread_pips:.1f} pips > max {Config.MAX_SPREAD_PIPS} pips)."
            return result

        # 4. Candles Check
        if not raw_candles or len(raw_candles) < 60:
            result["reason"] = "Insufficient candle history."
            return result

        df = self.parse_candles(raw_candles)
        # Use completed candles only to ensure zero repaint
        closed_df = df[df["complete"] == True].copy()
        if len(closed_df) < 50:
            result["reason"] = "Insufficient closed candles."
            return result

        closed_df = self.calculate_indicators(closed_df)
        
        # Latest closed candle
        curr = closed_df.iloc[-1]
        candle_time = curr["time"]
        result["candle_time"] = candle_time

        # Check if we already traded on this specific candle
        if self.last_signal_time.get(instrument) == candle_time:
            result["reason"] = f"Already processed signal for candle {candle_time}."
            return result

        atr = curr["atr_14"]
        rsi = curr["rsi_14"]
        ema20 = curr["ema_20"]
        ema50 = curr["ema_50"]
        ema100 = curr["ema_100"]
        close_p = curr["close"]
        open_p = curr["open"]

        result["atr"] = round(float(atr), 5)
        result["rsi"] = round(float(rsi), 2)
        result["ema_20"] = round(float(ema20), 5)
        result["ema_50"] = round(float(ema50), 5)

        # Volatility check: ATR must be valid and non-zero
        if pd.isna(atr) or atr <= 0.0001:
            result["reason"] = "ATR indicator calculation invalid."
            return result

        bid = live_pricing.get("bid", 0.0)
        ask = live_pricing.get("ask", 0.0)

        # Lookback for pullback: last 3 candles
        recent_lows = closed_df["low"].iloc[-3:].values
        recent_highs = closed_df["high"].iloc[-3:].values

        # --- BULLISH (BUY) SETUP ---
        # 1. Trend: EMA 20 > EMA 50 and Close > EMA 100
        # 2. Pullback: Low touched or penetrated within 0.75 ATR of EMA 20
        # 3. RSI: Healthy zone (40 to 62)
        # 4. Trigger: Current closed candle is green (Close > Open) and above EMA 20
        is_bullish_trend = (ema20 > ema50) and (close_p > ema100)
        pulled_back_support = any(low <= (ema20 + 0.75 * atr) for low in recent_lows)
        rsi_bullish = 40.0 <= rsi <= 62.0
        trigger_bullish = (close_p > open_p) and (close_p > ema20)

        if is_bullish_trend and pulled_back_support and rsi_bullish and trigger_bullish:
            entry = float(ask)
            sl = round(float(entry - (1.5 * atr)), 5)
            tp = round(float(entry + (3.0 * atr)), 5)

            result["has_signal"] = True
            result["direction"] = "BUY"
            result["entry_price"] = entry
            result["stop_loss"] = sl
            result["take_profit"] = tp
            result["reason"] = f"Bullish Pullback Confirmed: EMA20>50, RSI={rsi:.1f}, SL={sl:.5f}, TP={tp:.5f}"
            self.last_signal_time[instrument] = candle_time
            return result

        # --- BEARISH (SELL) SETUP ---
        # 1. Trend: EMA 20 < EMA 50 and Close < EMA 100
        # 2. Pullback: High touched near EMA 20
        # 3. RSI: Healthy zone (38 to 60)
        # 4. Trigger: Current closed candle is red (Close < Open) and below EMA 20
        is_bearish_trend = (ema20 < ema50) and (close_p < ema100)
        pulled_back_resistance = any(high >= (ema20 - 0.75 * atr) for high in recent_highs)
        rsi_bearish = 38.0 <= rsi <= 60.0
        trigger_bearish = (close_p < open_p) and (close_p < ema20)

        if is_bearish_trend and pulled_back_resistance and rsi_bearish and trigger_bearish:
            entry = float(bid)
            sl = round(float(entry + (1.5 * atr)), 5)
            tp = round(float(entry - (3.0 * atr)), 5)

            result["has_signal"] = True
            result["direction"] = "SELL"
            result["entry_price"] = entry
            result["stop_loss"] = sl
            result["take_profit"] = tp
            result["reason"] = f"Bearish Pullback Confirmed: EMA20<50, RSI={rsi:.1f}, SL={sl:.5f}, TP={tp:.5f}"
            self.last_signal_time[instrument] = candle_time
            return result

        result["reason"] = f"No setup on {instrument} (Trend: {'BULL' if ema20 > ema50 else 'BEAR'}, RSI: {rsi:.1f})"
        return result
