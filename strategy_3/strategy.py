import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config import Config

logger = logging.getLogger("SMC_Strategy")

class LiquiditySweepStrategy:
    """
    ICT / SMC Asian Range Liquidity Sweep & Judas Swing Strategy (M15 Timeframe)
    1. Identifies the Asian Session Range (00:00 to 06:45 UTC).
    2. Identifies institutional liquidity pools resting above Asian High (Buy-side Liquidity)
       and below Asian Low (Sell-side Liquidity).
    3. During London Killzone (07:00 to 11:00 UTC), detects liquidity raids / sweeps.
    4. Triggers entry when price sweeps liquidity and closes back inside range with market structure shift.
    5. Sets dynamic stop beyond sweep wick and target at 1:1.5 to 1:2 R:R.
    """

    def __init__(self, symbol: str = "EUR_USD"):
        self.symbol = symbol
        self.max_spread = Config.MAX_SPREAD_PIPS
        self.last_traded_day = None

    def get_pip_scale(self) -> Tuple[float, int]:
        """Return (pip_scale, decimals) based on traded instrument."""
        sym = self.symbol.upper().replace("_", "")
        if "XAU" in sym or "GOLD" in sym:
            return 0.10, 2
        elif "JPY" in sym:
            return 0.01, 3
        else:
            return 0.0001, 5

    def get_range_bounds(self) -> Tuple[float, float]:
        """Return (min_range_pips, max_range_pips) dynamically based on instrument volatility."""
        sym = self.symbol.upper().replace("_", "")
        if "XAU" in sym or "GOLD" in sym:
            return 30.0, 350.0 # Gold moves $3.00 to $35.00 in Asia
        elif "JPY" in sym:
            return 15.0, 90.0
        else: # Standard EUR/USD, GBP/USD
            return 8.0, 55.0

    def is_market_open(self) -> Tuple[bool, str]:
        """Check if Forex market is open (closes Friday 21:00 UTC, opens Sunday 21:00 UTC)."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour

        if weekday == 4 and hour >= 21:
            return False, "Market closed for the weekend (Friday post-close)."
        if weekday == 5:
            return False, "Market closed (Saturday)."
        if weekday == 6 and hour < 21:
            return False, "Market closed (Sunday pre-open)."
        return True, "Forex market is open."

    def is_killzone_active(self, current_time: datetime) -> Tuple[bool, str]:
        """Check if current time is within London Killzone (07:00 - 11:00 UTC)."""
        hour = current_time.hour
        if Config.LONDON_KILLZONE_START_UTC <= hour < Config.LONDON_KILLZONE_END_UTC:
            return True, f"London Killzone Active ({hour:02d}:{current_time.minute:02d} UTC)."
        return False, f"Outside London Killzone ({hour:02d}:{current_time.minute:02d} UTC). Active window: {Config.LONDON_KILLZONE_START_UTC}:00 - {Config.LONDON_KILLZONE_END_UTC}:00 UTC."

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Standard Average True Range (ATR)."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    def analyze(self, candles: List[Dict[str, Any]], current_spread_pips: float = 1.0) -> Dict[str, Any]:
        """
        Evaluate closed candles for ICT / SMC Liquidity Sweep setup.
        """
        result = {
            "signal": "HOLD",
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_pips": 0.0,
            "reward_pips": 0.0,
            "reason": "Scanning markets.",
            "asian_high": 0.0,
            "asian_low": 0.0,
            "asian_range_pips": 0.0
        }

        # Filter completed candles
        completed_candles = [c for c in candles if c.get("complete", False)]
        if len(completed_candles) < 30:
            result["reason"] = f"Insufficient completed candles ({len(completed_candles)}/30)."
            return result

        # Check market open
        open_ok, open_msg = self.is_market_open()
        if not open_ok:
            result["reason"] = open_msg
            return result

        # Spread check
        if current_spread_pips > self.max_spread:
            result["reason"] = f"Spread too wide ({current_spread_pips:.1f} pips > {self.max_spread:.1f} max)."
            return result

        # Build DataFrame
        records = []
        for c in completed_candles:
            mid = c.get("mid", {})
            t_str = c.get("time", "")
            t_dt = pd.to_datetime(t_str)
            records.append({
                "time": t_dt,
                "open": float(mid.get("o", 0.0)),
                "high": float(mid.get("h", 0.0)),
                "low": float(mid.get("l", 0.0)),
                "close": float(mid.get("c", 0.0)),
                "volume": int(c.get("volume", 0))
            })
        df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)

        curr_bar = df.iloc[-1]
        curr_time = curr_bar["time"]
        curr_day = curr_time.strftime("%Y-%m-%d")

        # Killzone check
        killzone_ok, killzone_msg = self.is_killzone_active(curr_time)
        if not killzone_ok:
            result["reason"] = killzone_msg
            return result

        # Check if already took a trade today on this instrument
        if self.last_traded_day == curr_day:
            result["reason"] = f"Already took session trade for today ({curr_day}). Waiting for next day."
            return result

        # Calculate Asian Session range (00:00 to 06:45 UTC of the current trading day)
        today_candles = df[df["time"].dt.strftime("%Y-%m-%d") == curr_day]
        asian_candles = today_candles[
            (today_candles["time"].dt.hour >= Config.ASIAN_SESSION_START_UTC) &
            (today_candles["time"].dt.hour < Config.ASIAN_SESSION_END_UTC)
        ]

        if len(asian_candles) < 12: # Need at least 3 hours of Asian candles
            result["reason"] = f"Building Asian Session range ({len(asian_candles)}/12 bars)."
            return result

        asian_high = asian_candles["high"].max()
        asian_low = asian_candles["low"].min()
        pip_scale, decimals = self.get_pip_scale()
        range_pips = round((asian_high - asian_low) / pip_scale, 1)

        result["asian_high"] = round(asian_high, decimals)
        result["asian_low"] = round(asian_low, decimals)
        result["asian_range_pips"] = range_pips

        # Volatility filter on Asian range
        min_range, max_range = self.get_range_bounds()
        if range_pips < min_range:
            result["reason"] = f"Asian range too narrow ({range_pips} pips < {min_range} min). Market stagnant."
            return result
        if range_pips > max_range:
            result["reason"] = f"Asian range too wide ({range_pips} pips > {max_range} max). News expansion."
            return result

        # Calculate ATR
        atr_series = self.calculate_atr(df["high"], df["low"], df["close"], 14)
        curr_atr = atr_series.iloc[-1]
        if np.isnan(curr_atr) or curr_atr <= 0:
            result["reason"] = "Calculating ATR."
            return result

        # Check London candles for sweep
        london_candles = today_candles[today_candles["time"].dt.hour >= Config.LONDON_KILLZONE_START_UTC]
        curr_c = curr_bar["close"]
        curr_o = curr_bar["open"]
        curr_h = curr_bar["high"]
        curr_l = curr_bar["low"]

        sweep_buffer = Config.SWEEP_BUFFER_PIPS * pip_scale

        # =====================================================================
        # 1. BEARISH LIQUIDITY SWEEP (JUDAS SWING SHORT)
        # Price spiked above Asian High, grabbing buy-side stops.
        # Current bar closes back BELOW Asian High with a bearish rejection (Close < Open).
        # =====================================================================
        has_pierced_high = any(h > asian_high for h in london_candles["high"])
        if has_pierced_high and (curr_c < asian_high) and (curr_c < curr_o):
            sweep_high = london_candles["high"].max()
            raw_risk = (sweep_high + sweep_buffer) - curr_c
            risk_dist = np.clip(raw_risk, curr_atr * 0.75, curr_atr * 2.5)

            stop_loss = round(curr_c + risk_dist, decimals)
            take_profit = round(curr_c - (Config.RISK_REWARD_RATIO * risk_dist), decimals)

            risk_pips = round(risk_dist / pip_scale, 1)
            reward_pips = round((Config.RISK_REWARD_RATIO * risk_dist) / pip_scale, 1)

            result.update({
                "signal": "SELL",
                "entry_price": round(curr_c, decimals),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pips": risk_pips,
                "reward_pips": reward_pips,
                "reason": f"SMC Bearish Judas Sweep: Swept Asian High ({asian_high:.{decimals}f}) -> Rejection close ({curr_c:.{decimals}f})."
            })
            self.last_traded_day = curr_day
            return result

        # =====================================================================
        # 2. BULLISH LIQUIDITY SWEEP (JUDAS SWING LONG)
        # Price plunged below Asian Low, grabbing sell-side stops.
        # Current bar closes back ABOVE Asian Low with a bullish rejection (Close > Open).
        # =====================================================================
        has_pierced_low = any(l < asian_low for l in london_candles["low"])
        if has_pierced_low and (curr_c > asian_low) and (curr_c > curr_o):
            sweep_low = london_candles["low"].min()
            raw_risk = curr_c - (sweep_low - sweep_buffer)
            risk_dist = np.clip(raw_risk, curr_atr * 0.75, curr_atr * 2.5)

            stop_loss = round(curr_c - risk_dist, decimals)
            take_profit = round(curr_c + (Config.RISK_REWARD_RATIO * risk_dist), decimals)

            risk_pips = round(risk_dist / pip_scale, 1)
            reward_pips = round((Config.RISK_REWARD_RATIO * risk_dist) / pip_scale, 1)

            result.update({
                "signal": "BUY",
                "entry_price": round(curr_c, decimals),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pips": risk_pips,
                "reward_pips": reward_pips,
                "reason": f"SMC Bullish Judas Sweep: Swept Asian Low ({asian_low:.{decimals}f}) -> Rejection close ({curr_c:.{decimals}f})."
            })
            self.last_traded_day = curr_day
            return result

        result["reason"] = f"Monitoring London Killzone (Asian Range: {asian_low:.{decimals}f} - {asian_high:.{decimals}f}, {range_pips} pips)."
        return result
