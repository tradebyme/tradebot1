import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger("Strategy4")


class AdaptiveSuperSmootherStrategy:
    """
    Strategy 4: Adaptive SuperSmoother Strategy (Pine Script by SchizoQuant)
    
    Architecture:
    1. Volatility-Adaptive Moving Average:
       - Uses rolling standard deviation of bar-to-bar price changes (volLen=9).
       - Period dynamically adapts between maMin (14) during high volatility and maMax (50) during consolidation.
    2. John Ehlers 2-Pole SuperSmoother Filter:
       - Applied recursively to the adaptive MA to remove high-frequency noise and spectral aliasing without phase lag.
    3. Kaufman Efficiency Ratio (ER):
       - Measures directional movement vs total path noise (erLength=10).
       - Dynamically blends the responsive Adaptive MA (trending) with the Ehlers SuperSmoother (choppy/noise).
    4. Adaptive Noise Band & State Machine:
       - Wilder's RMA of slope magnitude (noiseLength=20) times noiseMult (0.60) establishes an adaptive deadband.
       - State machine (SQ) flips to +1 on bullish breakout and -1 on bearish breakout, preventing churn in flat markets.
    5. Execution & Risk:
       - Strict zero-repainting: Evaluates only completed/closed candles.
       - Dynamic ATR (14) stop loss (1.5x ATR) and take profit (3.0x ATR) for institutional 1:2 R:R.
       - Session & Weekend safety filters.
    """

    def __init__(self):
        self.instruments = Config.INSTRUMENTS
        self.timeframe = Config.TIMEFRAME
        self.rr_ratio = Config.RISK_REWARD_RATIO
        self.last_signal_time: Dict[str, str] = {}

    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if Forex markets are currently open.
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

    @staticmethod
    def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Standard Average True Range (ATR)."""
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    @staticmethod
    def calculate_adaptive_supersmoother(
        close_series: pd.Series,
        ma_min: int = 14,
        ma_max: int = 50,
        vol_len: int = 9,
        smooth_length: int = 3,
        er_length: int = 10,
        fast_bias: float = 0.30,
        noise_length: int = 20,
        noise_mult: float = 0.60
    ) -> Dict[str, pd.Series]:
        """
        Pure Python/Pandas calculation of the Adaptive SuperSmoother indicator.
        Matches pine script math exactly.
        """
        n = len(close_series)
        src = close_series.values

        # 1. Price Change Volatility & Adaptive Length
        delta_p = close_series.diff().fillna(0.0)
        s = delta_p.rolling(window=vol_len, min_periods=1).std(ddof=1).fillna(0.0)
        s_mean = s.rolling(window=vol_len, min_periods=1).mean().fillna(0.0)

        s_min = s_mean * 0.25
        s_max = s_mean * 1.75
        volatility_range = s_max - s_min

        vol_range_vals = volatility_range.values
        s_vals = s.values
        s_min_vals = s_min.values
        s_max_vals = s_max.values

        vol_pos = np.where(
            vol_range_vals > 0.0,
            np.clip((s_vals - s_min_vals) / np.maximum(vol_range_vals, 1e-9), 0.0, 1.0),
            0.0
        )

        adaptive_length = np.where(
            s_vals <= s_min_vals,
            float(ma_max),
            np.where(
                s_vals >= s_max_vals,
                float(ma_min),
                ma_max - (ma_max - ma_min) * vol_pos
            )
        )
        alphas = 2.0 / (adaptive_length + 1.0)

        # Iterative calculation of Adaptive MA
        adaptive_ma = np.zeros(n)
        adaptive_ma[0] = src[0] if n > 0 else 0.0
        for i in range(1, n):
            adaptive_ma[i] = adaptive_ma[i - 1] + alphas[i] * (src[i] - adaptive_ma[i - 1])

        # 2. John Ehlers 2-Pole SuperSmoother Filter
        eff_smooth_len = max(3, smooth_length)
        a1 = math.exp(-1.414 * math.pi / eff_smooth_len)
        b1 = 2.0 * a1 * math.cos(1.414 * math.pi / eff_smooth_len)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3

        smooth_ma = np.zeros(n)
        smooth_ma[0] = adaptive_ma[0] if n > 0 else 0.0
        if n > 1:
            smooth_ma[1] = adaptive_ma[1]
        for i in range(2, n):
            smooth_ma[i] = (
                c1 * (adaptive_ma[i] + adaptive_ma[i - 1]) * 0.5
                + c2 * smooth_ma[i - 1]
                + c3 * smooth_ma[i - 2]
            )

        # 3. Kaufman Efficiency Ratio
        direction = close_series.diff(er_length).abs().fillna(0.0)
        noise = close_series.diff().abs().rolling(window=er_length, min_periods=1).sum().fillna(0.0)

        efficiency = np.where(
            noise.values > 0.0,
            np.clip(direction.values / np.maximum(noise.values, 1e-9), 0.0, 1.0),
            0.0
        )

        # 4. Dynamic Blended Trend Line
        fast_weight = fast_bias + (1.0 - fast_bias) * efficiency
        slow_weight = 1.0 - fast_weight
        trend = adaptive_ma * fast_weight + smooth_ma * slow_weight

        # 5. Slope & RMA Noise Threshold
        trend_series = pd.Series(trend, index=close_series.index)
        slope = trend_series.diff().fillna(0.0)

        # ta.rma(math.abs(slope), noiseLength) in Pine Script is Wilder's RMA
        slope_noise = slope.abs().ewm(alpha=1.0 / max(1, noise_length), adjust=False).mean()
        threshold = slope_noise * noise_mult

        # 6. Trend Regime State Machine (SQ)
        long_cond = slope > threshold
        short_cond = slope < -threshold

        sq = np.zeros(n, dtype=int)
        curr_sq = 0
        for i in range(n):
            if long_cond.iloc[i]:
                curr_sq = 1
            elif short_cond.iloc[i]:
                curr_sq = -1
            sq[i] = curr_sq

        sq_series = pd.Series(sq, index=close_series.index)
        prev_sq = sq_series.shift(1).fillna(0).astype(int)

        long_signal = (sq_series == 1) & (prev_sq != 1)
        short_signal = (sq_series == -1) & (prev_sq != -1)

        return {
            "adaptive_ma": pd.Series(adaptive_ma, index=close_series.index),
            "smooth_ma": pd.Series(smooth_ma, index=close_series.index),
            "efficiency": pd.Series(efficiency, index=close_series.index),
            "trend": trend_series,
            "slope": slope,
            "threshold": threshold,
            "sq": sq_series,
            "long_signal": long_signal,
            "short_signal": short_signal
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and append them to DataFrame."""
        if len(df) < max(Config.MA_MAX, Config.NOISE_LENGTH) + 10:
            return df

        # ATR 14
        df["atr_14"] = self.calculate_atr(df, length=Config.ATR_LENGTH)

        # Adaptive SuperSmoother
        res = self.calculate_adaptive_supersmoother(
            df["close"],
            ma_min=Config.MA_MIN,
            ma_max=Config.MA_MAX,
            vol_len=Config.VOL_LENGTH,
            smooth_length=Config.SMOOTH_LENGTH,
            er_length=Config.ER_LENGTH,
            fast_bias=Config.FAST_BIAS,
            noise_length=Config.NOISE_LENGTH,
            noise_mult=Config.NOISE_MULT
        )

        df["adaptive_ma"] = res["adaptive_ma"]
        df["smooth_ma"] = res["smooth_ma"]
        df["efficiency"] = res["efficiency"]
        df["trend"] = res["trend"]
        df["slope"] = res["slope"]
        df["threshold"] = res["threshold"]
        df["sq"] = res["sq"]
        df["long_signal"] = res["long_signal"]
        df["short_signal"] = res["short_signal"]

        return df

    def analyze(self, instrument: str, raw_candles: List[Dict[str, Any]], live_pricing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze completed candles and generate trading signal with exact Stop Loss & Take Profit.
        """
        result = {
            "instrument": instrument,
            "has_signal": False,
            "direction": None,  # 'BUY' or 'SELL'
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "atr": 0.0,
            "trend": 0.0,
            "slope": 0.0,
            "threshold": 0.0,
            "sq": 0,
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
        if not raw_candles or len(raw_candles) < 70:
            result["reason"] = "Insufficient candle history."
            return result

        df = self.parse_candles(raw_candles)
        # Use completed candles only to ensure zero repaint
        closed_df = df[df["complete"] == True].copy()
        if len(closed_df) < 60:
            result["reason"] = "Insufficient closed candles."
            return result

        closed_df = self.calculate_indicators(closed_df)

        curr = closed_df.iloc[-1]
        candle_time = curr["time"]
        result["candle_time"] = candle_time

        atr = curr["atr_14"]
        trend = curr["trend"]
        slope = curr["slope"]
        threshold = curr["threshold"]
        sq = int(curr["sq"])
        long_sig = bool(curr["long_signal"])
        short_sig = bool(curr["short_signal"])

        result["atr"] = round(float(atr), 5) if pd.notna(atr) else 0.0
        result["trend"] = round(float(trend), 5) if pd.notna(trend) else 0.0
        result["slope"] = round(float(slope), 6) if pd.notna(slope) else 0.0
        result["threshold"] = round(float(threshold), 6) if pd.notna(threshold) else 0.0
        result["sq"] = sq

        if pd.isna(atr) or atr <= 0.00005:
            result["reason"] = "ATR indicator calculation invalid."
            return result

        # Check if we already processed a signal on this specific candle
        if self.last_signal_time.get(instrument) == candle_time:
            result["reason"] = f"Already processed signal for candle {candle_time}."
            return result

        bid = live_pricing.get("bid", 0.0)
        ask = live_pricing.get("ask", 0.0)
        pip_unit = 0.01 if "JPY" in instrument else (0.10 if "XAU" in instrument else 0.0001)

        allow_long = Config.TRADE_DIRECTION != "Short Only"
        allow_short = Config.TRADE_DIRECTION != "Long Only"

        # Stop and Profit distances based on Strategy Pine Script (atrSL=2.0, atrTP=3.0)
        sl_distance = max(Config.ATR_SL_MULTIPLIER * atr, 10.0 * pip_unit)
        tp_distance = max(Config.ATR_TP_MULTIPLIER * atr, 10.0 * pip_unit)

        # LONG SIGNAL TRIGGER
        if long_sig:
            if not allow_long:
                result["reason"] = f"Bullish flip detected but long trading disabled by TRADE_DIRECTION={Config.TRADE_DIRECTION}."
                return result

            entry = float(ask) if ask > 0 else float(curr["close"])
            sl = round(entry - sl_distance, 5 if pip_unit == 0.0001 else 3) if Config.USE_SL else None
            tp = round(entry + tp_distance, 5 if pip_unit == 0.0001 else 3) if Config.USE_TP else None

            sl_label = f"{sl:.5f}" if sl else "None (Riding Trend)"
            tp_label = f"{tp:.5f}" if tp else "None (Exits on Trend Flip)"

            result["has_signal"] = True
            result["direction"] = "BUY"
            result["entry_price"] = entry
            result["stop_loss"] = sl
            result["take_profit"] = tp
            result["reason"] = (
                f"Adaptive SuperSmoother BULLISH FLIP: SQ=1, Slope={slope:+.6f} > Thresh={threshold:.6f}, "
                f"Entry={entry:.5f}, SL={sl_label}, TP={tp_label}"
            )
            self.last_signal_time[instrument] = candle_time
            return result

        # SHORT SIGNAL TRIGGER
        if short_sig:
            if not allow_short:
                result["reason"] = f"Bearish flip detected but short trading disabled by TRADE_DIRECTION={Config.TRADE_DIRECTION}."
                return result

            entry = float(bid) if bid > 0 else float(curr["close"])
            sl = round(entry + sl_distance, 5 if pip_unit == 0.0001 else 3) if Config.USE_SL else None
            tp = round(entry - tp_distance, 5 if pip_unit == 0.0001 else 3) if Config.USE_TP else None

            sl_label = f"{sl:.5f}" if sl else "None (Riding Trend)"
            tp_label = f"{tp:.5f}" if tp else "None (Exits on Trend Flip)"

            result["has_signal"] = True
            result["direction"] = "SELL"
            result["entry_price"] = entry
            result["stop_loss"] = sl
            result["take_profit"] = tp
            result["reason"] = (
                f"Adaptive SuperSmoother BEARISH FLIP: SQ=-1, Slope={slope:+.6f} < -Thresh={-threshold:.6f}, "
                f"Entry={entry:.5f}, SL={sl_label}, TP={tp_label}"
            )
            self.last_signal_time[instrument] = candle_time
            return result

        regime_str = "BULLISH" if sq == 1 else ("BEARISH" if sq == -1 else "NEUTRAL")
        result["reason"] = (
            f"Active Regime: {regime_str} (Slope: {slope:+.6f}, Threshold: ±{threshold:.6f}). Waiting for regime transition."
        )
        return result

