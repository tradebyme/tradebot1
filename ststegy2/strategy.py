import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger("Strategy2")

class ConfluenceTrendStrategy:
    """
    Strategy 2: Adaptive SuperSmoother (Ehlers Filter + Kaufman Efficiency)
    combined with Trend Target Ribbon (ALMA + Standard Deviation Bands + Swing Structure SL).
    """

    def __init__(self):
        self.session_enabled = Config.SESSION_FILTER_ENABLED
        self.session_start = Config.SESSION_START_HOUR_UTC
        self.session_end = Config.SESSION_END_HOUR_UTC
        self.max_spread = Config.MAX_SPREAD_PIPS

    def is_session_active(self) -> Tuple[bool, str]:
        """Check if current UTC time is inside the high-liquidity London/NY overlap."""
        if not self.session_enabled:
            return True, "Session filter disabled."

        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        # Weekend check: Closed from Friday 21:00 UTC to Sunday 21:00 UTC
        if weekday == 5:
            return False, "Market closed (Saturday)."
        if weekday == 4 and now.hour >= 21:
            return False, "Market closed (Friday weekend close)."
        if weekday == 6 and now.hour < 21:
            return False, "Market closed (Sunday before open)."

        hour = now.hour
        if self.session_start <= hour < self.session_end:
            return True, f"London/NY Session Active ({hour:02d}:00 UTC)."
        else:
            return False, f"Off-hours ({hour:02d}:00 UTC). Active between {self.session_start:02d}:00 and {self.session_end:02d}:00 UTC."

    def is_market_open(self) -> Tuple[bool, str]:
        """Verify Forex trading is open on institutional servers."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        if weekday == 5 or (weekday == 4 and now.hour >= 21) or (weekday == 6 and now.hour < 21):
            return False, "Forex market is closed for the weekend."
        return True, "Forex market is open."

    # =========================================================================
    # MATHEMATICAL INDICATORS
    # =========================================================================

    @staticmethod
    def calculate_alma(series: pd.Series, length: int = 34, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
        """
        Arnaud Legoux Moving Average (ALMA).
        Uses Gaussian distribution to reduce lag while providing extreme smoothness.
        """
        m = offset * (length - 1)
        s = length / sigma
        indices = np.arange(length)
        weights = np.exp(-((indices - m) ** 2) / (2.0 * s * s))
        weights /= weights.sum()

        alma_vals = series.rolling(window=length).apply(lambda window: np.dot(window, weights), raw=True)
        return alma_vals

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Standard Average True Range (ATR)."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    @staticmethod
    def calculate_adaptive_supersmoother(close: pd.Series,
                                         ma_min: int = 14,
                                         ma_max: int = 50,
                                         vol_len: int = 9,
                                         smooth_len: int = 3,
                                         er_len: int = 10,
                                         fast_bias: float = 0.30,
                                         noise_len: int = 20,
                                         noise_mult: float = 0.60) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Adaptive SuperSmoother (SchizoQuant):
        Blends an Adaptive Moving Average with an Ehlers 2-Pole SuperSmoother filter
        weighted by the Kaufman Efficiency Ratio.
        Returns: (trend, slope, threshold)
        """
        n = len(close)
        close_vals = close.values

        # 1. Volatility Position & Adaptive MA Length
        delta_p = close.diff().fillna(0)
        s = delta_p.rolling(window=vol_len).std().fillna(0)
        s_mean = s.rolling(window=vol_len).mean().fillna(0)
        s_min = s_mean * 0.25
        s_max = s_mean * 1.75
        vol_range = s_max - s_min

        vol_pos = np.where(vol_range > 0, np.clip((s - s_min) / np.maximum(vol_range, 1e-9), 0.0, 1.0), 0.0)
        adaptive_len = np.where(s <= s_min, ma_max, np.where(s >= s_max, ma_min, ma_max - (ma_max - ma_min) * vol_pos))
        alphas = 2.0 / (adaptive_len + 1.0)

        # Iterative calculation of Adaptive MA
        adaptive_ma = np.zeros(n)
        adaptive_ma[0] = close_vals[0]
        for i in range(1, n):
            adaptive_ma[i] = adaptive_ma[i - 1] + alphas[i] * (close_vals[i] - adaptive_ma[i - 1])

        # 2. Ehlers 2-Pole SuperSmoother Filter
        a1 = math.exp(-1.414 * math.pi / max(3, smooth_len))
        b1 = 2.0 * a1 * math.cos(1.414 * math.pi / max(3, smooth_len))
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3

        smooth_ma = np.zeros(n)
        smooth_ma[0] = adaptive_ma[0]
        if n > 1:
            smooth_ma[1] = adaptive_ma[1]
        for i in range(2, n):
            smooth_ma[i] = (c1 * (adaptive_ma[i] + adaptive_ma[i - 1]) * 0.5 +
                            c2 * smooth_ma[i - 1] +
                            c3 * smooth_ma[i - 2])

        # 3. Kaufman Efficiency Ratio
        direction = close.diff(er_len).abs().fillna(0)
        noise = close.diff().abs().rolling(window=er_len).sum().fillna(0)
        efficiency = np.where(noise > 0, np.clip(direction / np.maximum(noise, 1e-9), 0.0, 1.0), 0.0)

        # 4. Blended Trend
        fast_weight = fast_bias + (1.0 - fast_bias) * efficiency
        slow_weight = 1.0 - fast_weight
        trend = adaptive_ma * fast_weight + smooth_ma * slow_weight

        # 5. Slope & RMA Noise Threshold
        trend_series = pd.Series(trend, index=close.index)
        slope = trend_series.diff().fillna(0)
        slope_noise = slope.abs().ewm(alpha=1.0 / max(1, noise_len), adjust=False).mean()
        threshold = slope_noise * noise_mult

        return trend_series, slope, threshold

    # =========================================================================
    # SIGNAL GENERATION & ANALYSIS
    # =========================================================================

    def analyze(self, candles: List[Dict[str, Any]], current_spread_pips: float = 1.0) -> Dict[str, Any]:
        """
        Evaluate closed candles for Confluence Setup:
        Layer 1: Adaptive SuperSmoother confirms trend regime and efficiency.
        Layer 2: Trend Target Ribbon (ALMA + StDev) triggers breakout with slope conviction.
        Layer 3: Swing Structure Stop Loss & 1:2 Take Profit.
        """
        result = {
            "signal": "HOLD",
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_pips": 0.0,
            "reward_pips": 0.0,
            "reason": "Scanning markets."
        }

        # Filter strictly for completed candles
        completed_candles = [c for c in candles if c.get("complete", False)]
        if len(completed_candles) < Config.ALMA_LENGTH + 20:
            result["reason"] = f"Insufficient completed candles ({len(completed_candles)}/{Config.ALMA_LENGTH + 20})."
            return result

        # Session & Spread check
        session_ok, session_msg = self.is_session_active()
        if not session_ok:
            result["reason"] = session_msg
            return result

        if current_spread_pips > self.max_spread:
            result["reason"] = f"Spread too wide ({current_spread_pips:.1f} pips > {self.max_spread:.1f} max)."
            return result

        # Build DataFrame
        df = pd.DataFrame([{
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in completed_candles])

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 1. 200 EMA Higher Trend Filter
        ema200 = close.ewm(span=200, adjust=False).mean()

        # 2. Trend Target Ribbon Calculations
        alma = self.calculate_alma(close, Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA)
        dev = close.rolling(window=Config.DEV_LENGTH).std()
        atr = self.calculate_atr(high, low, close, Config.ATR_LENGTH)

        slope_score = np.where(atr > 0, (alma - alma.shift(Config.SLOPE_LENGTH)) / atr, 0.0)
        upper_confirm = alma + dev * Config.DEV_MULT
        lower_confirm = alma - dev * Config.DEV_MULT

        bull_ribbon = (slope_score > Config.SLOPE_MIN) & (close > upper_confirm)
        bear_ribbon = (slope_score < -Config.SLOPE_MIN) & (close < lower_confirm)

        # 3. Adaptive SuperSmoother Calculations
        _, slope, threshold = self.calculate_adaptive_supersmoother(
            close, Config.MA_MIN, Config.MA_MAX, Config.VOL_LENGTH,
            Config.SMOOTH_LENGTH, Config.ER_LENGTH, Config.FAST_BIAS,
            Config.NOISE_LENGTH, Config.NOISE_MULT
        )

        super_smoother_bull = slope > threshold
        super_smoother_bear = slope < -threshold

        # Latest closed bar index
        curr_idx = len(df) - 1
        prev_idx = curr_idx - 1

        curr_close = close.iloc[curr_idx]
        curr_atr = atr.iloc[curr_idx]
        curr_ema200 = ema200.iloc[curr_idx]

        # Check for Trend Flip / Breakout Entry with 200 EMA Alignment
        is_bull_confluence = (bull_ribbon.iloc[curr_idx] and
                              super_smoother_bull.iloc[curr_idx] and
                              not bull_ribbon.iloc[prev_idx] and
                              curr_close > curr_ema200)

        is_bear_confluence = (bear_ribbon.iloc[curr_idx] and
                              super_smoother_bear.iloc[curr_idx] and
                              not bear_ribbon.iloc[prev_idx] and
                              curr_close < curr_ema200)

        # Dynamic pip scale: 0.10 for Gold, 0.01 for JPY pairs, 0.0001 for standard Forex
        sym_upper = self.symbol.upper()
        if "XAU" in sym_upper or "GOLD" in sym_upper:
            pip_unit = 0.10
            decimals = 2
        elif "JPY" in sym_upper:
            pip_unit = 0.01
            decimals = 3
        else:
            pip_unit = 0.0001
            decimals = 5

        # =====================================================================
        # BULLISH ENTRY
        # =====================================================================
        if is_bull_confluence:
            # 12-bar swing low structure stop
            lookback = Config.STOP_LOOKBACK
            recent_low = low.iloc[max(0, curr_idx - lookback):curr_idx].min()
            raw_risk = curr_close - recent_low

            min_risk = curr_atr * Config.MIN_STOP_ATR
            max_risk = curr_atr * Config.MAX_STOP_ATR
            risk_dist = np.clip(raw_risk if not np.isnan(raw_risk) else min_risk, min_risk, max_risk)

            stop_loss = round(curr_close - risk_dist, decimals)
            take_profit = round(curr_close + (Config.RISK_REWARD_RATIO * risk_dist), decimals)

            risk_pips = round(risk_dist / pip_unit, 1)
            reward_pips = round((Config.RISK_REWARD_RATIO * risk_dist) / pip_unit, 1)

            result.update({
                "signal": "BUY",
                "entry_price": round(curr_close, decimals),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pips": risk_pips,
                "reward_pips": reward_pips,
                "reason": f"Bullish Confluence: ALMA break ({curr_close:.{decimals}f} > {upper_confirm.iloc[curr_idx]:.{decimals}f}) + SuperSmoother trending + Above 200 EMA."
            })
            return result

        # =====================================================================
        # BEARISH ENTRY
        # =====================================================================
        elif is_bear_confluence:
            # 12-bar swing high structure stop
            lookback = Config.STOP_LOOKBACK
            recent_high = high.iloc[max(0, curr_idx - lookback):curr_idx].max()
            raw_risk = recent_high - curr_close

            min_risk = curr_atr * Config.MIN_STOP_ATR
            max_risk = curr_atr * Config.MAX_STOP_ATR
            risk_dist = np.clip(raw_risk if not np.isnan(raw_risk) else min_risk, min_risk, max_risk)

            stop_loss = round(curr_close + risk_dist, decimals)
            take_profit = round(curr_close - (Config.RISK_REWARD_RATIO * risk_dist), decimals)

            risk_pips = round(risk_dist / pip_unit, 1)
            reward_pips = round((Config.RISK_REWARD_RATIO * risk_dist) / pip_unit, 1)

            result.update({
                "signal": "SELL",
                "entry_price": round(curr_close, decimals),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pips": risk_pips,
                "reward_pips": reward_pips,
                "reason": f"Bearish Confluence: ALMA break ({curr_close:.{decimals}f} < {lower_confirm.iloc[curr_idx]:.{decimals}f}) + SuperSmoother trending down + Below 200 EMA."
            })
            return result

        result["reason"] = f"Market in consolidation or continuation (ALMA slope: {slope_score[curr_idx]:.3f}, SS slope: {slope.iloc[curr_idx]:.5f})."
        return result

