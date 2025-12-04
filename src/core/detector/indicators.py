from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IndicatorConfig:
    """Shared indicator knobs for 5m/1m data.

    This is intentionally minimal; we can extend as needed without
    touching setup logic.
    """

    ema_fast: int = 20
    ema_slow: int = 200
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_sma_period: int = 20


def add_5m_indicators(df_5m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 5m indicators (EMA fast/slow, VWAP, RSI, BB, ATR, MACD, Volume).

    All setup detectors should expect these columns if they need them.
    """
    df = df_5m.copy()
    
    # EMA indicators
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    # VWAP
    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
    
    # RSI
    df["rsi"] = _calculate_rsi(df["close"], cfg.rsi_period)
    
    # Bollinger Bands
    bb_sma = df["close"].rolling(window=cfg.bb_period).mean()
    bb_std = df["close"].rolling(window=cfg.bb_period).std()
    df["bb_upper"] = bb_sma + (cfg.bb_std * bb_std)
    df["bb_middle"] = bb_sma
    df["bb_lower"] = bb_sma - (cfg.bb_std * bb_std)
    
    # ATR (Average True Range)
    df["atr"] = _calculate_atr(df, cfg.atr_period)
    
    # MACD
    macd_line, signal_line, histogram = _calculate_macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    
    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=cfg.volume_sma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)
    
    return df


def add_1m_indicators(df_1m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 1m indicators (EMA fast/slow, VWAP, RSI, BB, ATR, MACD, Volume)."""
    df = df_1m.copy()
    
    # EMA indicators
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    
    # VWAP
    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
    
    # RSI
    df["rsi"] = _calculate_rsi(df["close"], cfg.rsi_period)
    
    # Bollinger Bands
    bb_sma = df["close"].rolling(window=cfg.bb_period).mean()
    bb_std = df["close"].rolling(window=cfg.bb_period).std()
    df["bb_upper"] = bb_sma + (cfg.bb_std * bb_std)
    df["bb_middle"] = bb_sma
    df["bb_lower"] = bb_sma - (cfg.bb_std * bb_std)
    
    # ATR
    df["atr"] = _calculate_atr(df, cfg.atr_period)
    
    # MACD
    macd_line, signal_line, histogram = _calculate_macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    
    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=cfg.volume_sma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)
    
    return df


@dataclass
class DecisionAnchorConfig:
    """Defines decision points on 1m where setups are allowed to trigger.

    By default, we trigger at the close of each 5th minute (minute % 5 == 4).
    """

    modulo: int = 5
    offset: int = 4


def mark_decision_points_1m(df_1m: pd.DataFrame, cfg: DecisionAnchorConfig) -> pd.Series:
    """Return a boolean Series marking decision points on 1m timeline.

    True indicates this 1m bar is the decision boundary (e.g., 5-min close).
    """
    idx = df_1m.index
    # Assume DatetimeIndex with minute granularity
    marks = ((idx.minute % cfg.modulo) == cfg.offset)
    return pd.Series(marks, index=idx)


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range (ATR).
    
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = SMA of True Range over period
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def _calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
