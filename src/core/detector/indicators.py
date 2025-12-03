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


def add_5m_indicators(df_5m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 5m indicators (EMA fast/slow, VWAP).

    All setup detectors should expect these columns if they need them.
    """
    df = df_5m.copy()
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
    return df


def add_1m_indicators(df_1m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 1m indicators (EMA fast/slow, VWAP)."""
    df = df_1m.copy()
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
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
