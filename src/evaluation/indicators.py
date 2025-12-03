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
