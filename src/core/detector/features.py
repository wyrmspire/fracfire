from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from .models import SetupOutcome
from .models import SetupEntry
from .library import evaluate_generic_entry_1m


@dataclass
class FeatureConfig:
    tick_size: float = 0.25


@dataclass
class BadTradeConfig:
    """Knobs to synthesize labeled 'bad trades' for ML practice."""

    enable_hesitation: bool = False
    hesitation_minutes: int = 5
    enable_chase: bool = False
    chase_window_minutes: int = 15
    max_variants_per_trade: int = 2


def nearest_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    if ts in df.index:
        return df.loc[ts]
    # fallback: nearest by time
    idx = df.index.get_indexer([ts], method="nearest")[0]
    return df.iloc[idx]


def compute_trade_features(
    outcome: SetupOutcome,
    df_ind: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> Dict[str, Any]:
    """Compute a compact set of trade features suitable for ML.

    Inputs:
      - outcome: SetupOutcome (contains entry, exit, R, mfe/mae)
      - df_5m_ind: 5m DataFrame with indicators (ema_fast, ema_slow, vwap)
    Returns:
      - dict of numeric/categorical features describing context and execution
    """
    if cfg is None:
        cfg = FeatureConfig()

    e = outcome.entry
    entry_ts = e.time
    exit_ts = outcome.exit_time

    entry_row = nearest_row(df_ind, entry_ts)
    exit_row = nearest_row(df_ind, exit_ts)

    # Price distances
    dist_ema_fast = float(entry_row["close"] - entry_row.get("ema_fast", entry_row["close"]))
    dist_ema_slow = float(entry_row["close"] - entry_row.get("ema_slow", entry_row["close"]))
    dist_vwap = float(entry_row["close"] - entry_row.get("vwap", entry_row["close"]))

    # Candle geometry at entry
    rng = float(entry_row["high"] - entry_row["low"]) if "high" in entry_row else 0.0
    body = abs(float(entry_row["close"] - entry_row["open"])) if "open" in entry_row else 0.0
    wick_up = float(entry_row["high"] - max(entry_row["open"], entry_row["close"])) if "open" in entry_row else 0.0
    wick_dn = float(min(entry_row["open"], entry_row["close"]) - entry_row["low"]) if "open" in entry_row else 0.0
    body_frac = body / rng if rng > 0 else 0.0
    wick_up_frac = wick_up / rng if rng > 0 else 0.0
    wick_dn_frac = wick_dn / rng if rng > 0 else 0.0

    # Timing features
    bars_to_exit = int(df_ind.index.get_indexer([exit_ts], method="nearest")[0]) - int(
        df_ind.index.get_indexer([entry_ts], method="nearest")[0]
    )
    hour = int(entry_ts.hour)
    minute = int(entry_ts.minute)

    # Risk & outcome
    risk = abs(e.entry_price - e.stop_price)
    mfe = float(outcome.mfe)
    mae = float(outcome.mae)
    mfe_mae_ratio = (mfe / max(1e-9, abs(mae))) if mae != 0 else 0.0

    # Simple flags for behavior labeling (placeholders for ML):
    low_rr_flag = 1 if outcome.r_multiple < 0.5 else 0
    chase_flag = 1 if (abs(dist_vwap) < 0.5 and body_frac > 0.7) else 0
    hesitation_flag = 1 if (wick_dn_frac > 0.6 and e.direction == "long") or (wick_up_frac > 0.6 and e.direction == "short") else 0

    features: Dict[str, Any] = {
        "kind": e.kind,
        "direction": e.direction,
        "entry_time": str(entry_ts),
        "exit_time": str(exit_ts),
        "hour": hour,
        "minute": minute,
        "entry_price": e.entry_price,
        "stop_price": e.stop_price,
        "target_price": e.target_price,
        "risk": risk,
        "r_multiple": outcome.r_multiple,
        "mfe": mfe,
        "mae": mae,
        "mfe_mae_ratio": mfe_mae_ratio,
        "dist_ema_fast": dist_ema_fast,
        "dist_ema_slow": dist_ema_slow,
        "dist_vwap": dist_vwap,
        "body_frac": body_frac,
        "wick_up_frac": wick_up_frac,
        "wick_dn_frac": wick_dn_frac,
        "bars_to_exit": bars_to_exit,
        "low_rr_flag": low_rr_flag,
        "chase_flag": chase_flag,
        "hesitation_flag": hesitation_flag,
    }

    # Include context if present (e.g., OR bounds, level names)
    for k, v in (e.context or {}).items():
        features[f"ctx_{k}"] = v

    return features


def inject_bad_trade_variants(
    df_1m_ind: pd.DataFrame,
    base_outcomes: list[SetupOutcome],
    cfg: BadTradeConfig,
) -> list[SetupOutcome]:
    """Create synthetic 'bad trade' variants (hesitation/chase) from outcomes.

    Returns a list of additional SetupOutcome objects labeled via entry.context["variant"].
    """
    variants: list[SetupOutcome] = []
    for o in base_outcomes:
        e = o.entry
        produced = 0

        # Hesitation: enter later at a decision point (or next minute), keep stop, recompute target by R
        if cfg.enable_hesitation and produced < cfg.max_variants_per_trade:
            later_idx = df_1m_ind.index.get_indexer([e.time + pd.Timedelta(minutes=cfg.hesitation_minutes)], method="nearest")[0]
            if later_idx < len(df_1m_ind):
                later_ts = df_1m_ind.index[later_idx]
                later_close = float(df_1m_ind.iloc[later_idx]["close"])
                new_risk = abs(later_close - e.stop_price)
                if new_risk > 0:
                    target_price = later_close + (o.r_multiple if e.direction == "long" else -o.r_multiple) * new_risk
                    entry2 = SetupEntry(
                        time=later_ts,
                        direction=e.direction,
                        kind=e.kind,
                        entry_price=later_close,
                        stop_price=e.stop_price,
                        target_price=target_price,
                        context={**(e.context or {}), "variant": "hesitation"},
                    )
                    out2 = evaluate_generic_entry_1m(df_1m_ind, entry2, max_minutes=240)
                    variants.append(out2)
                    produced += 1

        # Chase: pick a later bar within window with max |close - vwap|
        if cfg.enable_chase and produced < cfg.max_variants_per_trade:
            window_end = e.time + pd.Timedelta(minutes=cfg.chase_window_minutes)
            window = df_1m_ind[(df_1m_ind.index > e.time) & (df_1m_ind.index <= window_end)]
            if not window.empty and "vwap" in window.columns:
                diffs = (window["close"] - window["vwap"]).abs()
                idxmax = int(diffs.idxmax() == diffs.index).bit_length()  # safe guard but not used
                ts_max = diffs.idxmax()
                close_max = float(window.loc[ts_max, "close"])
                new_risk = abs(close_max - e.stop_price)
                if new_risk > 0:
                    target_price = close_max + (o.r_multiple if e.direction == "long" else -o.r_multiple) * new_risk
                    entry2 = SetupEntry(
                        time=ts_max,
                        direction=e.direction,
                        kind=e.kind,
                        entry_price=close_max,
                        stop_price=e.stop_price,
                        target_price=target_price,
                        context={**(e.context or {}), "variant": "chase"},
                    )
                    out2 = evaluate_generic_entry_1m(df_1m_ind, entry2, max_minutes=240)
                    variants.append(out2)
                    produced += 1

    return variants
