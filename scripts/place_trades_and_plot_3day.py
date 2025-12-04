"""
Place trades using multiple setups on 3-day windows and plot regular 5m charts
with entries, stop loss (SL), and take profit (TP). Confirmation on 5m only.

- Generates synthetic data for many 3-day periods
- Detects various setups (ORB, EMA200 continuation, breakout, reversal, opening push, MOC)
- Derives SL/TP using levels/indicators from setup context
- Evaluates outcomes using 1m data but confirms entries on 5m
- Accumulates P&L starting from a given capital, with risk per trade
- Saves per-run charts and a CSV summary
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator
from src.core.detector.indicators import IndicatorConfig, add_5m_indicators
from src.core.detector.library import (
    find_opening_orb_continuations,
    evaluate_orb_entry,
    ORBConfig,
    find_ema200_continuation,
    EMA200ContinuationConfig,
    find_breakout,
    BreakoutConfig,
    find_reversal,
    ReversalConfig,
    find_opening_push,
    OpeningPushConfig,
    find_moc,
    MOCConfig,
)
from src.core.detector.features import evaluate_generic_entry_1m
from src.core.detector.models import SetupEntry, SetupOutcome


def generate_synth_1m(days: int, seed: int) -> pd.DataFrame:
    cfg = PhysicsConfig()
    gen = PriceGenerator(physics_config=cfg, seed=seed)
    all_days = []
    current = pd.Timestamp.utcnow().normalize()
    for i in range(days):
        day_date = current - pd.Timedelta(days=(days - 1 - i))
        df_day = gen.generate_day(day_date)
        all_days.append(df_day)
    df = pd.concat(all_days).sort_values("time")
    df = df.set_index("time")
    return df


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_1m.resample("5min").agg(ohlc).dropna()


def pick_entries(df_5m_ind: pd.DataFrame, df_1m: pd.DataFrame) -> List[SetupEntry]:
    # Try multiple setups; return a small selection per window
    entries: List[SetupEntry] = []

    orb_entries = find_opening_orb_continuations(df_5m_ind, ORBConfig())
    entries.extend(orb_entries[:2])

    ema_entries = find_ema200_continuation(df_5m_ind, EMA200ContinuationConfig())
    entries.extend(ema_entries[:2])

    brk_entries = find_breakout(df_5m_ind, BreakoutConfig())
    entries.extend(brk_entries[:2])

    rev_entries = find_reversal(df_5m_ind, ReversalConfig())
    entries.extend(rev_entries[:2])

    op_entries = find_opening_push(df_5m_ind, OpeningPushConfig())
    entries.extend(op_entries[:1])

    moc_entries = find_moc(df_5m_ind, MOCConfig())
    entries.extend(moc_entries[:1])

    # De-duplicate by time/kind
    seen = set()
    unique = []
    for e in entries:
        key = (pd.Timestamp(e.time), e.kind, e.direction)
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique[:6]  # cap per window


def derive_sl_tp(entry: SetupEntry, df_5m_ind: pd.DataFrame) -> SetupEntry:
    # Use indicator/level hints to set stop/target if missing
    price = float(entry.entry_price)
    kind = entry.kind
    direction = entry.direction

    # Default: ATR-based bands
    atr = float(df_5m_ind["atr"].loc[pd.Timestamp(entry.time)]) if "atr" in df_5m_ind.columns else 5.0
    stop_off = max(atr * 0.8, 4.0)
    tp_off = max(atr * 1.6, 8.0)

    if kind == "orb":
        # Support both dict context or direct attributes
        ctx = getattr(entry, "context", {}) or {}
        or_high = ctx.get("or_high", getattr(entry, "or_high", price))
        or_low = ctx.get("or_low", getattr(entry, "or_low", price))
        if direction == "long":
            entry.stop_price = min(price - stop_off, or_low)
            entry.target_price = price + tp_off
        else:
            entry.stop_price = max(price + stop_off, or_high)
            entry.target_price = price - tp_off
    elif kind == "ema200_cont":
        ema200 = (getattr(entry, "context", {}) or {}).get("ema200", getattr(entry, "ema200", price))
        if direction == "long":
            entry.stop_price = min(price - stop_off, ema200 - atr)
            entry.target_price = price + tp_off
        else:
            entry.stop_price = max(price + stop_off, ema200 + atr)
            entry.target_price = price - tp_off
    elif kind == "breakout":
        level = (getattr(entry, "context", {}) or {}).get("breakout_level", getattr(entry, "breakout_level", price))
        buff = atr * 0.5
        if direction == "long":
            entry.stop_price = level - buff
            entry.target_price = price + tp_off
        else:
            entry.stop_price = level + buff
            entry.target_price = price - tp_off
    elif kind == "reversal":
        rsi = (getattr(entry, "context", {}) or {}).get("rsi", getattr(entry, "rsi", 50))
        if direction == "long":
            entry.stop_price = price - stop_off
            entry.target_price = price + tp_off
        else:
            entry.stop_price = price + stop_off
            entry.target_price = price - tp_off
    else:
        # Generic
        if direction == "long":
            entry.stop_price = price - stop_off
            entry.target_price = price + tp_off
        else:
            entry.stop_price = price + stop_off
            entry.target_price = price - tp_off
    return entry


def evaluate_entries(df_1m: pd.DataFrame, df_5m_ind: pd.DataFrame, entries: List[SetupEntry]) -> List[SetupOutcome]:
    outcomes: List[SetupOutcome] = []
    for e in entries:
        e = derive_sl_tp(e, df_5m_ind)
        if e.kind == "orb":
            out = evaluate_orb_entry(df_5m_ind, e, ORBConfig())
        else:
            out = evaluate_generic_entry_1m(df_1m, e, max_minutes=240)
        if out:
            outcomes.append(out)
    return outcomes


def plot_5m_with_trades(df_5m: pd.DataFrame, outcomes: List[SetupOutcome], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    # Wick and body
    for ts, row in df_5m.iterrows():
        o = float(row["open"]) ; c = float(row["close"]) ; h = float(row["high"]) ; l = float(row["low"]) 
        color = "#26a69a" if c >= o else "#ef5350"
        ax.vlines(ts, l, h, color="#666", linewidth=1)
        ax.vlines(ts, min(o, c), max(o, c), color=color, linewidth=4)
    # Overlay trades with shaded boxes and duration-limited lines
    for out in outcomes:
        e = out.entry
        entry_ts = pd.Timestamp(e.time)
        exit_ts = pd.Timestamp(out.exit_time)
        
        # Entry marker
        ax.scatter([entry_ts], [e.entry_price], color=("#27ae60" if out.r_multiple>0 else "#c0392b"), s=60, zorder=5)
        
        # Draw SL/TP lines only for trade duration (entry to exit)
        ax.hlines(e.stop_price, xmin=entry_ts, xmax=exit_ts, colors="#e74c3c", linestyles="dashed", linewidth=1.5, zorder=3)
        ax.hlines(e.target_price, xmin=entry_ts, xmax=exit_ts, colors="#2ecc71", linestyles="dashed", linewidth=1.5, zorder=3)
        
        # Add shaded boxes: red for stop loss zone, green for profit zone
        # Stop loss zone (between entry and stop)
        stop_zone_bottom = min(e.entry_price, e.stop_price)
        stop_zone_top = max(e.entry_price, e.stop_price)
        ax.fill_between([entry_ts, exit_ts], stop_zone_bottom, stop_zone_top, 
                        color="#ef5350", alpha=0.15, zorder=1, label="_nolegend_")
        
        # Profit zone (between entry and target)
        profit_zone_bottom = min(e.entry_price, e.target_price)
        profit_zone_top = max(e.entry_price, e.target_price)
        ax.fill_between([entry_ts, exit_ts], profit_zone_bottom, profit_zone_top, 
                        color="#26a69a", alpha=0.15, zorder=1, label="_nolegend_")
        
        # Label with setup kind and R-multiple
        ax.text(entry_ts, e.entry_price, f"{e.kind}\nR={out.r_multiple:.2f}", fontsize=8, color="#2c3e50", zorder=6)
    
    ax.set_title(title)
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Place trades on 3-day windows and plot 5m charts")
    parser.add_argument("--risk", type=float, default=100.0, help="Risk per trade in USD; 1R = this amount")
    parser.add_argument("--start-cap", type=float, default=2000.0, help="Starting capital/P&L")
    parser.add_argument("--n-windows", type=int, default=20, help="Number of 3-day windows to test")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for generator")
    parser.add_argument("--out-dir", type=str, default=str(os.path.join(PROJECT_ROOT, "out", "trades_3day")))
    args = parser.parse_args()

    pnl = args.start_cap
    summary_rows: List[Dict[str, Any]] = []

    for i in range(args.n_windows):
        seed = args.seed + i
        df_1m = generate_synth_1m(days=3, seed=seed)
        df_5m = resample_5m(df_1m)
        df_5m_ind = add_5m_indicators(df_5m, IndicatorConfig())

        entries = pick_entries(df_5m_ind, df_1m)
        outcomes = evaluate_entries(df_1m, df_5m_ind, entries)

        # PnL update: 1R equals risk dollars
        for out in outcomes:
            pnl += out.r_multiple * args.risk

        # Save chart
        title = f"3-Day Window #{i+1} | Trades: {len(outcomes)} | PnL={pnl:.0f}"
        out_img = os.path.join(args.out_dir, f"window_{i+1:02d}.png")
        plot_5m_with_trades(df_5m, outcomes, title, out_img)

        # Log summary
        for out in outcomes:
            summary_rows.append({
                "window": i+1,
                "time": str(out.entry.time),
                "exit_time": str(out.exit_time),
                "kind": out.entry.kind,
                "direction": out.entry.direction,
                "entry": out.entry.entry_price,
                "stop": out.entry.stop_price,
                "target": out.entry.target_price,
                "R": out.r_multiple,
                "mfe": out.mfe,
                "mae": out.mae,
                "pnl_after": pnl,
            })

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Saved {args.n_windows} charts to {args.out_dir} and summary to {csv_path}")


if __name__ == "__main__":
    main()
