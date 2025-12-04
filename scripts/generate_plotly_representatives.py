"""Generate interactive Plotly HTML + PNG for 3 representative windows.

Select windows (best / median / worst) by the maximum R per window from
`summary.csv`, regenerate the 3-day 1m data using the same generator seed
logic used by `place_trades_and_plot_3day.py`, build a Plotly candlestick
figure, overlay entries/SL/TP, save both an HTML and a PNG via Kaleido.

Usage:
  python scripts/generate_plotly_representatives.py \
      --summary out/trades_3day/summary.csv \
      --base-seed 123 \
      --out-dir out/trades_3day/representative
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator
from src.core.detector.indicators import IndicatorConfig, add_5m_indicators


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


def plot_plotly_window(df_5m: pd.DataFrame, trades: pd.DataFrame, title: str, out_html: str, out_png: str):
    """Plot 5m candles with trades showing shaded profit/loss zones and duration-limited SL/TP lines."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_5m.index,
            open=df_5m["open"],
            high=df_5m["high"],
            low=df_5m["low"],
            close=df_5m["close"],
            name="Price",
        )
    )

    # Overlay trades with shaded boxes and duration-limited lines
    for _, r in trades.iterrows():
        entry_time = pd.to_datetime(r["time"])
        # Use exit_time if available, otherwise estimate 60 minutes
        exit_time = pd.to_datetime(r["exit_time"]) if "exit_time" in r and not pd.isna(r["exit_time"]) else entry_time + pd.Timedelta(minutes=60)
        
        entry = float(r["entry"])
        stop = float(r["stop"]) if not pd.isna(r["stop"]) else None
        target = float(r["target"]) if not pd.isna(r["target"]) else None
        r_multiple = float(r["R"])
        color = "green" if r_multiple > 0 else "red"
        
        # Add shaded profit zone (green)
        if target is not None:
            profit_zone_y = [min(entry, target), max(entry, target)]
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time, exit_time, entry_time, entry_time],
                y=[profit_zone_y[0], profit_zone_y[0], profit_zone_y[1], profit_zone_y[1], profit_zone_y[0]],
                fill="toself",
                fillcolor="rgba(38, 166, 154, 0.15)",  # Green with transparency
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="_profit_zone"
            ))
        
        # Add shaded stop loss zone (red)
        if stop is not None:
            stop_zone_y = [min(entry, stop), max(entry, stop)]
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time, exit_time, entry_time, entry_time],
                y=[stop_zone_y[0], stop_zone_y[0], stop_zone_y[1], stop_zone_y[1], stop_zone_y[0]],
                fill="toself",
                fillcolor="rgba(239, 83, 80, 0.15)",  # Red with transparency
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="_stop_zone"
            ))
        
        # Draw stop/target lines only for trade duration
        if stop is not None:
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time], 
                y=[stop, stop], 
                mode="lines",
                line=dict(color="firebrick", dash="dash", width=2),
                showlegend=False,
                hovertext=f"Stop Loss: {stop:.2f}",
                name="_stop_line"
            ))
        if target is not None:
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time], 
                y=[target, target], 
                mode="lines",
                line=dict(color="green", dash="dash", width=2),
                showlegend=False,
                hovertext=f"Take Profit: {target:.2f}",
                name="_target_line"
            ))
        
        # Add entry marker with label
        fig.add_trace(go.Scatter(
            x=[entry_time], 
            y=[entry], 
            mode="markers+text", 
            marker=dict(color=color, size=12, symbol="circle"),
            text=[f"{r['kind']}<br>R={r_multiple:.2f}"], 
            textposition="top center",
            textfont=dict(size=10, color="#2c3e50"),
            showlegend=False,
            hovertext=f"Entry: {entry:.2f}<br>Exit: {exit_time}<br>R: {r_multiple:.2f}",
            name="_entry"
        ))

    fig.update_layout(
        title=title, 
        xaxis_rangeslider_visible=False, 
        template="plotly_white", 
        height=700, 
        width=1400,
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    # write image via kaleido
    try:
        fig.write_image(out_png, engine="kaleido", scale=2)
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido may not be installed): {e}")


def pick_representative_windows(summary_df: pd.DataFrame) -> List[int]:
    # choose windows by max R in each window: best, median, worst
    grp = summary_df.groupby("window").agg(maxR=("R", "max"))
    grp_sorted = grp.sort_values("maxR")
    worst = int(grp_sorted.index[0])
    best = int(grp_sorted.index[-1])
    median_idx = int(grp_sorted.index[len(grp_sorted) // 2])
    return [best, median_idx, worst]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "out", "trades_3day", "representative"))
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if summary.empty:
        print("Empty summary")
        return

    picks = pick_representative_windows(summary)
    print("Picked windows:", picks)

    for w in picks:
        seed = args.base_seed + (w - 1)
        df_1m = generate_synth_1m(days=3, seed=seed)
        df_5m = resample_5m(df_1m)
        df_5m = add_5m_indicators(df_5m, IndicatorConfig())

        trades = summary[summary["window"] == w]
        out_html = os.path.join(args.out_dir, f"window_{w:02d}.html")
        out_png = os.path.join(args.out_dir, f"window_{w:02d}.png")
        title = f"Window {w} (seed={seed}) | trades={len(trades)}"
        plot_plotly_window(df_5m, trades, title, out_html, out_png)
        print("Saved", out_html, out_png)


if __name__ == "__main__":
    main()
