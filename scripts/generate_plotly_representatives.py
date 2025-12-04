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
    fig = go.Figure()
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

    # overlay trades
    for _, r in trades.iterrows():
        t = pd.to_datetime(r["time"])  # time index in summary.csv
        entry = float(r["entry"])
        stop = float(r["stop"]) if not pd.isna(r["stop"]) else None
        target = float(r["target"]) if not pd.isna(r["target"]) else None
        color = "green" if float(r["R"]) > 0 else "red"
        fig.add_trace(go.Scatter(x=[t], y=[entry], mode="markers+text", marker=dict(color=color, size=10),
                                 text=[f"{r['kind']}\nR={float(r['R']):.2f}"], textposition="top center", showlegend=False))
        if stop is not None:
            fig.add_trace(go.Scatter(x=[df_5m.index[0], df_5m.index[-1]], y=[stop, stop], mode="lines",
                                     line=dict(color="firebrick", dash="dash"), showlegend=False))
        if target is not None:
            fig.add_trace(go.Scatter(x=[df_5m.index[0], df_5m.index[-1]], y=[target, target], mode="lines",
                                     line=dict(color="green", dash="dash"), showlegend=False))

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_white", height=700, width=1400)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    # write image via kaleido
    fig.write_image(out_png, engine="kaleido", scale=2)


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
