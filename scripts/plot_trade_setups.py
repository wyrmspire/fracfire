"""Plot 1m candlesticks with trade overlays.

Reads a scenario folder containing synthetic_1m.csv and summary.json,
then saves trade_plot.png with entries, stops, targets, and outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

COLOR_UP = "#2ecc71"
COLOR_DN = "#e74c3c"
COLOR_BODY = "#34495e"
COLOR_ENTRY = {
    "win": "#27ae60",
    "loss": "#c0392b",
    "open": "#f1c40f",
}


def load_scenario(data_dir: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df_1m = pd.read_csv(os.path.join(data_dir, "synthetic_1m.csv"))
    df_1m["time"] = pd.to_datetime(df_1m["time"]) if "time" in df_1m.columns else pd.to_datetime(df_1m.iloc[:, 0])
    df_1m = df_1m.set_index("time")
    with open(os.path.join(data_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    return df_1m, summary


def plot_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Minimal candlestick: lines for high-low, rectangles for body
    for ts, row in df.iterrows():
        o = float(row["open"]) if "open" in row else float(row["close"])  # fallback
        c = float(row["close"]) if "close" in row else o
        h = float(row.get("high", max(o, c)))
        l = float(row.get("low", min(o, c)))
        color = COLOR_UP if c >= o else COLOR_DN
        # wick
        ax.plot([ts, ts], [l, h], color=color, linewidth=1)
        # body
        ax.plot([ts, ts], [o, c], color=COLOR_BODY, linewidth=4)


def overlay_trades(ax: plt.Axes, df: pd.DataFrame, entries: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> None:
    # Map entry times to outcomes
    outcome_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for o in outcomes:
        outcome_by_time[pd.Timestamp(o["entry"]["time"])] = o

    for e in entries:
        ts = pd.Timestamp(e["time"])
        entry_price = float(e["entry_price"]) if "entry_price" in e else float(df.loc[ts, "close"]) if ts in df.index else None
        if entry_price is None:
            continue
        stop_price = float(e.get("stop_price", entry_price))
        target_price = float(e.get("target_price", entry_price))
        kind = e.get("kind", "orb")

        outcome = outcome_by_time.get(ts)
        if outcome is None:
            status = "open"
            r_text = "R=?"
        else:
            hit_t = outcome.get("hit_target", False)
            hit_s = outcome.get("hit_stop", False)
            status = "win" if hit_t else ("loss" if hit_s else "open")
            r_text = f"R={float(outcome.get("r_multiple", 0.0)):.2f}"

        ax.scatter([ts], [entry_price], color=COLOR_ENTRY[status], s=40, label=None)
        # plot stop/target lines for +/- 30 minutes
        left = ts - pd.Timedelta(minutes=30)
        right = ts + pd.Timedelta(minutes=30)
        ax.hlines(stop_price, xmin=left, xmax=right, colors="#e74c3c", linestyles="dashed", linewidth=1)
        ax.hlines(target_price, xmin=left, xmax=right, colors="#2ecc71", linestyles="dashed", linewidth=1)
        ax.text(ts, entry_price, f"{kind}\n{r_text}", fontsize=8, color="#2c3e50")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 1m candlesticks with trade overlays")
    parser.add_argument("--data-dir", type=str, required=True, help="Scenario folder with synthetic_1m.csv and summary.json")
    parser.add_argument("--out-img", type=str, default="trade_plot.png")
    args = parser.parse_args()

    df_1m, summary = load_scenario(args.data_dir)
    entries = summary.get("entries", [])
    outcomes = summary.get("outcomes", [])

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_candles(ax, df_1m)
    overlay_trades(ax, df_1m, entries, outcomes)
    ax.set_title("1m Candles with Trade Overlays")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    out_path = os.path.join(args.data_dir, args.out_img)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
