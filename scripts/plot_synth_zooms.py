"""Load saved synthetic 1m data and create zoomed charts (Plotly HTML + PNG).

This script does NOT run the generator; it reads `out/synth_6m/synth_1m.parquet`.

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  /C/fracfire/.venv312/Scripts/python.exe scripts/plot_synth_zooms.py

Outputs:
  out/synth_6m/zooms/window_*.html
  out/synth_6m/zooms/window_*.png
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_1m(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"1m data not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time')
    return df


def make_candlestick_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='price'))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template='plotly_white', height=700, width=1300)
    return fig


def save_fig(fig: go.Figure, html_path: str, png_path: str):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    # writer via kaleido
    try:
        fig.write_image(png_path, engine='kaleido', scale=2)
    except Exception as e:
        print('PNG export failed:', e)


def pick_zoom_ranges(df: pd.DataFrame):
    # monthly starts (1st of each month), mid-month dates, and last 7 days
    idx = df.index
    start = idx.min().normalize()
    end = idx.max().normalize()
    # generate month start dates between start and end
    months = pd.date_range(start=start, end=end, freq='MS')
    mids = [d + pd.Timedelta(days=14) for d in months]
    last_week_end = end + pd.Timedelta(days=1)
    last_week_start = last_week_end - pd.Timedelta(days=7)

    windows = []
    for d in months:
        center = pd.Timestamp(d)
        windows.append((center - pd.Timedelta(days=3), center + pd.Timedelta(days=3)))
    for d in mids:
        windows.append((pd.Timestamp(d) - pd.Timedelta(days=3), pd.Timestamp(d) + pd.Timedelta(days=3)))
    windows.append((last_week_start, last_week_end))
    # dedupe and clamp to data range
    clean = []
    for a, b in windows:
        a_clamped = max(a, idx.min())
        b_clamped = min(b, idx.max())
        if a_clamped < b_clamped:
            clean.append((a_clamped, b_clamped))
    return clean


def main():
    base = os.path.join(PROJECT_ROOT, 'out', 'synth_6m')
    p1 = os.path.join(base, 'synth_1m.parquet')
    df = load_1m(p1)
    df.sort_index(inplace=True)

    zooms = pick_zoom_ranges(df)
    out_dir = os.path.join(base, 'zooms')
    for i, (a, b) in enumerate(zooms, start=1):
        window_df = df.loc[a:b]
        if window_df.empty:
            continue
        title = f'Synth zoom {i}: {a.date()} to {b.date()}'
        fig = make_candlestick_fig(window_df, title)
        html = os.path.join(out_dir, f'window_{i:02d}.html')
        png = os.path.join(out_dir, f'window_{i:02d}.png')
        save_fig(fig, html, png)
        print('Saved', html, png)


if __name__ == '__main__':
    main()
