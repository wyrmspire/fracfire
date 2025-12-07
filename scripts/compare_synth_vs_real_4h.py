"""Compare a synthetic 1m parquet vs the real continuous contract at 4H.

Saves:
 - out_dir/real_4h.parquet
 - out_dir/synth_4h.parquet
 - out_dir/metrics.csv
 - out_dir/compare_4h.html
 - out_dir/compare_4h.png

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  /C/fracfire/.venv312/Scripts/python.exe scripts/compare_synth_vs_real_4h.py \
      --real src/data/continuous_contract.json --synth out/synth_new_3mo/synth_1m.parquet --months 3 --out-dir out/compare_4h_new
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import math
import time
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import RealDataLoader


def resample_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_1m.resample("4H").agg(ohlc).dropna()


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff().dropna()


def acf_lags(x: pd.Series, lags: int = 10) -> np.ndarray:
    out = []
    x = x.dropna()
    n = len(x)
    if n < 2:
        return np.array([np.nan] * lags)
    x = x - x.mean()
    denom = float((x * x).mean())
    if denom == 0:
        return np.array([np.nan] * lags)
    for lag in range(1, lags + 1):
        num = float((x.iloc[lag:] * x.shift(lag).iloc[lag:]).mean())
        out.append(num / denom)
    return np.array(out)


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if _HAS_SCIPY:
        stat, p = stats.ks_2samp(a, b)
        return float(stat), float(p)
    # fallback: empirical KS implementation
    a = np.sort(a)
    b = np.sort(b)
    na = len(a); nb = len(b)
    vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, vals, side='right') / na
    cdf_b = np.searchsorted(b, vals, side='right') / nb
    stat = np.max(np.abs(cdf_a - cdf_b))
    # p-value unknown in fallback
    return float(stat), float('nan')


def compute_metrics(df_4h: pd.DataFrame) -> dict:
    body = (df_4h["close"] - df_4h["open"]).abs()
    full = (df_4h["high"] - df_4h["low"]).replace(0, pd.NA)
    wick_frac = 1.0 - (body / full)
    wick_frac = wick_frac.fillna(0.0)

    ret = log_returns(df_4h["close"])
    ret_std = float(ret.std()) if len(ret) > 0 else float('nan')
    if len(ret) > 2 and not math.isnan(ret_std) and ret_std > 0:
        z = (ret - ret.mean()) / ret_std
        kurt = float((z ** 4).mean())
        kurt_excess = kurt - 3.0
    else:
        kurt = float('nan'); kurt_excess = float('nan')

    acf_abs = acf_lags(ret.abs(), lags=10)

    return {
        "n_bars": len(df_4h),
        "avg_range": float((df_4h["high"] - df_4h["low"]).mean()),
        "avg_body": float(body.mean()),
        "avg_wick_frac": float(wick_frac.mean()),
        "kurtosis": kurt,
        "kurtosis_excess": kurt_excess,
        "volatility": ret_std,
        "acf_abs_lag1": float(acf_abs[0]) if len(acf_abs) >= 1 else float('nan'),
        "acf_abs_lag1_to_10": ",".join([f"{v:.6f}" if not np.isnan(v) else "nan" for v in acf_abs]),
    }


def make_plotly(real_4h: pd.DataFrame, synth_4h: pd.DataFrame, out_html: str, out_png: str, title: str = "Compare 4H") -> None:
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}], [{"colspan": 2, "type": "xy"}, None]],
                        subplot_titles=("Real 4H", "Synth 4H", "Q-Q: returns"))

    fig.add_trace(go.Candlestick(x=real_4h.index, open=real_4h['open'], high=real_4h['high'], low=real_4h['low'], close=real_4h['close'], name='Real'), row=1, col=1)
    fig.add_trace(go.Candlestick(x=synth_4h.index, open=synth_4h['open'], high=synth_4h['high'], low=synth_4h['low'], close=synth_4h['close'], name='Synth'), row=1, col=2)

    # Q-Q data
    rret = log_returns(real_4h['close']).dropna().values
    sret = log_returns(synth_4h['close']).dropna().values
    # make quantiles
    q = np.linspace(0.01, 0.99, 100)
    rq = np.quantile(rret, q) if len(rret) > 0 else np.array([np.nan]*len(q))
    sq = np.quantile(sret, q) if len(sret) > 0 else np.array([np.nan]*len(q))
    fig.add_trace(go.Scatter(x=rq, y=sq, mode='markers', name='Q-Q'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[rq.min(), rq.max()], y=[rq.min(), rq.max()], mode='lines', name='y=x', line=dict(dash='dash', color='gray')), row=2, col=1)

    fig.update_layout(height=900, width=1400, title_text=title)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    try:
        fig.write_image(out_png)
    except Exception as e:
        print("Warning: failed to write PNG (kaleido may be missing):", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, required=True)
    parser.add_argument('--synth', type=str, required=True)
    parser.add_argument('--months', type=int, default=3)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real data from {args.real}...")
    loader = RealDataLoader()
    real_1m = loader.load_json(Path(args.real))

    print(f"Loading synth data from {args.synth}...")
    synth_1m = pd.read_parquet(args.synth)

    # Trim to last N days (~months*30)
    days = int(args.months * 30)
    real_1m_sorted = real_1m.sort_index()
    synth_1m_sorted = synth_1m.sort_index()

    real_trim = real_1m_sorted.iloc[-days * 24 * 60 :].copy()
    synth_trim = synth_1m_sorted.iloc[-days * 24 * 60 :].copy()

    real_4h = resample_4h(real_trim)
    synth_4h = resample_4h(synth_trim)

    real_4h.to_parquet(out_dir / 'real_4h.parquet')
    synth_4h.to_parquet(out_dir / 'synth_4h.parquet')

    metrics_real = compute_metrics(real_4h)
    metrics_synth = compute_metrics(synth_4h)

    # KS test on returns
    rret = log_returns(real_4h['close']).dropna().values
    sret = log_returns(synth_4h['close']).dropna().values
    ks_stat, ks_p = ks_two_sample(rret, sret)

    # Jump inter-arrival: threshold = 3 * std of real returns
    if len(rret) > 0:
        thresh = 3.0 * np.nanstd(rret)
    else:
        thresh = 0.0

    def inter_arrival_times(returns: np.ndarray, threshold: float):
        idx = np.where(np.abs(returns) > threshold)[0]
        if len(idx) <= 1:
            return np.array([])
        return np.diff(idx)

    iat_real = inter_arrival_times(rret, thresh)
    iat_synth = inter_arrival_times(sret, thresh)

    # Summarize metrics
    out_metrics = {
        'metric': [],
        'real': [],
        'synth': []
    }

    for k, v in metrics_real.items():
        out_metrics['metric'].append(k)
        out_metrics['real'].append(v)
        out_metrics['synth'].append(metrics_synth.get(k, float('nan')))

    out_metrics['metric'].append('ks_stat')
    out_metrics['real'].append(ks_stat)
    out_metrics['synth'].append(ks_p)

    out_metrics['metric'].append('jump_iat_real_mean')
    out_metrics['real'].append(float(np.nanmean(iat_real)) if len(iat_real) > 0 else float('nan'))
    out_metrics['synth'].append(float(np.nanmean(iat_synth)) if len(iat_synth) > 0 else float('nan'))

    df_out = pd.DataFrame(out_metrics)
    df_out.to_csv(out_dir / 'metrics.csv', index=False)
    print(f"Saved metrics to {out_dir / 'metrics.csv'}")

    # Create plotly visuals
    html = out_dir / 'compare_4h.html'
    png = out_dir / 'compare_4h.png'
    make_plotly(real_4h, synth_4h, str(html), str(png), title=f"Compare 4H: {Path(args.synth).name}")
    print(f"Saved visuals to {html} and {png}")

    # Print top-level metrics and a short verdict line
    print('\nTop-level metrics (real vs synth):')
    print(df_out.to_string(index=False))

    # Simple verdict: compare kurtosis_excess and acf_abs_lag1
    kurt_real = metrics_real.get('kurtosis_excess', float('nan'))
    kurt_synth = metrics_synth.get('kurtosis_excess', float('nan'))
    acf_real = metrics_real.get('acf_abs_lag1', float('nan'))
    acf_synth = metrics_synth.get('acf_abs_lag1', float('nan'))

    verdict = []
    if not math.isnan(kurt_real) and not math.isnan(kurt_synth):
        if abs(kurt_synth - kurt_real) < abs(kurt_real - 0):
            verdict.append(f"kurtosis_excess closer to real: Δ={abs(kurt_synth-kurt_real):.4f}")
    if not math.isnan(acf_real) and not math.isnan(acf_synth):
        verdict.append(f"acf_abs_lag1: real={acf_real:.4f}, synth={acf_synth:.4f}")

    print('\nVerdict: ' + ("; ".join(verdict) if verdict else "No strong verdict (insufficient data)"))


if __name__ == '__main__':
    main()
