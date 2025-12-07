"""
Compare real continuous contract 1-minute data with synthetic generator.
- Loads `data/continuous_contract.json` (assumes UTC timestamps).
- Generates synthetic 1-minute closes for the same timestamp range using GPU if available.
- Aggregates both to 1-day OHLCV and produces an HTML comparison page at `charts/compare_continuous_vs_synth.html`.
"""

import os
from datetime import datetime
import json
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _quick_first_last(path, look_bytes=20000):
    """Quickly extract first and last timestamp strings from the large JSON file using regex.
    Returns (first_ts_str, last_ts_str) as ISO strings, or (None, None) if not found."""
    import re

    first_ts = None
    last_ts = None
    pattern = re.compile(r'"time"\s*:\s*"([^"]+)"')

    with open(path, 'rb') as f:
        head = f.read(look_bytes).decode('utf-8', errors='ignore')
        m = pattern.search(head)
        if m:
            first_ts = m.group(1)

        # tail
        try:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - look_bytes)
            f.seek(start)
            tail = f.read().decode('utf-8', errors='ignore')
            matches = pattern.findall(tail)
            if matches:
                last_ts = matches[-1]
        except Exception:
            pass

    return first_ts, last_ts


def load_continuous_range(path, start_ts=None, end_ts=None):
    """Stream the JSON array and only parse objects whose 'time' falls between start_ts and end_ts.
    Assumes the file is sorted ascending by time so we can stop once we've passed end_ts."""
    import re

    print(f"Streaming continuous data from: {path}")
    time_pat = re.compile(r'"time"\s*:\s*"([^"]+)"')

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    start_pd = pd.to_datetime(start_ts).tz_convert('UTC') if start_ts is not None and pd.to_datetime(start_ts).tzinfo else (pd.to_datetime(start_ts).tz_localize('UTC') if start_ts is not None else None)
    end_pd = pd.to_datetime(end_ts).tz_convert('UTC') if end_ts is not None and pd.to_datetime(end_ts).tzinfo else (pd.to_datetime(end_ts).tz_localize('UTC') if end_ts is not None else None)

    buf = ''
    with open(path, 'r', encoding='utf-8') as f:
        # skip initial '['
        first_ch = f.read(1)
        for line in f:
            if not line.strip():
                continue
            buf += line
            if line.strip().endswith('},') or line.strip().endswith('}'):
                obj_text = buf.rstrip(',\n ')
                # quickly extract time string before full json.loads
                m = time_pat.search(obj_text)
                if not m:
                    buf = ''
                    continue
                ts_str = m.group(1)
                try:
                    ts = pd.to_datetime(ts_str).tz_convert('UTC') if pd.to_datetime(ts_str).tzinfo else pd.to_datetime(ts_str).tz_localize('UTC')
                except Exception:
                    buf = ''
                    continue

                if start_pd is not None and ts < start_pd:
                    buf = ''
                    continue

                if end_pd is not None and ts > end_pd:
                    # since file is sorted we can stop reading further
                    break

                try:
                    obj = json.loads(obj_text)
                except Exception:
                    buf = ''
                    continue

                timestamps.append(ts)
                opens.append(obj.get('open'))
                highs.append(obj.get('high'))
                lows.append(obj.get('low'))
                closes.append(obj.get('close'))
                volumes.append(obj.get('volume'))
                buf = ''

    df = pd.DataFrame({
        'timestamp': pd.DatetimeIndex(timestamps),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def generate_synth_between(start_ts, end_ts, start_price=5000, seed=42, device='cuda'):
    # generate one close per minute between start_ts and end_ts inclusive
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Generating synthetic data on device: {device}")

    if seed is not None:
        torch.manual_seed(seed)

    # ensure timezone-naive for arithmetic or use pandas timestamps
    start = pd.to_datetime(start_ts).tz_convert('UTC') if pd.to_datetime(start_ts).tzinfo else pd.to_datetime(start_ts).tz_localize('UTC')
    end = pd.to_datetime(end_ts).tz_convert('UTC') if pd.to_datetime(end_ts).tzinfo else pd.to_datetime(end_ts).tz_localize('UTC')

    num_minutes = int(((end - start).total_seconds() // 60) + 1)
    print(f"Minutes to generate: {num_minutes}")

    timestamps = pd.date_range(start=start, periods=num_minutes, freq='min', tz='UTC')

    price_changes = torch.normal(mean=0.0, std=1.0, size=(num_minutes,), device=device)
    prices = torch.cumsum(price_changes, dim=0) + float(start_price)
    volumes = torch.randint(1, 1000, size=(num_minutes,), device=device)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices.cpu().numpy(),
        'volume': volumes.cpu().numpy()
    })
    # make synthetic OHLC: open = previous close, high = max(prev close, close) + tiny noise, low = min(prev, close) - tiny noise
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df[['open', 'close']].max(axis=1) + (abs(df['close'] - df['open']) * 0.1)
    df['low'] = df[['open', 'close']].min(axis=1) - (abs(df['close'] - df['open']) * 0.1)
    return df


def aggregate_to_1d(df):
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    # resample to 1 day
    ohlc = df['close'].resample('1d').ohlc()
    volume = df['volume'].resample('1d').sum()
    ohlc['volume'] = volume
    ohlc = ohlc.dropna(how='all')
    ohlc = ohlc.reset_index()
    return ohlc


def build_comparison_page(real_1d, synth_1d, out_path):
    # align on intersecting dates
    real_1d['date'] = pd.to_datetime(real_1d['timestamp']).dt.normalize()
    synth_1d['date'] = pd.to_datetime(synth_1d['timestamp']).dt.normalize()

    merged = pd.merge(real_1d, synth_1d, on='date', how='inner', suffixes=('_real', '_synth'))
    if merged.empty:
        print('No overlapping dates between datasets after aggregation.')

    # compute stats
    merged['close_diff'] = merged['close_synth'] - merged['close_real']
    mean_abs_diff = merged['close_diff'].abs().mean()
    corr = merged['close_real'].corr(merged['close_synth'])

    # Plot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        specs=[[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "bar"}]])

    # real candlestick
    fig.add_trace(go.Candlestick(x=merged['date'], open=merged['open_real'], high=merged['high_real'], low=merged['low_real'], close=merged['close_real'], name='Real'), row=1, col=1)
    # synth close line
    fig.add_trace(go.Scatter(x=merged['date'], y=merged['close_synth'], mode='lines+markers', name='Synth Close', line=dict(color='orange')), row=1, col=1)

    # diff plot
    fig.add_trace(go.Bar(x=merged['date'], y=merged['close_diff'], name='Synth - Real (close)', marker_color='purple'), row=2, col=1)

    # volumes both
    fig.add_trace(go.Bar(x=merged['date'], y=merged['volume_real'], name='Real Volume', marker_color='grey', opacity=0.6), row=3, col=1)
    fig.add_trace(go.Bar(x=merged['date'], y=merged['volume_synth'], name='Synth Volume', marker_color='blue', opacity=0.4), row=3, col=1)

    fig.update_layout(title=f"Daily Comparison — mean_abs_diff={mean_abs_diff:.3f}, corr={corr:.3f}", xaxis_rangeslider_visible=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved comparison page to: {out_path}")
    return merged, mean_abs_diff, corr


if __name__ == '__main__':
    CONT_PATH = os.path.join('data', 'continuous_contract.json')
    out_html = os.path.join('charts', 'compare_continuous_vs_synth.html')
    # Quickly find first/last timestamps in the large JSON without parsing everything
    first_ts_str, last_ts_str = _quick_first_last(CONT_PATH)
    if not last_ts_str:
        raise RuntimeError('Could not determine timestamps in continuous file')

    # Default compare window: last 30 days in the continuous file
    end_pd = pd.to_datetime(last_ts_str).tz_convert('UTC') if pd.to_datetime(last_ts_str).tzinfo else pd.to_datetime(last_ts_str).tz_localize('UTC')
    start_pd = end_pd - pd.Timedelta(days=30)
    print(f"Comparing window: {start_pd} to {end_pd}")

    # Stream only that window from disk (fast)
    real_df = load_continuous_range(CONT_PATH, start_ts=start_pd.isoformat(), end_ts=end_pd.isoformat())
    if real_df.empty:
        raise RuntimeError('No real data found in selected window')

    synth_df = generate_synth_between(real_df['timestamp'].iloc[0], real_df['timestamp'].iloc[-1], start_price=real_df['close'].iloc[0], seed=123)

    real_1d = aggregate_to_1d(real_df)
    synth_1d = aggregate_to_1d(synth_df)

    merged, mad, corr = build_comparison_page(real_1d, synth_1d, out_html)

    # Print a few lines of comparison
    print('Sample comparison rows:')
    print(merged[['date', 'close_real', 'close_synth', 'close_diff']].head())
    print(f"Mean absolute daily close diff: {mad:.4f}")
    print(f"Daily close correlation: {corr:.4f}")
