"""
Compare real continuous contract data vs synthetic over last 3 months.
Generates synthetic only on real trading-hour timestamps and plots 15m candlesticks.
Opens the chart in the default browser and saves HTML to charts/compare_real_synth.html.
"""

import os
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


def load_real_data(path, days=90):
    print(f"Loading real data: {path}")
    df = pd.read_json(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    cutoff = df['time'].max() - pd.Timedelta(days=days)
    df = df[df['time'] >= cutoff]
    return df


def generate_synth(timestamps, start_price=5000.0, seed=42, vol_scale=1.0):
    """Generate synthetic minute data exactly on given timestamps (no weekends)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating synthetic on {device}")
    if seed is not None:
        torch.manual_seed(seed)
    num_steps = len(timestamps)
    price_changes = torch.normal(0, 1, size=(num_steps,), device=device) * float(vol_scale)
    prices = torch.cumsum(price_changes, dim=0) + float(start_price)
    volumes = torch.randint(1, 1000, size=(num_steps,), device=device)
    df = pd.DataFrame({
        'time': pd.to_datetime(timestamps),
        'close': prices.cpu().numpy(),
        'volume': volumes.cpu().numpy()
    })
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df[['open', 'close']].max(axis=1) + 0.1
    df['low'] = df[['open', 'close']].min(axis=1) - 0.1
    return df


def agg_to_timeframe(df, freq='15min'):
    df = df.copy().set_index('time')
    ohlc = df['close'].resample(freq).ohlc()
    vol = df['volume'].resample(freq).sum()
    ohlc['volume'] = vol
    ohlc = ohlc.dropna()
    return ohlc.reset_index()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vol-scale', type=float, default=1.0, help='Multiply synthetic noise amplitude')
    args = parser.parse_args()
    # Load real and synth (on real timestamps)
    real = load_real_data('data/continuous_contract.json', days=90)
    print(f"Real: {len(real)} rows, {real['time'].min()} to {real['time'].max()}")
    synth = generate_synth(real['time'].values, start_price=real['close'].iloc[0], vol_scale=args.vol_scale)
    print(f"Synth: {len(synth)} rows")

    # Precompute timeframes
    tfs = ['15min', '1h', '4h']
    real_aggs = {tf: agg_to_timeframe(real, freq=tf) for tf in tfs}
    synth_aggs = {tf: agg_to_timeframe(synth, freq=tf) for tf in tfs}
    for tf in tfs:
        print(f"{tf}: real={len(real_aggs[tf])} synth={len(synth_aggs[tf])}")

    # Print amplitude metrics (mean candle range) for 15m
    def mean_range(df):
        if len(df) == 0:
            return 0.0
        return float((df['high'] - df['low']).mean())
    real_rng = mean_range(real_aggs['15min'])
    synth_rng = mean_range(synth_aggs['15min'])
    ratio = (synth_rng / real_rng) if real_rng > 0 else 0.0
    print(f"15m mean range: real={real_rng:.2f} synth={synth_rng:.2f} ratio={ratio:.2f} (vol-scale={args.vol_scale})")

    # Build figure (add all TFs as separate traces, show 15m initially)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Real Data (3 months)', 'Synthetic Data (3 months)')
    )

    visible = []
    for tf in tfs:
        r = real_aggs[tf]
        s = synth_aggs[tf]
        vis = (tf == '15min')
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(r))),
                open=r['open'],
                high=r['high'],
                low=r['low'],
                close=r['close'],
                customdata=r['time'],
                hovertemplate='<b>Real</b><br>Time: %{customdata}<br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<extra></extra>',
                name=f'Real ({tf})',
                visible=vis
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(s))),
                open=s['open'],
                high=s['high'],
                low=s['low'],
                close=s['close'],
                customdata=s['time'],
                hovertemplate='<b>Synthetic</b><br>Time: %{customdata}<br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<extra></extra>',
                name=f'Synthetic ({tf})',
                visible=vis
            ),
            row=2, col=1
        )

    # Compute baseline ranges per TF
    ranges = {}
    for tf in tfs:
        r = real_aggs[tf]
        s = synth_aggs[tf]
        ranges[tf] = {
            'top': [float(r['low'].min()), float(r['high'].max())] if len(r) else [0.0, 1.0],
            'bot': [float(s['low'].min()), float(s['high'].max())] if len(s) else [0.0, 1.0],
            'x': [0, max(len(r), len(s)) - 1]
        }

    # Updatemenus for timeframe switching + reset Y to fit
    buttons = []
    for i, tf in enumerate(tfs):
        vis = [False] * (2 * len(tfs))
        vis[2 * i] = True
        vis[2 * i + 1] = True
        buttons.append(dict(
            label=f'Timeframe: {tf}',
            method='update',
            args=[
                {'visible': vis},
                {
                    'yaxis': {'range': ranges[tf]['top']},
                    'yaxis2': {'range': ranges[tf]['bot']},
                    'xaxis': {'range': ranges[tf]['x']},
                    'xaxis2': {'range': ranges[tf]['x']},
                    'title': f'Real vs Synthetic (3 Months, {tf})'
                }
            ]
        ))

    # Add a separate reset-y button
    buttons.append(dict(
        label='Reset Y (15m)',
        method='relayout',
        args=[{'yaxis.range': ranges['15min']['top'], 'yaxis2.range': ranges['15min']['bot']}]
    ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0.5,
            y=1.12,
            xanchor='center',
            yanchor='top',
            buttons=buttons
        )],
        height=1400,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False,
        title='Real vs Synthetic (3 Months, 15m)',
        dragmode='zoom',
        yaxis={'fixedrange': False},
        yaxis2={'fixedrange': False}
    )

    # Save and open
    os.makedirs('charts', exist_ok=True)
    out = 'charts/compare_real_synth.html'
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"Saved to: {out}")
    fig.show()
