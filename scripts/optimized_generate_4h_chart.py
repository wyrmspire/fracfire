"""
Optimized script to generate synthetic 1-minute data, aggregate to 4-hour candles, and visualize with Plotly.
"""

import torch
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Generate synthetic 1-minute data for `days` days
def generate_1m_data(days, start_price=5000, seed=None, device="cuda"):
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_minutes = days * 1440  # 1440 minutes in a day

    # Generate timestamps (use 'min' to avoid deprecated 'T')
    timestamps = pd.date_range(datetime.now(), periods=num_minutes, freq="min")

    # Generate price changes and calculate prices
    price_changes = torch.normal(mean=0, std=1, size=(num_minutes,), device=device)
    prices = torch.cumsum(price_changes, dim=0) + start_price

    # Generate volumes
    volumes = torch.randint(100, 1000, size=(num_minutes,), device=device)

    # Move data back to CPU for pandas compatibility
    data = {
        "timestamp": timestamps,
        "close": prices.cpu().numpy(),
        "volume": volumes.cpu().numpy(),
    }

    return pd.DataFrame(data)

# Aggregate 1-minute data to 4-hour candles
def aggregate_to_4h(data):
    data.set_index("timestamp", inplace=True)
    # Use '4h' to avoid deprecated 'H'
    ohlc = data["close"].resample("4h").ohlc()
    volume = data["volume"].resample("4h").sum()
    ohlc["volume"] = volume
    return ohlc.reset_index()

# Plot the 4-hour data using Plotly (candles + volume subplots)
def plot_4h_data(ohlc, output_html=None):
    from plotly.subplots import make_subplots
    import os

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.8, 0.2],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]]
    )

    fig.add_trace(
        go.Candlestick(
            x=ohlc["timestamp"],
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=ohlc["timestamp"],
            y=ohlc["volume"],
            name="Volume",
            marker_color="#1f77b4",
            opacity=0.6
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title="4H Candlestick Chart",
        xaxis_rangeslider_visible=False,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly"
    )

    # Save to HTML if requested
    if output_html:
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        fig.write_html(output_html)
        print(f"Saved chart to: {output_html}")

    # Also show interactively if available
    try:
        fig.show()
    except Exception as e:
        print(f"Interactive display failed: {e}")

if __name__ == "__main__":
    # Parameters
    DAYS = 30  # Reduced to 30 days for faster testing
    START_PRICE = 5000
    SEED = 42

    # Generate and process data
    print("Generating 1-minute data...")
    data = generate_1m_data(DAYS, start_price=START_PRICE, seed=SEED)
    print("Aggregating to 4-hour candles...")
    ohlc = aggregate_to_4h(data)
    print("Plotting data...")
    output_path = "charts/optimized_4h_chart.html"
    plot_4h_data(ohlc, output_html=output_path)