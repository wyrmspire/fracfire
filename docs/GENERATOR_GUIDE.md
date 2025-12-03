# 📘 FracFire Generator Guide

This guide explains how to use the **Physics Engine** components of FracFire to generate, control, and visualize synthetic market data.

## 1. The Price Generator

The `PriceGenerator` is the core engine. It simulates price action tick-by-tick (0.25 increments for MES).

### Basic Usage

```python
from lab.generators import PriceGenerator, MarketState
from datetime import datetime

# Initialize
gen = PriceGenerator(initial_price=5000.0, seed=42)

# Generate a single bar
bar = gen.generate_bar(
    timestamp=datetime.now(),
    state=MarketState.RALLY
)

# Generate a full day (1440 bars)
df = gen.generate_day(
    start_date=datetime(2024, 1, 1),
    auto_transition=True  # Randomly switch states based on session
)
```

### Controlled Generation

You can dictate the exact sequence of states to create specific patterns (Archetypes).

```python
# Define a sequence: (minute_index, state)
sequence = [
    (0, MarketState.RANGING),      # Start ranging
    (60, MarketState.BREAKOUT),    # Breakout after 1 hour
    (120, MarketState.RALLY),      # Rally for the rest of the day
]

df = gen.generate_day(
    start_date=datetime(2024, 1, 1),
    state_sequence=sequence,
    auto_transition=False  # Disable random transitions
)
```

### Output Data Schema

The generator produces a DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime | Bar timestamp |
| `open`, `high`, `low`, `close` | float | Price levels |
| `volume` | int | Simulated volume |
| `open_ticks`, `high_ticks`, ... | int | Prices as integer ticks (0.25 units) |
| `delta_ticks` | int | Close - Prev Close (in ticks) |
| `range_ticks` | int | High - Low (in ticks) |
| `body_ticks` | int | Abs(Close - Open) |
| `state` | str | Market state label (e.g., 'rally') |
| `session` | str | Trading session (e.g., 'rth') |
| `segment_id` | int | ID of the segment (if used) |
| `macro_regime` | str | Day-level label (if provided) |

---

## 2. Market States

States define the statistical behavior of the price action (volatility, bias, trend persistence).

### Standard States (`MarketState`)

*   **RANGING**: Mean-reverting, normal volatility.
*   **FLAT**: Low volatility, tight range.
*   **ZOMBIE**: Slow, low-volatility grind in one direction.
*   **RALLY**: Strong upward bias, high persistence.
*   **IMPULSIVE**: High volatility, large moves.
*   **BREAKOUT**: Sharp upward move.
*   **BREAKDOWN**: Sharp downward move.

### Custom States (`custom_states.py`)

For extreme or specific scenarios, use `custom_state_config`.

```python
from lab.generators import get_custom_state

# Get a "Flash Crash" config
crash_config = get_custom_state("flash_crash")

# Generate bar with this config
bar = gen.generate_bar(
    timestamp=datetime.now(),
    custom_state_config=crash_config
)
```

**Available Custom States:**
*   `mega_volatile`, `flash_crash`, `melt_up`
*   `whipsaw`, `news_spike`
*   `opening_bell`, `closing_squeeze`
*   `slow_bleed`, `dead_zone`

---

## 3. Visualization

Use `ChartVisualizer` to inspect generated data.

```python
from lab.visualizers import quick_chart

# Quick one-liner
quick_chart(df, title="My Synthetic Day", save_path="chart.png")
```

**Features:**
*   **Candlesticks**: Colored up/down.
*   **Volume**: Lower subplot.
*   **State Annotations**: Vertical lines and labels when state changes.
*   **Session Shading**: Background colors for RTH, London, etc.

---

## 4. Fractal State Manager

(Advanced) The `FractalStateManager` manages states across timeframes.

*   **Day State** (e.g., `TREND_DAY`) influences ->
*   **Hour State** (e.g., `IMPULSE`) influences ->
*   **Minute State** (e.g., `RALLY`)

Currently, this is a standalone module used to *plan* state sequences, which are then fed into `PriceGenerator`.

```python
from lab.generators import FractalStateManager

fsm = FractalStateManager()
day_state, hour_state, minute_state = fsm.update(timestamp)
```

---

## 5. Best Practices for ML

1.  **Use Tick Columns**: Train your models on `delta_ticks`, `range_ticks`, etc., not raw prices. This avoids floating-point noise and makes the data scale-invariant.
2.  **Pre-train on Archetypes**: Use `scripts/generate_archetypes.py` to create clean datasets of specific patterns (e.g., "Pure Rally") to teach your model what they look like.
3.  **Validate**: Use `scripts/validate_archetypes.py` to ensure your synthetic data has the statistical properties you expect.
