# Setup Engine and Trade Features

This doc describes the unified API for detecting setups, evaluating trades, and producing ML-friendly features.

## Data Models

- `SetupEntry`:
  - `time`: timestamp on 1m tape (decision at 5-min boundaries by default)
  - `direction`: `long | short`
  - `kind`: `orb | level_scalp | ema20_vwap_revert | ema200_continuation | breakout | reversal | opening_push | moc`
  - `entry_price`, `stop_price`, `target_price`
  - `context`: dict for setup-specific fields (e.g., `or_high`, `or_low`)
- `SetupOutcome`:
  - `entry`: the `SetupEntry`
  - `hit_target`, `hit_stop`, `exit_time`
  - `r_multiple`, `mfe`, `mae`

## Indicators

- `IndicatorConfig` (in `src/core/detector/indicators.py`):
  - **EMA (Exponential Moving Average)**:
    - `ema_fast=20`, `ema_slow=200`
  - **RSI (Relative Strength Index)**:
    - `rsi_period=14`
  - **Bollinger Bands**:
    - `bb_period=20`, `bb_std=2.0`
  - **ATR (Average True Range)**:
    - `atr_period=14`
  - **MACD (Moving Average Convergence Divergence)**:
    - `macd_fast=12`, `macd_slow=26`, `macd_signal=9`
  - **Volume Indicators**:
    - `volume_sma_period=20`

- `add_1m_indicators(df_1m, cfg)` and `add_5m_indicators(df_5m, cfg)` add columns:
  - `ema_fast`, `ema_slow`, `vwap`
  - `rsi`
  - `bb_upper`, `bb_middle`, `bb_lower`
  - `atr`
  - `macd`, `macd_signal`, `macd_histogram`
  - `volume_sma`, `volume_ratio`

- Decision anchors on 1m:
  - `DecisionAnchorConfig(modulo=5, offset=4)`
  - `mark_decision_points_1m(df_1m, cfg)` → boolean Series marking 5-min closes

## Setup Engine

- `SetupConfig` (in `src/core/detector/engine.py`):
  - `indicator_cfg`: `IndicatorConfig`
  - One `SetupFamilyConfig` per setup family (enabled + `params` dict)
- `run_setups(df_1m, df_5m, cfg)` → `List[SetupOutcome]` (run all enabled families)

## Setup Types

### 1. ORB (Opening Range Breakout)

- `find_opening_orb_continuations_1m(df_1m, cfg, anchor)`
- `evaluate_orb_entry_1m(df_1m, entry, cfg)`
- `summarize_orb_day_1m(df_1m, cfg, anchor)`

**Configuration (`ORBConfig`):**
- `session_start_hour=14`, `session_start_minute=30` (09:30 ET in UTC)
- `or_minutes=30` (opening range duration)
- `buffer_ticks=2.0` (extra ticks beyond OR for breakout)
- `min_drive_rr=0.5`, `max_counter_rr=0.3` (drive/counter thresholds)
- `stop_type="or"` or `"bar"` (stop placement)
- `target_rr=2.0`, `max_hold_bars=48`

### 2. EMA200 Continuation

Trend continuation setup looking for pullbacks to EMA200 with RSI confirmation.

**Configuration (`EMA200ContinuationConfig`):**
- `ema_proximity_ticks=4.0` (distance to EMA200)
- `rsi_oversold=35.0`, `rsi_overbought=65.0`
- `min_volume_ratio=0.8`
- `stop_atr_mult=1.5`
- `target_rr=2.0`, `max_hold_bars=48`

**Entry criteria:**
- **Long**: Price near EMA200 from above, RSI < 35, volume >= 0.8x average
- **Short**: Price near EMA200 from below, RSI > 65, volume >= 0.8x average

### 3. Breakout

Price breaking above/below recent highs/lows with volume and MACD confirmation.

**Configuration (`BreakoutConfig`):**
- `lookback_bars=20` (bars to find high/low)
- `buffer_ticks=2.0`
- `min_volume_ratio=1.2`
- `macd_threshold=0.0`
- `stop_atr_mult=1.5`
- `target_rr=2.5`, `max_hold_bars=48`

**Entry criteria:**
- **Long**: Close > recent_high + buffer, volume > 1.2x average, MACD histogram > 0
- **Short**: Close < recent_low - buffer, volume > 1.2x average, MACD histogram < 0

### 4. Reversal

Mean reversion at Bollinger Band extremes with RSI divergence and wick patterns.

**Configuration (`ReversalConfig`):**
- `bb_touch_ticks=2.0` (proximity to BB)
- `rsi_extreme_long=30.0`, `rsi_extreme_short=70.0`
- `min_wick_ratio=0.4` (wick size vs range)
- `stop_atr_mult=1.5`
- `target_rr=2.0`, `max_hold_bars=36`

**Entry criteria:**
- **Long**: Touch lower BB, RSI < 30, rejection wick > 40% of range
- **Short**: Touch upper BB, RSI > 70, rejection wick > 40% of range

### 5. Opening Push

Captures momentum in the first 30 minutes of the session.

**Configuration (`OpeningPushConfig`):**
- `session_start_hour=14`, `session_start_minute=30`
- `push_window_minutes=30`
- `min_move_ticks=8.0` (minimum move from open)
- `min_volume_ratio=1.5`
- `stop_atr_mult=2.0`
- `target_rr=2.0`, `max_hold_bars=48`

**Entry criteria:**
- **Long**: Move up >= 8 ticks from open, volume > 1.5x average
- **Short**: Move down >= 8 ticks from open, volume > 1.5x average

### 6. MOC (Market on Close)

Trend continuation or reversal behavior in the last hour before session close.

**Configuration (`MOCConfig`):**
- `session_end_hour=20`, `session_end_minute=0`
- `moc_window_minutes=60` (last hour)
- `min_directional_bars=3` (consecutive bars in same direction)
- `min_volume_ratio=1.2`
- `stop_atr_mult=1.5`
- `target_rr=1.5`, `max_hold_bars=24`

**Entry criteria:**
- **Long**: 3+ consecutive up bars, volume > 1.2x average
- **Short**: 3+ consecutive down bars, volume > 1.2x average

## Usage Examples

### Running All Setups

```python
from src.core.detector.engine import SetupConfig, SetupFamilyConfig, run_setups
from src.core.detector.indicators import IndicatorConfig

# Configure which setups to run
cfg = SetupConfig(
    indicator_cfg=IndicatorConfig(
        ema_fast=20,
        ema_slow=200,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
    ),
    orb=SetupFamilyConfig(enabled=True, params={"target_rr": 2.0}),
    ema200_continuation=SetupFamilyConfig(enabled=True),
    breakout=SetupFamilyConfig(enabled=True, params={"lookback_bars": 20}),
    reversal=SetupFamilyConfig(enabled=True),
    opening_push=SetupFamilyConfig(enabled=True),
    moc=SetupFamilyConfig(enabled=True),
)

# Run all enabled setups
outcomes = run_setups(df_1m, df_5m, cfg)
```

### Demo Script

Run the comprehensive demo to see all setups in action:

```bash
python scripts/demo_all_setups.py --days 5 --seed 42
```

This will:
1. Generate synthetic 1m data
2. Resample to 5m bars
3. Compute all indicators
4. Detect all setup types
5. Evaluate outcomes
6. Display summary statistics

### Indicator Reference

All indicators are computed automatically when using `add_5m_indicators()` or `add_1m_indicators()`:

| Indicator | Column Names | Description |
|-----------|-------------|-------------|
| EMA | `ema_fast`, `ema_slow` | Exponential moving averages (20, 200) |
| RSI | `rsi` | Relative Strength Index (14 period) |
| Bollinger Bands | `bb_upper`, `bb_middle`, `bb_lower` | 20-period SMA ± 2 std dev |
| ATR | `atr` | Average True Range (14 period) |
| MACD | `macd`, `macd_signal`, `macd_histogram` | 12/26/9 MACD configuration |
| Volume | `volume_sma`, `volume_ratio` | 20-period volume average and ratio |
| VWAP | `vwap` | Volume-weighted average price |

### Customizing Setups

Each setup family can be customized via the `params` dict:

```python
# Custom breakout with tighter parameters
breakout_cfg = SetupFamilyConfig(
    enabled=True,
    params={
        "lookback_bars": 15,      # shorter lookback
        "min_volume_ratio": 1.5,  # higher volume requirement
        "target_rr": 3.0,         # larger target
    }
)

# Custom reversal with wider BB tolerance
reversal_cfg = SetupFamilyConfig(
    enabled=True,
    params={
        "bb_touch_ticks": 4.0,         # more room for BB touch
        "rsi_extreme_long": 25.0,      # more extreme RSI
        "rsi_extreme_short": 75.0,
        "min_wick_ratio": 0.5,         # larger wick requirement
    }
)
```

## Generic Evaluation

- `evaluate_generic_entry_1m(df_1m, entry, max_minutes)` evaluates any `SetupEntry` minute-by-minute.

## Trade Features

- `compute_trade_features(outcome, df_ind)` (in `src/evaluation/trade_features.py`):
  - Price distances to `ema_fast`, `ema_slow`, `vwap`
  - Candle geometry at entry (`body_frac`, `wick_up_frac`, `wick_dn_frac`)
  - Timing (`hour`, `minute`, `bars_to_exit`)
  - Risk, `r_multiple`, `mfe`, `mae`, `mfe_mae_ratio`
  - Simple behavioral flags: `low_rr_flag`, `chase_flag`, `hesitation_flag`
  - Includes `ctx_*` fields from `entry.context`

## Bad Trade Injection (for ML practice)

- `BadTradeConfig` (in `trade_features.py`):
  - `enable_hesitation`, `hesitation_minutes`
  - `enable_chase`, `chase_window_minutes`
  - `max_variants_per_trade`
- `inject_bad_trade_variants(df_1m_ind, outcomes, cfg)` returns `List[SetupOutcome]` with `entry.context['variant']` set to `hesitation` or `chase`.

## Runner

- `scripts/run_orb_trade_runner.py`:
  - Generates synthetic 1m, computes 1m indicators
  - Uses 1m-based ORB detection with 5-min decision anchors
  - Extracts features and writes:
    - `synthetic_1m.csv`, `synthetic_5m.csv`
    - `summary.json` (entries, outcomes, miss, features)
    - `trades_features.csv` (tabular features)
  - Bad-trade knobs:
    - `--inject-hesitation --hesitation-minutes 5`
    - `--inject-chase --chase-window-minutes 10`
    - `--max-variants-per-trade 2`

## Future Work

- ✅ **COMPLETED**: EMA200 continuation, breakout, reversal, opening push, MOC setups
- ✅ **COMPLETED**: RSI, Bollinger Bands, ATR, MACD, Volume indicators
- Add setup families: level scalp, EMA/VWAP reversion
- Enhanced plotting with indicator overlays on charts
- Richer behavioral labels for ML (overtrading, wrong time-of-day, chasing extremes)
- Multi-timeframe confirmation setups
- Adaptive parameters based on market regime

## Plotting Overlay

- `scripts/plot_trade_setups.py` loads a scenario folder containing `synthetic_1m.csv` and `summary.json`.
- Produces `trade_plot.png` with:
  - 1m candlesticks
  - Entry markers colored by outcome (green=target, red=stop, yellow=open)
  - Horizontal lines for stop/target around entry time
  - Legend with trade kind and R
- Intended for GUI reuse: import the plotting function or call the script.
