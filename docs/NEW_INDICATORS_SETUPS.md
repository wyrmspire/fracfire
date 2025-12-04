# New Indicators and Setups Guide

This document provides a comprehensive overview of the newly added indicators and trading setups in FracFire.

## 📊 New Indicators (5 Added)

### 1. RSI (Relative Strength Index)
- **Period**: 14 (configurable via `rsi_period`)
- **Purpose**: Momentum oscillator measuring overbought/oversold conditions
- **Range**: 0-100
- **Usage**: 
  - RSI < 30: Oversold (potential long reversal)
  - RSI > 70: Overbought (potential short reversal)
- **Column**: `rsi`

### 2. Bollinger Bands
- **Period**: 20 (configurable via `bb_period`)
- **Standard Deviations**: 2.0 (configurable via `bb_std`)
- **Purpose**: Volatility bands for mean reversion and breakout detection
- **Columns**: `bb_upper`, `bb_middle`, `bb_lower`
- **Usage**:
  - Price touching lower band → potential long reversal
  - Price touching upper band → potential short reversal
  - Price outside bands → extended move, possible reversal

### 3. ATR (Average True Range)
- **Period**: 14 (configurable via `atr_period`)
- **Purpose**: Volatility measurement for dynamic stop placement
- **Usage**: Multiply ATR by a factor (e.g., 1.5x) for stop distance
- **Column**: `atr`

### 4. MACD (Moving Average Convergence Divergence)
- **Fast**: 12 (configurable via `macd_fast`)
- **Slow**: 26 (configurable via `macd_slow`)
- **Signal**: 9 (configurable via `macd_signal`)
- **Purpose**: Trend following and momentum confirmation
- **Columns**: `macd`, `macd_signal`, `macd_histogram`
- **Usage**:
  - MACD histogram > 0: Bullish momentum
  - MACD histogram < 0: Bearish momentum
  - MACD crossing signal: Trend change

### 5. Volume Indicators
- **Volume SMA Period**: 20 (configurable via `volume_sma_period`)
- **Purpose**: Volume confirmation for breakouts and continuations
- **Columns**: `volume_sma`, `volume_ratio`
- **Usage**:
  - `volume_ratio > 1.2`: Above-average volume (confirmation)
  - `volume_ratio > 1.5`: High volume (strong confirmation)
  - `volume_ratio < 0.8`: Below-average volume (caution)

## 🎯 New Trading Setups (5 Added)

### 1. EMA200 Continuation
**Type**: Trend Following

**Strategy**: 
- Looks for pullbacks to the 200 EMA in a trending market
- Confirms with RSI and volume
- Enters when price respects the EMA as support/resistance

**Parameters** (`EMA200ContinuationConfig`):
```python
ema_proximity_ticks = 4.0       # Distance to EMA200 for pullback
rsi_oversold = 35.0             # RSI threshold for long entries
rsi_overbought = 65.0           # RSI threshold for short entries
min_volume_ratio = 0.8          # Minimum volume confirmation
stop_atr_mult = 1.5             # Stop distance in ATR
target_rr = 2.0                 # Risk:Reward ratio
max_hold_bars = 48              # Maximum bars to hold (5m bars)
```

**Entry Logic**:
- **Long**: Price within 4 ticks of EMA200 from above + RSI < 35 + volume ≥ 0.8x average
- **Short**: Price within 4 ticks of EMA200 from below + RSI > 65 + volume ≥ 0.8x average

**Typical Use**: NY session trending markets, 2-3 trades per hour

---

### 2. Breakout Setup
**Type**: Momentum/Breakout

**Strategy**:
- Identifies price breaking through recent highs/lows
- Confirms with above-average volume and MACD
- Enters on validated breakout with continuation

**Parameters** (`BreakoutConfig`):
```python
lookback_bars = 20              # Bars to find recent high/low
buffer_ticks = 2.0              # Extra ticks for valid breakout
min_volume_ratio = 1.2          # Volume confirmation threshold
macd_threshold = 0.0            # MACD histogram requirement
stop_atr_mult = 1.5             # Stop placement multiplier
target_rr = 2.5                 # Higher R:R for momentum
max_hold_bars = 48
```

**Entry Logic**:
- **Long**: Close > (20-bar high + 2 ticks) + volume > 1.2x + MACD histogram > 0
- **Short**: Close < (20-bar low - 2 ticks) + volume > 1.2x + MACD histogram < 0

**Typical Use**: Range breakouts, level breaks, 3-5 signals per hour in volatile sessions

---

### 3. Reversal Setup
**Type**: Mean Reversion

**Strategy**:
- Catches reversals at Bollinger Band extremes
- Confirms with RSI divergence and rejection wicks
- Enters when price shows clear rejection of extreme

**Parameters** (`ReversalConfig`):
```python
bb_touch_ticks = 2.0            # Proximity to BB for touch
rsi_extreme_long = 30.0         # RSI oversold for long
rsi_extreme_short = 70.0        # RSI overbought for short
min_wick_ratio = 0.4            # Wick size vs bar range
stop_atr_mult = 1.5
target_rr = 2.0
max_hold_bars = 36              # Shorter hold for reversals
```

**Entry Logic**:
- **Long**: Touch lower BB + RSI < 30 + lower wick ≥ 40% of bar range
- **Short**: Touch upper BB + RSI > 70 + upper wick ≥ 40% of bar range

**Typical Use**: Range-bound markets, overextension reversals, 1-3 per hour

---

### 4. Opening Push Setup
**Type**: Session Momentum

**Strategy**:
- Captures strong directional moves in first 30 minutes
- Requires significant move from session open with volume
- Enters on momentum confirmation

**Parameters** (`OpeningPushConfig`):
```python
session_start_hour = 14         # 09:30 ET in UTC
session_start_minute = 30
push_window_minutes = 30        # First 30 minutes
min_move_ticks = 8.0            # Minimum move from open (2 points)
min_volume_ratio = 1.5          # High volume requirement
stop_atr_mult = 2.0             # Wider stop for volatility
target_rr = 2.0
max_hold_bars = 48
```

**Entry Logic**:
- **Long**: Move ≥ 8 ticks up from open + volume > 1.5x average
- **Short**: Move ≥ 8 ticks down from open + volume > 1.5x average

**Typical Use**: NY session open (9:30-10:00 ET), 0-1 trade per day

---

### 5. MOC (Market on Close) Setup
**Type**: End-of-Day Momentum

**Strategy**:
- Captures institutional flows in last hour before close
- Looks for directional runs with consistent momentum
- Enters on trend confirmation with volume

**Parameters** (`MOCConfig`):
```python
session_end_hour = 20           # 16:00 ET in UTC
session_end_minute = 0
moc_window_minutes = 60         # Last hour
min_directional_bars = 3        # Consecutive bars same direction
min_volume_ratio = 1.2
stop_atr_mult = 1.5
target_rr = 1.5                 # Conservative for EOD
max_hold_bars = 24              # Shorter hold near close
```

**Entry Logic**:
- **Long**: 3+ consecutive up bars + volume > 1.2x average
- **Short**: 3+ consecutive down bars + volume > 1.2x average

**Typical Use**: Last hour before close (15:00-16:00 ET), 0-1 trade per day

---

## 🔧 Configuration Example

```python
from src.core.detector.engine import SetupConfig, SetupFamilyConfig
from src.core.detector.indicators import IndicatorConfig

# Full configuration with all indicators and setups
config = SetupConfig(
    # Indicator settings
    indicator_cfg=IndicatorConfig(
        ema_fast=20,
        ema_slow=200,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        volume_sma_period=20,
    ),
    
    # Enable setups with custom parameters
    orb=SetupFamilyConfig(
        enabled=True,
        params={"target_rr": 2.0, "or_minutes": 30}
    ),
    
    ema200_continuation=SetupFamilyConfig(
        enabled=True,
        params={"rsi_oversold": 30.0, "rsi_overbought": 70.0}
    ),
    
    breakout=SetupFamilyConfig(
        enabled=True,
        params={"lookback_bars": 20, "min_volume_ratio": 1.5}
    ),
    
    reversal=SetupFamilyConfig(
        enabled=True,
        params={"min_wick_ratio": 0.5}
    ),
    
    opening_push=SetupFamilyConfig(
        enabled=True,
        params={"min_move_ticks": 10.0}
    ),
    
    moc=SetupFamilyConfig(
        enabled=True,
        params={"min_directional_bars": 4}
    ),
)
```

## 📈 Expected Trade Frequency

Based on typical NY session (9:30 AM - 4:00 PM ET):

| Setup Type | Expected Trades/Hour | Session Focus |
|------------|---------------------|---------------|
| ORB | 0-1 per day | Opening (9:30-10:30) |
| EMA200 Continuation | 2-3 | Trending periods |
| Breakout | 3-5 | High volatility |
| Reversal | 1-3 | Range-bound |
| Opening Push | 0-1 per day | First 30 min |
| MOC | 0-1 per day | Last hour |

**Total**: ~2-3 confirmed setups per hour during active trading

## 🎯 Usage Patterns

### High-Frequency Training Data
Enable all setups with looser parameters:
```python
config = SetupConfig(
    ema200_continuation=SetupFamilyConfig(enabled=True, params={"rsi_oversold": 40}),
    breakout=SetupFamilyConfig(enabled=True, params={"lookback_bars": 15}),
    reversal=SetupFamilyConfig(enabled=True, params={"min_wick_ratio": 0.3}),
    # ... etc
)
```

### Quality-Focused (Conservative)
Enable select setups with tighter parameters:
```python
config = SetupConfig(
    ema200_continuation=SetupFamilyConfig(enabled=True, params={
        "rsi_oversold": 30,
        "min_volume_ratio": 1.0,
    }),
    reversal=SetupFamilyConfig(enabled=True, params={
        "min_wick_ratio": 0.6,
        "rsi_extreme_long": 25,
    }),
    # Disable high-frequency setups
    breakout=SetupFamilyConfig(enabled=False),
)
```

## 🧪 Testing

Run the demo script to test all setups:
```bash
python scripts/demo_all_setups.py --days 5 --seed 42
```

This generates synthetic data and displays:
- Total setups found per type
- Win rate and average R-multiple
- Entry details with indicator values
- Summary statistics

## 📚 Documentation

- **Full Setup Details**: `docs/SETUP_ENGINE.md`
- **Code Implementation**: `src/core/detector/library.py`
- **Indicator Functions**: `src/core/detector/indicators.py`
- **Engine Configuration**: `src/core/detector/engine.py`

## 🎓 Next Steps

1. **Generate Training Data**: Run with various seeds to create diverse scenarios
2. **Tune Parameters**: Adjust knobs for your specific market conditions
3. **Backtest**: Use generated data to train ML models
4. **Analyze**: Use feature extraction to understand winning patterns
5. **Iterate**: Create custom variants by combining indicators

---

**Note**: All setups are configurable with "knobs galore" - every parameter can be adjusted without changing code!
