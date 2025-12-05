# Price Generation Methodology

## Overview

This document explains how FracFire generates synthetic price action and trades for MES (Micro E-mini S&P 500 futures), specifically for the charts generated on 2024-12-04.

## Architecture: Procedural Generative System (Sim-to-Real)

FracFire uses a **Procedural Generative System** (also known as "Sim-to-Real" environment), which is fundamentally different from ML/GAN approaches. Instead of learning patterns from data, it **models the structure and physics** of market behavior.

### Core Components

#### 1. **Physics Engine** (`src/core/generator/engine.py`)

The heart of the system is a tick-based simulation engine that generates price movements with realistic microstructure:

```python
class PriceGenerator:
    """Generates synthetic MES candles tick-by-tick with realistic physics"""
```

**Key Features:**

- **Tick Integrity**: Enforces 0.25 tick increments (MES standard)
  - Prevents invalid prices like 4500.231
  - All generated prices are valid tradeable values

- **Wick Physics**: Explicitly models rejection wicks separately from body trend
  - `wick_probability`: Controls frequency of extended wicks (default: 0.20)
  - `wick_extension_avg`: Average wick size in ticks (default: 2.0)
  - Creates realistic stop-hunt behavior

- **Volatility Control**: Multiple parameters tune market "heat"
  - `base_volatility`: Overall volatility multiplier (default: 2.0)
  - `avg_ticks_per_bar`: Base activity level (default: 8.0)
  - State-specific volatility multipliers

#### 2. **Fractal State Management** (`src/core/generator/states.py`)

Implements hierarchical market states across timeframes:

```
Day State (Trend Day, Range Day, etc.)
  └─> Hour State (Impulse, Consolidation, etc.)
      └─> Minute State (Rally, Ranging, Breakdown, etc.)
```

**Why This Matters:**
- A "Trending Day" can have a "Choppy Hour" which contains a "Rallying Minute"
- Prevents monotonic behavior (price going straight up forever)
- Creates realistic multi-timeframe market rhythm

**Markov Chain Transitions:**

The system uses probability matrices to determine state transitions:

```python
DAY_TO_HOUR_TRANSITIONS = {
    DayState.TREND_DAY: {
        HourState.IMPULSE: 0.3,
        HourState.CONTINUATION: 0.3,
        HourState.RETRACEMENT: 0.2,
        ...
    }
}
```

These probabilities capture the "rhythm" of markets better than pure LSTM prediction.

#### 3. **Session Awareness** (`src/core/generator/engine.py`)

The generator knows that different trading sessions have different characteristics:

- **Asian Session** (18:00-03:00 CT): Lower volume/volatility
- **London Session** (03:00-08:30 CT): Increasing activity
- **Regular Trading Hours (RTH)** (09:30-16:00 CT): Highest volume
- **After Hours** (16:00-18:00 CT): Declining activity

This prevents the common GAN mistake of averaging these sessions, which makes overnight too fast and RTH open too slow.

#### 4. **Daily Structure Targets**

The generator aims for realistic daily ranges:

```python
daily_range_mean: float = 120.0    # Target daily range (points)
daily_range_std: float = 60.0      # Standard deviation
runner_prob: float = 0.20          # Probability of "Runner Day"
runner_target_mult: float = 4.0    # Multiplier for runner range
```

**Runner Days**: Days with exceptional range (4x normal), capturing occasional high-volatility days.

#### 5. **Macro Gravity** (Mean Reversion at Large Scale)

Prevents price from drifting too far from starting point:

```python
macro_gravity_threshold: float = 5000.0  # Points deviation before gravity
macro_gravity_strength: float = 0.15     # Strength of pull back
```

This creates realistic multi-day oscillation around fair value.

## Trade Generation Process (12/4 Charts)

### Step 1: Generate Synthetic Price Data

For the 12/4 charts, we used `place_trades_and_plot_3day.py`:

```bash
python scripts/place_trades_and_plot_3day.py --risk 100 --start-cap 2000 --n-windows 20
```

This generates multiple 3-day windows of 1-minute synthetic data using different random seeds.

### Step 2: Detect Trading Setups

The system runs multiple setup detectors on 5-minute resampled data:

1. **ORB (Opening Range Breakout)**: Breaks above/below first 30 minutes
2. **EMA200 Continuation**: Pullback to 200 EMA with RSI confirmation
3. **Breakout**: Price breaks recent high/low with volume
4. **Reversal**: Bollinger Band extreme + RSI oversold/overbought
5. **Opening Push**: First 30-min strong directional move
6. **MOC (Market on Close)**: End-of-day directional run

Each detector uses indicator-based logic defined in `src/core/detector/library.py`.

### Step 3: Derive Stop Loss and Take Profit

For each detected setup, SL/TP are derived using context from the setup:

```python
def derive_sl_tp(entry: SetupEntry, df_5m_ind: pd.DataFrame):
    # Uses ATR for dynamic stops
    atr = df_5m_ind["atr"].loc[entry.time]
    
    # ORB: Use OR levels
    if kind == "orb":
        stop_price = or_low - buffer  # for long
        target_price = entry_price + (2.0 * atr)
    
    # EMA200: Use EMA as support/resistance
    elif kind == "ema200_cont":
        stop_price = ema200 - atr  # for long
        target_price = entry_price + (1.6 * atr)
    
    # ... (similar logic for other setups)
```

**Key Principle**: Each setup has a logical reason for its SL/TP based on the market structure that triggered it.

### Step 4: Evaluate Outcomes

Trades are evaluated using 1-minute data to determine:
- Whether target was hit first (`hit_target`)
- Whether stop was hit first (`hit_stop`)
- R-multiple: `(exit_price - entry_price) / (entry_price - stop_price)`
- MFE (Max Favorable Excursion): Best price reached
- MAE (Max Adverse Excursion): Worst price reached

### Step 5: Visualization

Charts are generated using Matplotlib with:
- 5-minute candlesticks
- Entry markers (green for wins, red for losses)
- Stop loss lines (red dashed)
- Take profit lines (green dashed)
- Cumulative P&L tracking

## Strengths of This Approach

### 1. **Microstructure Awareness**
- Valid tick sizes (0.25 increments)
- Realistic wick behavior (stop runs)
- Not just "smooth curves"

### 2. **Hierarchical Logic**
- Multiple timeframe coherence
- State transitions capture market rhythm
- Prevents unrealistic monotonic trends

### 3. **Session Awareness**
- Different characteristics for Asian/London/RTH
- Volume and volatility match session expectations

### 4. **Detector Integration**
- Infinite labeled dataset of "Setup → Outcome" pairs
- Can train ML agents on consistent, structured scenarios

### 5. **Reproducibility**
- Seed-based generation
- Same seed = same price action
- Perfect for A/B testing strategies

## Limitations and Improvement Opportunities

### Current Limitations

#### 1. **Gaussian Distribution Fallacy**

The current implementation uses normal distributions for tick movements:

```python
num_ticks = int(self.rng.normal(avg_ticks, state_config.ticks_per_bar_std))
```

**Problem**: Financial markets are **leptokurtic** (fat-tailed), not Gaussian. Real markets have:
- More frequent extreme moves than normal distribution predicts
- "Black swan" events that are statistically "impossible" under Gaussian assumptions

**Impact**: Generated data is too smooth and polite. ML models trained on this will underperform during real market spikes.

#### 2. **Fixed Transition Probabilities**

State transition probabilities are hardcoded:

```python
DAY_TO_HOUR_TRANSITIONS = {
    DayState.TREND_DAY: {HourState.IMPULSE: 0.3, ...}
}
```

**Problem**: In reality, these probabilities shift based on:
- VIX (volatility index)
- News events
- Options gamma exposure (GEX)
- Time of year (earnings season, FOMC meetings)

**Impact**: Simulation feels "robotic" - it always follows the same probability rules.

#### 3. **Lack of Order Flow Causality**

Current system: `if random_number < 0.5: price_up else: price_down`

Real markets: `if aggressive_buyers > passive_sellers: price_up`

**Impact**: Volume-Price Analysis (VPA) features like delta divergence will be artificial and won't train useful patterns.

### Available Improvements (Optional Features)

These improvements are now **available as optional configuration flags** in `PhysicsConfig`. They are disabled by default to maintain backward compatibility, but can be enabled for more realistic market simulation.

#### 1. **Fat-Tailed Distribution (Student-t) - OPTIONAL**

**Status**: ✅ **Implemented as Option**

Enable via `PhysicsConfig`:
```python
config = PhysicsConfig(
    use_fat_tails=True,      # Enable Student-t distribution
    fat_tail_df=3.0          # Degrees of freedom (lower = fatter tails)
)
```

**What it does:**
- Replaces Gaussian sampling with Student-t distribution
- Produces occasional extreme moves ("Black Swan" events)
- More realistic volatility spikes compared to smooth Gaussian

**Default**: `use_fat_tails=False` (maintains backward compatibility with Gaussian)

**How it works:**
```python
# When use_fat_tails=False (default):
tick_size = self.rng.normal(avg, std)  # Smooth, predictable

# When use_fat_tails=True:
tick_size = self.rng.standard_t(df=3) * std + avg  # Fat tails, occasional spikes
```

#### 2. **Volatility Clustering (GARCH-like) - OPTIONAL**

**Status**: ✅ **Implemented as Option**

Enable via `PhysicsConfig`:
```python
config = PhysicsConfig(
    use_volatility_clustering=True,  # Enable clustering
    volatility_persistence=0.3       # How much recent vol affects current (0-1)
)
```

**What it does:**
- Volatile bars tend to be followed by more volatile bars
- Creates realistic "clumping" of volatility
- Mimics GARCH behavior where volatility is autocorrelated

**Default**: `use_volatility_clustering=False` (maintains backward compatibility)

**How it works:**
```python
# Tracks recent volatility
self.recent_volatility = exponential_moving_average(bar_volatility)

# Applies clustering effect
clustered_vol = base_vol * (1.0 + persistence * (recent_vol - 1.0))
```

#### 3. **Implement Self-Exciting Processes (Hawkes Process) - FUTURE**

**Status**: 🔜 **Not Yet Implemented**

Model how volatility events trigger more volatility:

```python
if abs(recent_move) > threshold:
    volatility_boost = event_intensity * decay_factor
```

This captures the "cascade" effect in real markets.

#### 4. **Dynamic Transition Probabilities - FUTURE**

**Status**: 🔜 **Not Yet Implemented**

Make state transitions context-aware:

```python
# Adjust probabilities based on recent volatility
if recent_vix_proxy > 30:
    increase_probability(HourState.IMPULSIVE)
```

#### 5. **Hybrid Approach: GAN Style Transfer - FUTURE**

**Status**: 🔜 **Not Yet Implemented**

Best of both worlds:

1. Use current PriceGenerator to create clean, structured scenarios
2. Train a GAN on real MES data to learn noise and texture
3. Apply GAN as "style transfer" to add realistic micro-structure to generated data

```
Clean Generated Data + Real Market Noise/Texture = Best Training Data
```

### Usage Examples

#### Default Mode (Backward Compatible)
```python
# Uses Gaussian distribution - smooth, no extreme moves
config = PhysicsConfig()  # All optional features disabled by default
gen = PriceGenerator(physics_config=config, seed=42)
df = gen.generate_day(pd.Timestamp('2024-12-04'))
```

#### Fat Tails Mode (Realistic Spikes)
```python
# Enable Student-t distribution for occasional extreme moves
config = PhysicsConfig(
    use_fat_tails=True,
    fat_tail_df=3.0  # Lower = more extreme moves
)
gen = PriceGenerator(physics_config=config, seed=42)
df = gen.generate_day(pd.Timestamp('2024-12-04'))
```

#### Volatility Clustering Mode
```python
# Enable GARCH-like behavior
config = PhysicsConfig(
    use_volatility_clustering=True,
    volatility_persistence=0.3  # 0.3 = moderate clustering
)
gen = PriceGenerator(physics_config=config, seed=42)
df = gen.generate_day(pd.Timestamp('2024-12-04'))
```

#### Full Realism Mode (Both Features)
```python
# Enable both fat tails and clustering for maximum realism
config = PhysicsConfig(
    use_fat_tails=True,
    fat_tail_df=3.0,
    use_volatility_clustering=True,
    volatility_persistence=0.3
)
gen = PriceGenerator(physics_config=config, seed=42)
df = gen.generate_day(pd.Timestamp('2024-12-04'))
```

**Note**: These are **optional enhancements**. The default behavior (both flags set to `False`) maintains backward compatibility with the original Gaussian-based generation.

## Current State vs. Old Code

### Active Price Generation (Current - As of 12/4)

**Files:**
- `src/core/generator/engine.py` - Main physics engine
- `src/core/generator/states.py` - Fractal state management
- `src/core/generator/sampler.py` - Physics sampling
- `src/core/generator/fractal_planner.py` - Trajectory planning

**Scripts:**
- `scripts/place_trades_and_plot_3day.py` - Primary 3-day trade generation
- `scripts/place_trades_and_plot_3d.py` - 3D P&L visualization
- `scripts/generate_plotly_representatives.py` - Interactive Plotly charts

### Detector System (Trade Setup Detection)

**Files:**
- `src/core/detector/engine.py` - Detection orchestration
- `src/core/detector/library.py` - Setup detection functions
- `src/core/detector/indicators.py` - Technical indicators (RSI, ATR, MACD, etc.)
- `src/core/detector/features.py` - Feature extraction
- `src/core/detector/models.py` - Data models (SetupEntry, SetupOutcome)

### Legacy/Research Code (May Need Cleanup)

**Indicators:**
- Check for duplicate indicator implementations
- Old moving average calculations that may be superseded

**Generators:**
- Look in `lab/generators/` for experimental code
- Old state machines that may not be used

## Usage Examples

### Generate 20 3-Day Windows with Trades

```bash
python scripts/place_trades_and_plot_3day.py \
    --risk 100 \
    --start-cap 2000 \
    --n-windows 20 \
    --seed 123 \
    --out-dir out/trades_3day
```

### Generate Single Day with Custom State

```bash
python scripts/demo_custom_states.py
```

### Compare Real vs Synthetic Data

```bash
python scripts/compare_real_vs_generator.py
```

## References

- **Sim-to-Real**: Using simulation to train models for real-world deployment
- **Leptokurtic Distributions**: Fat-tailed distributions common in finance
- **Hawkes Process**: Self-exciting point processes for modeling event cascades
- **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity (volatility clustering)
- **Market Microstructure**: Study of how trades form prices at tick-level

## Conclusion

The current price generation system is sophisticated and well-suited for:
- Testing execution logic
- Training basic strategy recognition
- Creating labeled datasets for ML

To make it production-ready for ML training, the primary improvement needed is **replacing Gaussian distributions with fat-tailed distributions** (Student-t) and adding **volatility clustering** (GARCH-like behavior).

The trade detection and visualization system is robust and extensible, with clear separation between detection logic (`library.py`) and evaluation logic (`features.py`).

---

**Last Updated**: 2024-12-04  
**Version**: 1.0  
**Author**: FracFire Development Team
