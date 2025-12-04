# Code Dump: 10_docs_core

## File: docs/ARCHITECTURE.md
```markdown
# FracFire Architecture

## System Overview

FracFire is a **synthetic-to-real ML platform** for futures price prediction. The architecture follows a clear separation of concerns with three main layers:

```
┌──────────────────────────────────────────────────────────────┐
│                    LAYER 0: PHYSICS ENGINE                    │
│                  (Synthetic Data Generation)                  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                LAYER 1: PATTERN RECOGNITION                   │
│              (Pre-training on Synthetic Data)                 │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  LAYER 2: PRODUCTION MODELS                   │
│               (Fine-tuning on Real Data)                      │
└──────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. **Generator = Physics, ML = Patterns**
- **Price Generator**: Understands tick mechanics, state configurations, session effects
- **ML Models**: Learn patterns, state transitions, entry/exit signals
- **Separation**: Generator stays pure (no ML inside), ML drives it externally

### 2. **Tick-Based from the Start**
- All prices stored as integer ticks (MES tick size = 0.25)
- No floating-point rounding errors
- Perfect for ML feature engineering
- Columns: `delta_ticks`, `range_ticks`, `body_ticks`, `upper_wick_ticks`, `lower_wick_ticks`

### 3. **Hierarchical State Management**
- **Day-level**: TREND_DAY, RANGE_DAY, BREAKOUT_DAY, REVERSAL_DAY, QUIET_DAY, VOLATILE_DAY
- **Hour-level**: IMPULSE, CONSOLIDATION, RETRACEMENT, CONTINUATION, REVERSAL, CHOPPY
- **Minute-level**: RALLY, BREAKDOWN, RANGING, FLAT, ZOMBIE, IMPULSIVE, BREAKOUT

### 4. **Labels Everywhere**
Every generated bar includes:
- Market state label
- Trading session
- Segment ID (for meso-scale patterns)
- Macro regime (day-level classification)

## Component Architecture

### Lab Module (`lab/`)

Research and experimentation code.

#### Generators (`lab/generators/`)

**PriceGenerator** - Main tick-based simulation engine
- Generates OHLCV bars from tick movements
- 7 standard market states + custom states
- Session effects (Asian, London, RTH, etc.)
- Day-of-week multipliers
- Segment-based state control

**FractalStateManager** - Hierarchical state management
- Day → Hour → Minute state cascade
- Proper transition probabilities
- Combined parameter calculation
- Ready for integration (currently standalone)

**Utils** - Analysis and validation
- `summarize_day()` - Comprehensive statistics
- `compare_states()` - State characteristic comparison
- `print_summary()` - Pretty-printed output

#### Visualizers (`lab/visualizers/`)

**ChartVisualizer** - Matplotlib candlestick charts
- Configurable colors, sizes, styles
- Volume subplot
- State change annotations
- Session background shading
- Quick chart helper function

### Source Module (`src/`)

Production code for ML pipeline.

#### Data (`src/data/`)

**Synthetic** - Synthetic data generators
- Archetype generation (clean patterns)
- Batch generation utilities
- Dataset versioning

**Real** - Real data adapters
- Continuous contract loader
- Data validation
- Feature alignment with synthetic

**Loaders** - Unified data interface
- Abstract base class
- Consistent API for synthetic and real data
- Automatic feature engineering

#### Models (`src/models/`)

**Base Model** - Abstract base class
- Standard interface for all models
- Save/load functionality
- Metrics tracking

**State Detector** - Market state classifier
- Pre-trained on synthetic archetypes
- Transfer learning to real data
- Confidence scores

**Price Predictor** - Price movement prediction
- Multi-horizon forecasting
- Tick-based targets
- Ensemble methods

#### Training (`src/training/`)

**Pretrain** - Pre-training on synthetic data
- Generate clean archetypes
- Train pattern recognition
- Save pre-trained weights

**Finetune** - Fine-tuning on real data
- Load pre-trained model
- Adapt to real market dynamics
- Continuous learning

**Train Utils** - Shared training utilities
- Data loaders
- Loss functions
- Callbacks and logging

#### Evaluation (`src/evaluation/`)

**Metrics** - Performance metrics
- Classification metrics (state detection)
- Regression metrics (price prediction)
- Custom trading metrics (Sharpe, max drawdown, etc.)

**Backtest** - Backtesting engine
- Simulate trading on historical data
- Position sizing
- Risk management
- Performance reporting

#### Utils (`src/utils/`)

**Config** - Configuration management
- YAML-based configs
- Environment variables
- Experiment tracking

**Logger** - Logging setup
- Structured logging
- File and console handlers
- Log rotation

## Data Flow

### Phase 1: Synthetic Archetype Generation

```python
# Generate clean patterns
gen = PriceGenerator(initial_price=5000.0, seed=42)

# Pure RALLY day
rally_day = gen.generate_day(
    start_date,
    state_sequence=[(0, MarketState.RALLY)],
    macro_regime='TREND_DAY'
)

# RANGING → BREAKOUT pattern
breakout_pattern = gen.generate_day(
    start_date,
    state_sequence=[
        (0, MarketState.RANGING),
        (60, MarketState.BREAKOUT),
        (120, MarketState.RALLY),
    ],
    segment_length=15,
)

# Save archetypes
rally_day.to_parquet('data/synthetic/archetypes/rally_day.parquet')
```

### Phase 2: Pattern Recognition Pre-training

```python
# Load synthetic archetypes
archetype_data = load_archetypes('data/synthetic/archetypes/')

# Extract features
features = extract_tick_features(archetype_data)

# Train state detector
state_detector = StateDetector()
state_detector.fit(features, labels=archetype_data['state'])

# Save pre-trained model
state_detector.save('models/state_detector_pretrained.pt')
```

### Phase 3: Fine-tuning on Real Data

```python
# Load real continuous contract data
real_data = load_continuous_contract('src/data/continuous_contract.json')

# Load pre-trained model
state_detector = StateDetector.load('models/state_detector_pretrained.pt')

# Fine-tune on real data
state_detector.finetune(real_data, epochs=10, lr=1e-4)

# Save production model
state_detector.save('models/state_detector_production.pt')
```

### Phase 4: Production Inference

```python
# Load production model
model = StateDetector.load('models/state_detector_production.pt')

# Real-time inference
current_features = extract_features(live_data)
predicted_state = model.predict(current_features)
confidence = model.predict_proba(current_features)

# Trading decision
if predicted_state == 'breakout' and confidence > 0.8:
    # Enter trade
    pass
```

## Configuration System

### Experiment Configs (`configs/`)

YAML files define experiments:

```yaml
# configs/synthetic_pretrain.yaml
experiment:
  name: "state_detector_pretrain_v1"
  description: "Pre-train state detector on synthetic archetypes"

data:
  source: "synthetic"
  archetypes:
    - rally_day
    - range_day
    - breakout_pattern
  num_samples: 10000

model:
  type: "RandomForestClassifier"
  params:
    n_estimators: 100
    max_depth: 10

training:
  validation_split: 0.2
  random_seed: 42
```

### State Configs

Market states defined in code:

```python
StateConfig(
    name="rally",
    avg_ticks_per_bar=18.0,
    up_probability=0.7,
    trend_persistence=0.8,
    volatility_multiplier=1.5,
)
```

## Testing Strategy

### Unit Tests (`tests/`)

- Test individual components in isolation
- Mock external dependencies
- Fast execution (<1s per test)

```python
def test_price_generator_tick_size():
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    bar = gen.generate_bar(datetime.now())
    
    # All prices should be multiples of 0.25
    assert bar['open'] % 0.25 == 0
    assert bar['high'] % 0.25 == 0
    assert bar['low'] % 0.25 == 0
    assert bar['close'] % 0.25 == 0
```

### Integration Tests

- Test component interactions
- Use small real datasets
- Verify end-to-end workflows

```python
def test_synthetic_to_real_pipeline():
    # Generate synthetic data
    gen = PriceGenerator(seed=42)
    synthetic_df = gen.generate_day(start_date)
    
    # Train model
    model = StateDetector()
    model.fit(synthetic_df)
    
    # Apply to real data
    real_df = load_continuous_contract()
    predictions = model.predict(real_df)
    
    # Verify predictions
    assert len(predictions) == len(real_df)
    assert all(p in MarketState for p in predictions)
```

### Backtests

- Full system tests on historical data
- Performance metrics
- Risk analysis

## Performance Considerations

### Generation Speed
- 1 day (1440 bars): ~50ms
- 30 days: ~1.5s
- Fast enough for rapid iteration

### Model Training
- Pre-training on synthetic: Minutes to hours
- Fine-tuning on real: Minutes
- Inference: Milliseconds per bar

### Storage
- Synthetic data: Parquet format (compressed)
- Models: PyTorch checkpoints
- Results: JSON + CSV for analysis

## Future Extensions

### Planned Features

1. **Markov State Driver**
   - Fit Markov model on real data
   - Generate realistic state sequences
   - Feed to PriceGenerator

2. **Advanced Models**
   - Transformer-based encoders
   - LSTM for sequence modeling
   - Ensemble methods

3. **Real-time Pipeline**
   - Live data ingestion
   - Online learning
   - Production deployment

4. **Risk Management**
   - Position sizing
   - Stop-loss optimization
   - Portfolio allocation

### Integration Points

- **Data Sources**: Easy to add new data providers
- **Models**: Plug-and-play model architecture
- **Strategies**: Modular strategy components
- **Execution**: Broker API integration

## Best Practices

1. **Version Everything**
   - Data: DVC or similar
   - Models: Git LFS or model registry
   - Configs: Git

2. **Document Experiments**
   - Use experiment tracking (MLflow, W&B)
   - Log all hyperparameters
   - Save all results

3. **Test Thoroughly**
   - Unit tests for all components
   - Integration tests for workflows
   - Backtests for strategies

4. **Keep It Simple**
   - Start with simple models
   - Add complexity only when needed
   - Measure everything

5. **Iterate Quickly**
   - Fast feedback loops
   - Small experiments
   - Learn from failures

```

---

## File: docs/GENERATOR_DEFAULTS.md
```markdown
# MES Price Generator Defaults

The **MES price generator** in `lab/generators/price_generator.py` is the canonical source
of synthetic price action used throughout this project.

## Default Physics (Global Knobs)

The global physics are defined by `PhysicsConfig` and are now tuned to match the
behavior seen in `scripts/compare_real_vs_generator.py` when compared against the
real continuous contract.

Key defaults (see `PhysicsConfig` for full details):

- `base_volatility = 2.0`
- `avg_ticks_per_bar = 8.0`
- `daily_range_mean = 120.0`
- `daily_range_std = 60.0`
- `runner_prob = 0.20`
- `runner_target_mult = 4.0`
- `macro_gravity_threshold = 5000.0`
- `macro_gravity_strength = 0.15`
- `wick_probability = 0.20`
- `wick_extension_avg = 2.0`

These values are what give the generator its "realistic" feel when you zoom out to
4H and 1D candles, including gaps and trend days.

## Comparison Harness

The script `scripts/compare_real_vs_generator.py` is the primary harness for
validating generator behavior against real data.

It:

- Loads the real continuous contract JSON via `RealDataLoader`.
- Picks a "week" and "quarter" window from the real 1m data.
- Generates synthetic 1m data using `PriceGenerator` with **PhysicsConfig defaults**
  plus any CLI overrides.
- Resamples and plots side-by-side charts for multiple timeframes, e.g.:
  - Week: `4H` and `4min`
  - Quarter: `1D` and `15min`
- Prints wick/body statistics for each timeframe so you can see how much of each
  bar is wick vs body.

Example command (from the project root, with `.venv312` active):

```bash
C:/fracfire/.venv312/Scripts/python.exe scripts/compare_real_vs_generator.py \
  --seed-week 7 \
  --seed-quarter 9 \
  --out-dir out/charts/diagnostic \
  --week-days 14 \
  --quarter-days 120
```

## Tweaking the Knobs

You can override any `PhysicsConfig` field from the CLI, without changing code:

```bash
C:/fracfire/.venv312/Scripts/python.exe scripts/compare_real_vs_generator.py \
  --base-volatility 1.8 \
  --avg-ticks-per-bar 7.0 \
  --wick-probability 0.15 \
  --week-timeframes 4H,4min \
  --quarter-timeframes 1D,15min
```

The script will start from the tuned defaults and then apply your overrides on top.
This makes the generator the single **source of synthetic truth** (or lies!) while
still letting you experiment.

For deeper structural changes (e.g., state-level behavior like RANGING vs RALLY),
see `STATE_CONFIGS` in `lab/generators/price_generator.py`.

```

---

## File: docs/SETUP_ENGINE.md
```markdown
# Setup Engine and Trade Features

This doc describes the unified API for detecting setups, evaluating trades, and producing ML-friendly features.

## Data Models

- `SetupEntry`:
  - `time`: timestamp on 1m tape (decision at 5-min boundaries by default)
  - `direction`: `long | short`
  - `kind`: `orb | level_scalp | ema20_vwap_revert | ema200_continuation`
  - `entry_price`, `stop_price`, `target_price`
  - `context`: dict for setup-specific fields (e.g., `or_high`, `or_low`)
- `SetupOutcome`:
  - `entry`: the `SetupEntry`
  - `hit_target`, `hit_stop`, `exit_time`
  - `r_multiple`, `mfe`, `mae`

## Indicators

- `IndicatorConfig` (in `src/evaluation/indicators.py`):
  - `ema_fast=20`, `ema_slow=200`
- `add_1m_indicators(df_1m, cfg)` and `add_5m_indicators(df_5m, cfg)` add columns:
  - `ema_fast`, `ema_slow`, `vwap`
- Decision anchors on 1m:
  - `DecisionAnchorConfig(modulo=5, offset=4)`
  - `mark_decision_points_1m(df_1m, cfg)` → boolean Series marking 5-min closes

## Setup Engine

- `SetupConfig` (in `src/evaluation/setup_engine.py`):
  - `indicator_cfg`: `IndicatorConfig`
  - One `SetupFamilyConfig` per setup family (enabled + `params` dict)
- `run_setups(df_1m, df_5m, cfg)` → `List[SetupOutcome]` (run all enabled families)

## ORB on 1m + 5-min anchors

- `find_opening_orb_continuations_1m(df_1m, cfg, anchor)`
- `evaluate_orb_entry_1m(df_1m, entry, cfg)`
- `summarize_orb_day_1m(df_1m, cfg, anchor)`

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

- Add setup families: level scalp, EMA/VWAP reversion, 200 EMA continuation
- Wire all families through `run_setups` and unify outputs
- Plotting overlay script for entries/stops/targets and outcomes on 1m
- Richer behavioral labels for ML (overtrading, wrong time-of-day, chasing extremes)

## Plotting Overlay

- `scripts/plot_trade_setups.py` loads a scenario folder containing `synthetic_1m.csv` and `summary.json`.
- Produces `trade_plot.png` with:
  - 1m candlesticks
  - Entry markers colored by outcome (green=target, red=stop, yellow=open)
  - Horizontal lines for stop/target around entry time
  - Legend with trade kind and R
- Intended for GUI reuse: import the plotting function or call the script.

```

---

## File: README.md
```markdown
# FracFire: Synthetic-to-Real MES Price Generator

FracFire is a research platform for generating high-fidelity synthetic futures data (specifically MES/ES) to train machine learning models. It uses a "Physics Engine" approach where price action is generated tick-by-tick based on market states, sessions, and fractal patterns.

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/fracfire.git
cd fracfire

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 🎮 Developer Playground

Run these scripts to see the system in action:

1.  **Basic Generation Demo**
    ```bash
    python scripts/demo_price_generation.py
    ```
    *Generates a full day with auto-transitions and saves charts to `out/charts/`.*

2.  **Enhanced Features Demo**
    ```bash
    python scripts/demo_enhanced_features.py
    ```
    *Shows segment-based control (e.g., 15-min blocks) and detailed statistical analysis.*

3.  **Custom States Demo**
    ```bash
    python scripts/demo_custom_states.py
    ```
    *Visualizes extreme market states like "Flash Crash", "Melt Up", and "News Spike".*

4.  **Generate Archetypes**
    ```bash
    python scripts/generate_archetypes.py
    ```
    *Creates a library of 1000+ labeled synthetic days (Rally, Range, Breakout, etc.) for ML training.*

## 🏗️ Architecture

The system is built in layers:

1.  **Physics Engine (`lab/generators/`)**:
    *   `PriceGenerator`: Tick-based simulation (0.25 tick size).
    *   `FractalStateManager`: Day/Hour/Minute hierarchical states.
    *   `ChartVisualizer`: Matplotlib-based rendering.

2.  **ML Pipeline (`src/`)**:
    *   `features/`: Feature extraction (rolling windows, etc.).
    *   `behavior/`: Markov/Regime learning.
    *   `models/`: Neural tilt models (PyTorch).
    *   `policy/`: Orchestration and consistency.

## 📚 Documentation

*   [Architecture Overview](docs/ARCHITECTURE.md)
*   [Generator Guide](docs/GENERATOR_GUIDE.md)
*   [Project Management](docs/PROJECT_MANAGEMENT.md)

## 🤝 Contributing

See `docs/PROJECT_MANAGEMENT.md` for the current roadmap and tasks.

```

---

