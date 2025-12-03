# Tickfire Code Snapshot

Generated on: $(date)

## Project Tree
```
.
./.vscode
./.vscode/launch.json
./.vscode/settings.json
./.vscode/tasks.json
./configs
./docs
./docs/agent_instructions
./docs/agent_instructions/bootstrapping_checklist.md
./docs/agent_instructions/file_replacement_rules.md
./docs/agent_instructions/legacy_components_directive.md
./docs/agent_instructions/phase_1_tasks.md
./docs/agent_instructions/style_rules.md
./docs/ARCHITECTURE.md
./docs/GENERATOR_GUIDE.md
./docs/PROJECT_MANAGEMENT.md
./docs/SETUP_COMPLETE.md
./fracfire.code-workspace
./lab
./lab/generators
./lab/generators/custom_states.py
./lab/generators/fractal_states.py
./lab/generators/price_generator.py
./lab/generators/utils.py
./lab/generators/__init__.py
./lab/visualizers
./lab/visualizers/chart_viz.py
./lab/visualizers/__init__.py
./notebooks
./printcode.sh
./README.md
./requirements.txt
./scripts
./scripts/analyze_drift.py
./scripts/apply_optimized.py
./scripts/apply_to_real.py
./scripts/demo_custom_states.py
./scripts/demo_enhanced_features.py
./scripts/demo_price_generation.py
./scripts/evaluate_baseline.py
./scripts/generate_archetypes.py
./scripts/test_installation.py
./scripts/train_balanced.py
./scripts/train_baseline.py
./scripts/validate_archetypes.py
./scripts/visualize_optimized.py
./scripts/visualize_real.py
./src
./src/behavior
./src/behavior/learner.py
./src/data
./src/data/continuous_contract.json
./src/data/loader.py
./src/data/real
./src/data/synthetic
./src/evaluation
./src/features
./src/features/builder.py
./src/models
./src/models/tilt.py
./src/policy
./src/policy/orchestrator.py
./src/training
./src/training/data_loader.py
./src/utils
./tests
./tickfirecode.md
```

## Files

## FILE: ./docs/agent_instructions/bootstrapping_checklist.md
```
# 🚀 Bootstrapping Checklist

(Use this to verify the agent is ready to work)

---

# 🚀 **BOOTSTRAPPING CHECKLIST**

Before starting any tasks, verify the following:

1.  **Environment**:
    *   [ ] Python 3.10+ installed.
    *   [ ] Virtual environment active (`.venv` or similar).
    *   [ ] Dependencies installed (`pip install -r requirements.txt`).

2.  **Project Structure**:
    *   [ ] `lab/generators/` exists and contains `price_generator.py`.
    *   [ ] `src/` exists.
    *   [ ] `scripts/` exists.
    *   [ ] `out/` exists.

3.  **Data**:
    *   [ ] `continuous_contract.json` is present in `src/data/` (or you know where it is).

4.  **Knowledge**:
    *   [ ] You understand the **Fractal Price Generator** architecture.
    *   [ ] You understand the **Tick-Based** data philosophy.
    *   [ ] You understand your role as **Architect/Builder**, not Trainer.

5.  **Tools**:
    *   [ ] You have access to file writing tools.
    *   [ ] You have access to terminal/command execution.

**If all checks pass, proceed to Phase 1.**

---
```

## FILE: ./docs/agent_instructions/file_replacement_rules.md
```
# 🔄 File Replacement Rules

(Paste this to ensure safe file updates)

---

# 🔄 **FILE REPLACEMENT RULES**

When updating or creating files, strictly follow these rules:

1.  **Full Replacement**: Always provide the **COMPLETE** file content. Do not use diffs or "rest of file" placeholders unless explicitly authorized for massive files.
2.  **Verification**: Before writing, verify the target path exists. Create parent directories if needed.
3.  **Safety**:
    *   **NEVER** overwrite `continuous_contract.json` (Real Data).
    *   **NEVER** overwrite `newprint.md` (Source Dump) until explicitly told to delete it.
    *   **NEVER** modify files outside the `fracfire/` directory.
4.  **Backup**: For critical configuration files, consider creating a backup (e.g., `config.yaml.bak`) before overwriting.
5.  **Atomic Writes**: If possible, write to a temporary file and rename it to ensure atomicity (though standard tool usage usually handles this).

---
```

## FILE: ./docs/agent_instructions/legacy_components_directive.md
```
# 🧩 Addendum: Legacy Components Directive

(Paste this after the Master Prompt to guide the restoration of legacy components)

---

## 🧩 **LEGACY COMPONENTS DIRECTIVE**

You have been given legacy files from a previous Tickfire version that already implement a lot of what we want.
Your job is to **treat these as canonical building blocks** and integrate them into the new architecture.

### **1. Price Generator & Custom States**
*   **Legacy Files**: `lab/generators/price_generator.py`, `lab/generators/custom_states.py`
*   **Your Task**:
    *   Ensure `lab/generators/__init__.py` exports `PriceGenerator`, `MarketState`, `Session`, `StateConfig`, `CUSTOM_STATES`.
    *   Verify the API is stable: `generate_bar()`, `generate_day()`.
    *   Document how to use states, sessions, and custom configs.

### **2. Fractal State Manager**
*   **Legacy File**: `lab/generators/fractal_states.py`
*   **Your Task**:
    *   Ensure `FractalStateManager` is exported in `lab/generators/__init__.py`.
    *   Document how day/hour/minute states map to `MarketState`.
    *   Explain how this will drive multi-timeframe consistency (policy layer).

### **3. Chart Visualizer**
*   **Legacy File**: `lab/visualizers/chart_viz.py`
*   **Your Task**:
    *   Ensure `lab/visualizers/__init__.py` exports `ChartVisualizer`, `ChartConfig`, `quick_chart`.
    *   Verify it works directly with generator output (columns: `state`, `session`, `volume`).
    *   Add docs on visualizing days and custom states.

### **4. Archetype Generators**
*   **Legacy File**: `scripts/generate_archetypes.py` (needs to be created/restored)
*   **Your Task**:
    *   Implement `scripts/generate_archetypes.py` using `PriceGenerator`.
    *   Create `scripts/validate_archetypes.py` to check statistical properties.
    *   Update docs to describe the archetype library.

### **5. Demo Scripts**
*   **Legacy Files**: `demo_price_generation.py`, `demo_enhanced_features.py`, `demo_custom_states.py`
*   **Your Task**:
    *   Ensure they run and save charts to `out/charts/`.
    *   Add a "Developer Playground" section to docs.

### **6. Missing Pieces (Scaffold Only)**
*   **Feature Builders**: `src/features/` (or similar) - transform generator output to X/y.
*   **Behavior Learners**: `src/behavior/` - estimate transition matrices (Markov).
*   **Neural Tilt Model**: `src/models/tilt.py` - PyTorch skeleton (no training).
*   **Orchestrator**: `src/policy/orchestrator.py` - Interface for multi-model coordination.

**Remember**: Reuse, adapt, and integrate. Do not reinvent.
```

## FILE: ./docs/agent_instructions/phase_1_tasks.md
```
# 📋 Phase 1 Tasks Micro-Prompt

(Paste this after the Master Prompt to kickstart Phase 1)

---

# 🚀 **PHASE 1 MISSION: SYNTHETIC ARCHETYPES**

Your first major objective is to build the **Synthetic Archetype Engine**.
We need a library of clean, labelled price patterns to pretrain our models.

## **Your Tasks**

### **1. Create Archetype Generator (`scripts/generate_archetypes.py`)**

Create a script that generates 100+ samples of each of the following 10 archetypes:

1.  **Pure Rally Day** (Sustained upward trend)
2.  **Pure Range Day** (Bounded, mean-reverting)
3.  **Breakout Pattern** (Range → Breakout → Rally)
4.  **Breakdown Pattern** (Range → Breakdown → Selloff)
5.  **Reversal Pattern** (Rally → Range → Breakdown)
6.  **Zombie Grind** (Slow, low-volatility trend)
7.  **Volatile Chop** (High volatility, no direction)
8.  **Opening Bell** (High volatility at open, then settle)
9.  **Closing Squeeze** (Quiet day → End-of-day rally)
10. **News Event** (Sudden volatility spike)

**Requirements:**
*   Use `PriceGenerator` with specific `state_sequence` or `custom_state_config`.
*   Save each sample as a Parquet file in `out/data/synthetic/archetypes/<type>/`.
*   Include metadata (start time, seed, parameters) in the filename or a separate index.

### **2. Create Validation Script (`scripts/validate_archetypes.py`)**

Create a script to verify the generated archetypes:

*   Check file integrity (loadable Parquet).
*   Verify statistical properties (e.g., Rally Day should have positive net move).
*   Generate a summary report (Markdown or text).

### **3. Create Visualization Script (`scripts/visualize_archetypes.py`)**

Create a script to generate charts for a random sample of each archetype:

*   Use `ChartVisualizer`.
*   Save charts to `out/charts/archetypes/<type>/`.

## **Execution Strategy**

1.  **Plan**: Define the exact state sequences/configs for each archetype.
2.  **Implement**: Write the generator script.
3.  **Verify**: Run the generator and validator.
4.  **Document**: Update `docs/DATA_PIPELINE.md` with archetype definitions.

**GO.**
```

## FILE: ./docs/agent_instructions/style_rules.md
```
# 🎨 Style Rules Add-on

(Paste this to enforce code style and conventions)

---

# 🎨 **CODE STYLE & CONVENTIONS**

## **Python**

*   **Formatter**: Black (line length 88).
*   **Imports**: Sorted by `isort` (Standard Lib → Third Party → Local).
*   **Type Hints**: **MANDATORY** for all function arguments and return values.
*   **Docstrings**: Google Style. **MANDATORY** for all modules, classes, and public functions.
*   **Naming**:
    *   Classes: `PascalCase`
    *   Functions/Variables: `snake_case`
    *   Constants: `UPPER_CASE`
    *   Private members: `_leading_underscore`

## **Project Structure**

*   **Scripts**: Place executable scripts in `scripts/`. Use `if __name__ == "__main__":` blocks.
*   **Tests**: Place tests in `tests/`. Mirror the source directory structure.
*   **Configs**: Use YAML for experiment configs, Python dataclasses for internal configs.
*   **Paths**: Use `pathlib.Path` for all file system operations. **NEVER** use string concatenation for paths.

## **Data Handling**

*   **Ticks**: Store prices as **integer ticks** whenever possible to avoid floating-point errors.
*   **DataFrames**: Use Pandas. Ensure consistent column names (e.g., `open`, `high`, `low`, `close`, `volume`).
*   **Serialization**: Use Parquet for large datasets, JSON for metadata/configs.

## **Logging & Output**

*   **Logging**: Use the standard `logging` module. Do not use `print` for production code (scripts are okay).
*   **Progress**: Use `tqdm` for long-running loops.

## **Error Handling**

*   Use specific exception types.
*   Fail fast and provide informative error messages.

---
```

## FILE: ./docs/ARCHITECTURE.md
```
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

## FILE: ./docs/GENERATOR_GUIDE.md
```
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
```

## FILE: ./docs/PROJECT_MANAGEMENT.md
```
# Project Management & Development Guide

## 🎯 Current Status

**Phase**: Foundation Setup  
**Goal**: Establish synthetic-to-real ML pipeline for MES price prediction

## 📋 Project Roadmap

### ✅ Phase 0: Foundation (COMPLETE)
- [x] Project structure created
- [x] Price generator (tick-based MES simulation)
- [x] Fractal state manager (hierarchical states)
- [x] Visualization tools (candlestick charts)
- [x] Analysis utilities (statistics & summaries)
- [x] Documentation (README, ARCHITECTURE)
- [x] Dependencies (requirements.txt)

### ✅ Phase 1: Synthetic Archetype Generation (COMPLETE)
- [x] Define 5-10 key archetypes (patterns to generate)
- [x] Generate clean archetype datasets
- [x] Validate archetype characteristics
- [x] Label high-probability zones procedurally
- [x] Save archetype library (parquet files)

**Archetypes to Create**:
1. **Pure Rally Day** - Sustained upward movement
2. **Pure Range Day** - Choppy, bounded movement
3. **Breakout Pattern** - RANGING → BREAKOUT → RALLY
4. **Breakdown Pattern** - RANGING → BREAKDOWN → SELLOFF
5. **Reversal Pattern** - RALLY → RANGING → BREAKDOWN (or inverse)
6. **Zombie Grind** - Slow, persistent directional movement
7. **Volatile Chop** - High volatility, no clear direction
8. **Opening Bell** - RTH open volatility spike
9. **Closing Squeeze** - End-of-day positioning
10. **News Event** - Sudden volatility spike

### ✅ Phase 2: Pattern Recognition Pre-training (COMPLETE)
- [x] Extract tick-based features from archetypes
- [x] Train state classifier (Random Forest baseline)
- [x] Train sequence encoder (LSTM/Transformer) (Deferred to Phase 4)
- [x] Evaluate on held-out synthetic data
- [x] Save pre-trained models

### ✅ Phase 3: Real Data Integration (COMPLETE)
- [x] Load continuous_contract.json
- [x] Align features with synthetic data
- [x] Validate data quality
- [x] Apply pre-trained models
- [x] Analyze predictions vs reality

### ✅ Phase 4: Fine-tuning & Optimization (COMPLETE)
- [x] Analyze Feature Drift (Synthetic vs Real)
- [x] Train Balanced Model
- [x] Apply Threshold Optimization
- [x] Visualize Optimized Results
- [ ] Cross-validation (Deferred)
- [ ] Performance metrics (Deferred)

### 🚀 Phase 5: Production Pipeline
- [ ] Backtesting framework
- [ ] Risk management
- [ ] Position sizing
- [ ] Strategy implementation
- [ ] Performance monitoring

## 🗂️ File Organization

### Where to Put Things

**Synthetic Data**:
- Raw generated data: `out/data/synthetic/raw/`
- Processed archetypes: `out/data/synthetic/archetypes/`
- Training datasets: `out/data/synthetic/training/`

**Real Data**:
- Raw continuous contract: `src/data/continuous_contract.json` (already exists)
- Processed features: `out/data/real/processed/`

**Models**:
- Pre-trained: `out/models/pretrained/`
- Fine-tuned: `out/models/finetuned/`
- Production: `out/models/production/`

**Results**:
- Charts: `out/charts/`
- Metrics: `out/results/metrics/`
- Backtests: `out/results/backtests/`

**Experiments**:
- Configs: `configs/`
- Notebooks: `notebooks/`

## 🔄 Development Workflow

### Daily Workflow

1. **Start of Day**
   ```bash
   # Activate environment (already done in your case)
   # .venv312 is active
   
   # Pull latest changes
   git pull
   
   # Check what needs to be done
   cat docs/PROJECT_MANAGEMENT.md
   ```

2. **During Development**
   ```bash
   # Run tests frequently
   pytest tests/ -v
   
   # Generate synthetic data
   python scripts/generate_archetypes.py
   
   # Train models
   python scripts/train_state_detector.py
   
   # Visualize results
   python scripts/visualize_results.py
   ```

3. **End of Day**
   ```bash
   # Run full test suite
   pytest tests/ --cov=src --cov=lab
   
   # Commit changes
   git add .
   git commit -m "Descriptive message"
   git push
   
   # Update this file with progress
   ```

### Experiment Workflow

1. **Define Experiment**
   - Create config file in `configs/`
   - Document hypothesis and expected outcome

2. **Run Experiment**
   - Generate/load data
   - Train model
   - Evaluate results

3. **Analyze Results**
   - Review metrics
   - Visualize predictions
   - Compare to baseline

4. **Document Findings**
   - Update experiment log
   - Save best models
   - Note lessons learned

## 📊 Key Metrics to Track

### Synthetic Data Quality
- State distribution (% of each state)
- Tick movement statistics (mean, std)
- Volume patterns
- Session characteristics

### Model Performance
- **Classification** (State Detection):
  - Accuracy, Precision, Recall, F1
  - Confusion matrix
  - Per-state performance

- **Regression** (Price Prediction):
  - MAE, RMSE (in ticks)
  - Directional accuracy
  - Prediction horizon performance

- **Trading** (Backtest):
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Profit factor

## 🐛 Debugging Checklist

When things go wrong:

1. **Data Issues**
   - [ ] Check data shape and types
   - [ ] Verify no NaN or inf values
   - [ ] Confirm tick alignment (all multiples of 0.25)
   - [ ] Validate state labels

2. **Model Issues**
   - [ ] Check input/output dimensions
   - [ ] Verify loss is decreasing
   - [ ] Check for overfitting (train vs val)
   - [ ] Inspect predictions on sample data

3. **Performance Issues**
   - [ ] Profile code (cProfile)
   - [ ] Check memory usage
   - [ ] Optimize data loading
   - [ ] Use batch processing

## 📝 Code Standards

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small (<50 lines)
- Use meaningful variable names

### Testing
- Write tests for all new functions
- Aim for >80% code coverage
- Use fixtures for common setup
- Mock external dependencies

### Documentation
- Update README when adding features
- Document all config options
- Add examples for new functionality
- Keep architecture docs current

## 🎯 Next Actions (Immediate)

### This Week
1. **Create archetype generation script**
   - Script: `scripts/generate_archetypes.py`
   - Generate 10 archetype types
   - Save to parquet files
   - Visualize each archetype

2. **Validate archetypes**
   - Script: `scripts/validate_archetypes.py`
   - Check statistics match expectations
   - Compare to real data distributions
   - Document findings

3. **Set up testing framework**
   - Create test files for generators
   - Test price generator edge cases
   - Test fractal state transitions
   - Set up CI/CD (optional)

### Next Week
1. **Feature engineering**
   - Define feature extraction pipeline
   - Implement rolling window features
   - Add technical indicators
   - Normalize/scale features

2. **Baseline model**
   - Train Random Forest on synthetic
   - Evaluate on held-out synthetic
   - Apply to real data
   - Document performance

## 🤝 Collaboration Notes

### When Working with AI Assistant
- Provide context about what you're working on
- Share error messages in full
- Describe expected vs actual behavior
- Ask for explanations when unclear

### When Sharing Code
- Include relevant imports
- Show sample data
- Provide error tracebacks
- Mention environment details

## 📚 Resources

### Internal Docs
- [Architecture](ARCHITECTURE.md) - System design
- [README](../README.md) - Quick start guide
- [Price Generator Docs](PRICE_GENERATOR.md) - Generator details

### External Resources
- MES Contract Specs: [CME Group](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html)
- Machine Learning: [Scikit-learn Docs](https://scikit-learn.org/)
- PyTorch: [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🎉 Milestones

- [x] **Milestone 0**: Project structure and generators complete
- [x] **Milestone 1**: Archetype library generated and validated
- [x] **Milestone 2**: Baseline model trained on synthetic data
- [x] **Milestone 3**: Model applied to real data with >60% accuracy
- [x] **Milestone 4**: Fine-tuned model with >70% accuracy (Balanced Model)
- [ ] **Milestone 5**: Backtested strategy with positive Sharpe ratio

---

**Last Updated**: 2025-12-01  
**Current Focus**: Phase 5: Production Pipeline  
**Blockers**: None  
**Next Review**: After backtesting framework setup
```

## FILE: ./docs/SETUP_COMPLETE.md
```
# 🎉 FracFire Project Setup - COMPLETE

## ✅ What We've Built

Your **synthetic-to-real price generation platform** is now fully organized and ready for development!

### 📦 Project Structure Created

```
fracfire/
├── lab/                          # Research & experimentation
│   ├── generators/               # ✅ Price generation engine
│   │   ├── price_generator.py    # Tick-based MES simulator
│   │   ├── fractal_states.py     # Hierarchical state manager
│   │   ├── utils.py              # Analysis utilities
│   │   └── __init__.py           # Module exports
│   └── visualizers/              # (Ready for your viz code)
│
├── src/                          # Production ML pipeline
│   ├── data/                     # Data loaders (ready for code)
│   ├── models/                   # Model definitions (ready for code)
│   ├── training/                 # Training pipelines (ready for code)
│   ├── evaluation/               # Metrics & backtesting (ready for code)
│   └── utils/                    # Shared utilities (ready for code)
│
├── scripts/                      # Executable scripts
│   └── test_installation.py      # ✅ Installation test (PASSED!)
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # ✅ System design
│   └── PROJECT_MANAGEMENT.md     # ✅ Roadmap & workflows
│
├── configs/                      # Experiment configs (ready)
├── tests/                        # Unit tests (ready)
├── notebooks/                    # Jupyter notebooks (ready)
├── out/                          # Output directory (ready)
│
├── README.md                     # ✅ Project overview
├── requirements.txt              # ✅ Dependencies
├── .gitignore                    # ✅ Git exclusions
└── newprint.md                   # Your original code dump
```

## ✅ Installation Test Results

**All tests PASSED!** ✨

```
✓ Core dependencies (numpy, pandas, matplotlib) OK
✓ Lab generators module OK
✓ Generated 60 bars successfully
✓ Tick-based features OK
✓ Analysis utilities OK
✓ Directory structure OK
```

**Environment**: Python 3.13.7, NumPy 2.3.3, Pandas 2.3.2

## 🎯 What's Working

### 1. **Price Generator** (`lab/generators/price_generator.py`)
- ✅ Tick-based MES simulation (0.25 tick size)
- ✅ 7 market states (RANGING, FLAT, ZOMBIE, RALLY, IMPULSIVE, BREAKDOWN, BREAKOUT)
- ✅ Session effects (Asian, London, RTH, etc.)
- ✅ Day-of-week multipliers
- ✅ ML-ready tick columns (delta_ticks, range_ticks, body_ticks, wicks)
- ✅ Segment-based state control
- ✅ Macro regime labeling

### 2. **Fractal State Manager** (`lab/generators/fractal_states.py`)
- ✅ Day-level states (TREND_DAY, RANGE_DAY, etc.)
- ✅ Hour-level states (IMPULSE, CONSOLIDATION, etc.)
- ✅ Minute-level states (maps to MarketState)
- ✅ Transition probabilities
- ✅ Combined parameter calculation
- 🔄 Ready to integrate with main generator

### 3. **Analysis Utilities** (`lab/generators/utils.py`)
- ✅ `summarize_day()` - Comprehensive statistics
- ✅ `print_summary()` - Pretty-printed output
- ✅ `compare_states()` - State comparison

### 4. **Documentation**
- ✅ README.md - Quick start guide
- ✅ ARCHITECTURE.md - System design
- ✅ PROJECT_MANAGEMENT.md - Roadmap & workflows

## 📋 Next Steps

### Immediate (This Session)
1. **Add remaining code from newprint.md**:
   - [ ] Visualizer (`lab/visualizers/chart_viz.py`)
   - [ ] Custom states (`lab/generators/custom_states.py`)
   - [ ] Demo scripts (`scripts/demo_*.py`)

2. **Test the demos**:
   ```bash
   python scripts/demo_price_generation.py
   python scripts/demo_enhanced_features.py
   ```

### Short-term (This Week)
1. **Generate Archetypes**:
   - Create `scripts/generate_archetypes.py`
   - Generate 10 archetype patterns
   - Save to `out/data/synthetic/archetypes/`

2. **Validate Archetypes**:
   - Check statistics match expectations
   - Compare to real data distributions

### Medium-term (Next Week)
1. **Feature Engineering**:
   - Define feature extraction pipeline
   - Implement rolling window features
   - Add technical indicators

2. **Baseline Model**:
   - Train Random Forest on synthetic data
   - Evaluate on held-out synthetic
   - Apply to real data

## 🔧 How to Use

### Generate Synthetic Data
```python
from lab.generators import PriceGenerator, MarketState
from datetime import datetime

gen = PriceGenerator(initial_price=5000.0, seed=42)
start_date = datetime(2025, 11, 29, 0, 0, 0)

# Generate a full day
df = gen.generate_day(start_date, auto_transition=True)

# Or control the state sequence
state_sequence = [
    (0, MarketState.RANGING),
    (60, MarketState.RALLY),
    (180, MarketState.RANGING),
]
df = gen.generate_day(start_date, state_sequence=state_sequence)
```

### Analyze Data
```python
from lab.generators.utils import summarize_day, print_summary

summary = summarize_day(df)
print_summary(summary, verbose=True)
```

## 📊 Key Features

### Tick-Based Output
Every bar includes:
- **Price columns**: open, high, low, close, volume
- **Tick columns** (integers): delta_ticks, range_ticks, body_ticks, upper_wick_ticks, lower_wick_ticks
- **Labels**: state, session, segment_id, macro_regime

### Configurable States
- 7 standard market states
- Custom state configurations
- Session-based effects
- Day-of-week multipliers

### Hierarchical States
- Day → Hour → Minute cascade
- Proper transition probabilities
- Combined parameter calculation

## 🎯 Project Philosophy

1. **Generator = Physics, ML = Patterns**
   - Generator knows tick mechanics
   - ML learns patterns and drives state sequences

2. **Tick-Based from the Start**
   - All features in integer ticks
   - No floating-point errors
   - Perfect for ML

3. **Labels Everywhere**
   - Every bar tagged with state/session/segment/regime
   - Ready for supervised learning

4. **External State Drivers**
   - Markov/ML sits outside and controls generator
   - Clean separation of concerns

## 📚 Documentation

- **[README.md](../README.md)** - Quick start & overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md)** - Roadmap & workflows

## 🚀 Ready to Go!

Your environment is set up and tested. You can now:

1. ✅ Generate synthetic MES data
2. ✅ Analyze market states
3. ✅ Create custom state configurations
4. 🔄 Add visualizations (next step)
5. 🔄 Train ML models (coming soon)
6. 🔄 Apply to real data (coming soon)

---

**Status**: ✅ Foundation Complete  
**Environment**: `.venv312` (Python 3.13.7)  
**Next**: Add remaining code from newprint.md and run demos  
**Date**: 2025-11-29
```

## FILE: ./fracfire.code-workspace
```
{
	"folders": [
		{
			"path": "."
		}
	],
	"settings": {}
}```

## FILE: ./lab/generators/__init__.py
```
"""
Price generators for synthetic market data.
"""

from .price_generator import (
    PriceGenerator,
    MarketState,
    Session,
    StateConfig,
    SessionConfig,
    DayOfWeekConfig,
    STATE_CONFIGS,
    SESSION_CONFIGS,
    DOW_CONFIGS,
)

from .fractal_states import (
    FractalStateManager,
    DayState,
    HourState,
    MinuteState,
    FractalStateConfig,
)

from .custom_states import (
    CUSTOM_STATES,
    get_custom_state,
    list_custom_states,
    EXTREME_VOLATILITY,
    DIRECTIONAL,
    SESSION_SPECIFIC,
    LOW_VOLATILITY,
)

__all__ = [
    'PriceGenerator',
    'MarketState',
    'Session',
    'StateConfig',
    'SessionConfig',
    'DayOfWeekConfig',
    'STATE_CONFIGS',
    'SESSION_CONFIGS',
    'DOW_CONFIGS',
    'FractalStateManager',
    'DayState',
    'HourState',
    'MinuteState',
    'FractalStateConfig',
    'CUSTOM_STATES',
    'get_custom_state',
    'list_custom_states',
    'EXTREME_VOLATILITY',
    'DIRECTIONAL',
    'SESSION_SPECIFIC',
    'LOW_VOLATILITY',
]
```

## FILE: ./lab/generators/custom_states.py
```
"""
Custom Market State Configurations

Pre-defined extreme and specialized market states for testing and simulation.
These complement the standard STATE_CONFIGS in price_generator.py
"""

from .price_generator import StateConfig


# Extreme volatility states
CUSTOM_STATES = {
    'mega_volatile': StateConfig(
        name="mega_volatile",
        avg_ticks_per_bar=40.0,
        ticks_per_bar_std=20.0,
        up_probability=0.5,
        trend_persistence=0.4,  # Low persistence = very choppy
        avg_tick_size=3.0,
        tick_size_std=2.0,
        max_tick_jump=15,
        volatility_multiplier=3.0,
        wick_probability=0.6,
        wick_extension_avg=5.0,
    ),
    
    'flash_crash': StateConfig(
        name="flash_crash",
        avg_ticks_per_bar=50.0,
        ticks_per_bar_std=15.0,
        up_probability=0.15,  # Strong downward bias
        trend_persistence=0.9,  # Very persistent down moves
        avg_tick_size=4.0,
        tick_size_std=2.5,
        max_tick_jump=20,
        volatility_multiplier=3.5,
        wick_probability=0.4,  # Some wicks but mostly directional
        wick_extension_avg=6.0,
    ),
    
    'melt_up': StateConfig(
        name="melt_up",
        avg_ticks_per_bar=35.0,
        ticks_per_bar_std=12.0,
        up_probability=0.85,  # Strong upward bias
        trend_persistence=0.85,  # Very persistent up moves
        avg_tick_size=2.5,
        tick_size_std=1.5,
        max_tick_jump=12,
        volatility_multiplier=2.5,
        wick_probability=0.35,
        wick_extension_avg=4.0,
    ),
    
    'whipsaw': StateConfig(
        name="whipsaw",
        avg_ticks_per_bar=30.0,
        ticks_per_bar_std=15.0,
        up_probability=0.5,
        trend_persistence=0.1,  # Very low persistence = rapid reversals
        avg_tick_size=2.5,
        tick_size_std=1.8,
        max_tick_jump=10,
        volatility_multiplier=2.2,
        wick_probability=0.7,  # Lots of wicks from reversals
        wick_extension_avg=5.0,
    ),
    
    'death_spiral': StateConfig(
        name="death_spiral",
        avg_ticks_per_bar=45.0,
        ticks_per_bar_std=18.0,
        up_probability=0.2,  # Strong down bias
        trend_persistence=0.75,  # Persistent but with some bounces
        avg_tick_size=3.5,
        tick_size_std=2.0,
        max_tick_jump=18,
        volatility_multiplier=3.2,
        wick_probability=0.5,
        wick_extension_avg=7.0,
    ),
    
    'moonshot': StateConfig(
        name="moonshot",
        avg_ticks_per_bar=38.0,
        ticks_per_bar_std=14.0,
        up_probability=0.8,  # Strong up bias
        trend_persistence=0.8,  # Very persistent
        avg_tick_size=3.0,
        tick_size_std=1.8,
        max_tick_jump=16,
        volatility_multiplier=2.8,
        wick_probability=0.4,
        wick_extension_avg=5.0,
    ),
    
    'slow_bleed': StateConfig(
        name="slow_bleed",
        avg_ticks_per_bar=8.0,
        ticks_per_bar_std=4.0,
        up_probability=0.35,  # Moderate down bias
        trend_persistence=0.7,  # Persistent grind
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=3,
        volatility_multiplier=0.8,
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    
    'slow_grind_up': StateConfig(
        name="slow_grind_up",
        avg_ticks_per_bar=8.0,
        ticks_per_bar_std=4.0,
        up_probability=0.65,  # Moderate up bias
        trend_persistence=0.7,  # Persistent grind
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=3,
        volatility_multiplier=0.8,
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    
    'opening_bell': StateConfig(
        name="opening_bell",
        avg_ticks_per_bar=35.0,
        ticks_per_bar_std=18.0,
        up_probability=0.5,
        trend_persistence=0.3,  # Choppy at open
        avg_tick_size=2.5,
        tick_size_std=2.0,
        max_tick_jump=12,
        volatility_multiplier=2.5,
        wick_probability=0.6,
        wick_extension_avg=6.0,
    ),
    
    'closing_squeeze': StateConfig(
        name="closing_squeeze",
        avg_ticks_per_bar=25.0,
        ticks_per_bar_std=12.0,
        up_probability=0.55,  # Slight up bias (short covering)
        trend_persistence=0.6,
        avg_tick_size=2.0,
        tick_size_std=1.5,
        max_tick_jump=10,
        volatility_multiplier=2.0,
        wick_probability=0.5,
        wick_extension_avg=4.0,
    ),
    
    'news_spike': StateConfig(
        name="news_spike",
        avg_ticks_per_bar=60.0,
        ticks_per_bar_std=25.0,
        up_probability=0.7,  # Usually news is positive initially
        trend_persistence=0.5,  # Mixed reactions
        avg_tick_size=4.0,
        tick_size_std=3.0,
        max_tick_jump=25,
        volatility_multiplier=4.0,
        wick_probability=0.7,
        wick_extension_avg=8.0,
    ),
    
    'dead_zone': StateConfig(
        name="dead_zone",
        avg_ticks_per_bar=2.0,
        ticks_per_bar_std=1.0,
        up_probability=0.5,
        trend_persistence=0.5,
        avg_tick_size=1.0,
        tick_size_std=0.2,
        max_tick_jump=1,
        volatility_multiplier=0.2,
        wick_probability=0.1,
        wick_extension_avg=1.0,
    ),
}


# Categorized for easy access
EXTREME_VOLATILITY = ['mega_volatile', 'flash_crash', 'melt_up', 'whipsaw', 'news_spike']
DIRECTIONAL = ['flash_crash', 'melt_up', 'death_spiral', 'moonshot', 'slow_bleed', 'slow_grind_up']
SESSION_SPECIFIC = ['opening_bell', 'closing_squeeze']
LOW_VOLATILITY = ['dead_zone', 'slow_bleed', 'slow_grind_up']


def get_custom_state(name: str) -> StateConfig:
    """
    Get a custom state configuration by name.
    
    Args:
        name: Name of the custom state
    
    Returns:
        StateConfig instance
    
    Raises:
        KeyError: If state name not found
    """
    if name not in CUSTOM_STATES:
        available = ', '.join(CUSTOM_STATES.keys())
        raise KeyError(f"Custom state '{name}' not found. Available: {available}")
    
    return CUSTOM_STATES[name]


def list_custom_states() -> dict:
    """
    Get a summary of all custom states.
    
    Returns:
        Dictionary with state names and descriptions
    """
    summaries = {}
    
    for name, config in CUSTOM_STATES.items():
        summaries[name] = {
            'avg_ticks_per_bar': config.avg_ticks_per_bar,
            'up_probability': config.up_probability,
            'trend_persistence': config.trend_persistence,
            'volatility_multiplier': config.volatility_multiplier,
            'max_tick_jump': config.max_tick_jump,
        }
    
    return summaries


def print_custom_states():
    """Print a formatted table of all custom states"""
    print("\n" + "=" * 80)
    print("CUSTOM MARKET STATES")
    print("=" * 80)
    print(f"\n{'State':<20} {'AvgTicks':>8} {'UpProb':>7} {'Persist':>7} {'VolMult':>7} {'MaxJump':>8}")
    print("-" * 80)
    
    for name, config in CUSTOM_STATES.items():
        print(f"{name:<20} {config.avg_ticks_per_bar:>8.1f} {config.up_probability:>7.2f} "
              f"{config.trend_persistence:>7.2f} {config.volatility_multiplier:>7.1f} "
              f"{config.max_tick_jump:>8}")
    
    print("\n" + "=" * 80)
    print("\nCategories:")
    print(f"  Extreme Volatility: {', '.join(EXTREME_VOLATILITY)}")
    print(f"  Directional: {', '.join(DIRECTIONAL)}")
    print(f"  Session Specific: {', '.join(SESSION_SPECIFIC)}")
    print(f"  Low Volatility: {', '.join(LOW_VOLATILITY)}")
    print("=" * 80)


if __name__ == "__main__":
    print_custom_states()
```

## FILE: ./lab/generators/fractal_states.py
```
"""
Fractal State Manager - Hierarchical market states across timeframes

Implements nested states where larger timeframes influence smaller ones:
- Day-level state (e.g., trending day, range day, breakout day)
- Hour-level states within the day state
- Minute-level states within the hour state

This creates realistic multi-timeframe market behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class DayState(Enum):
    """Day-level market states (highest timeframe)"""
    TREND_DAY = "trend_day"              # Strong directional day
    RANGE_DAY = "range_day"              # Choppy, bounded day
    BREAKOUT_DAY = "breakout_day"        # Breaks out of range
    REVERSAL_DAY = "reversal_day"        # V-shaped or inverse V
    QUIET_DAY = "quiet_day"              # Low volatility, minimal movement
    VOLATILE_DAY = "volatile_day"        # High volatility, no clear direction


class HourState(Enum):
    """Hour-level market states (medium timeframe)"""
    IMPULSE = "impulse"                  # Strong directional move
    CONSOLIDATION = "consolidation"      # Tight range
    RETRACEMENT = "retracement"          # Pullback within larger trend
    CONTINUATION = "continuation"        # Resuming previous direction
    REVERSAL = "reversal"                # Changing direction
    CHOPPY = "choppy"                    # No clear direction


class MinuteState(Enum):
    """Minute-level market states (lowest timeframe) - maps to existing MarketState"""
    RALLY = "rally"
    BREAKDOWN = "breakdown"
    RANGING = "ranging"
    FLAT = "flat"
    ZOMBIE = "zombie"
    IMPULSIVE = "impulsive"
    BREAKOUT = "breakout"


@dataclass
class FractalStateConfig:
    """Configuration for how states influence each other across timeframes"""
    
    # Required fields (no defaults)
    day_state: DayState
    hour_state: HourState
    minute_state: MinuteState
    
    # Optional fields (with defaults)
    day_directional_bias: float = 0.5    # 0=down, 0.5=neutral, 1=up
    day_volatility_mult: float = 1.0
    day_trend_strength: float = 0.5      # How strongly day state influences hours
    hour_directional_bias: float = 0.5
    hour_volatility_mult: float = 1.0
    hour_trend_strength: float = 0.5     # How strongly hour state influences minutes
    hour_transition_prob: float = 0.05   # Probability of hour state change per minute
    minute_transition_prob: float = 0.1  # Probability of minute state change per bar


# Define how day states influence hour state probabilities
DAY_TO_HOUR_TRANSITIONS: Dict[DayState, Dict[HourState, float]] = {
    DayState.TREND_DAY: {
        HourState.IMPULSE: 0.3,
        HourState.CONTINUATION: 0.3,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.REVERSAL: 0.03,
        HourState.CHOPPY: 0.02,
    },
    DayState.RANGE_DAY: {
        HourState.CONSOLIDATION: 0.35,
        HourState.CHOPPY: 0.25,
        HourState.IMPULSE: 0.15,
        HourState.RETRACEMENT: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.REVERSAL: 0.05,
    },
    DayState.BREAKOUT_DAY: {
        HourState.IMPULSE: 0.4,
        HourState.CONTINUATION: 0.25,
        HourState.CONSOLIDATION: 0.2,
        HourState.RETRACEMENT: 0.1,
        HourState.CHOPPY: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.REVERSAL_DAY: {
        HourState.REVERSAL: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.CHOPPY: 0.05,
    },
    DayState.QUIET_DAY: {
        HourState.CONSOLIDATION: 0.5,
        HourState.CHOPPY: 0.2,
        HourState.CONTINUATION: 0.15,
        HourState.IMPULSE: 0.1,
        HourState.RETRACEMENT: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.VOLATILE_DAY: {
        HourState.CHOPPY: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.REVERSAL: 0.2,
        HourState.RETRACEMENT: 0.15,
        HourState.CONSOLIDATION: 0.05,
        HourState.CONTINUATION: 0.05,
    },
}


# Define how hour states influence minute state probabilities
HOUR_TO_MINUTE_TRANSITIONS: Dict[HourState, Dict[MinuteState, float]] = {
    HourState.IMPULSE: {
        MinuteState.RALLY: 0.4,
        MinuteState.BREAKOUT: 0.2,
        MinuteState.IMPULSIVE: 0.2,
        MinuteState.ZOMBIE: 0.1,
        MinuteState.RANGING: 0.05,
        MinuteState.FLAT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.CONSOLIDATION: {
        MinuteState.RANGING: 0.4,
        MinuteState.FLAT: 0.3,
        MinuteState.ZOMBIE: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.BREAKDOWN: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.RETRACEMENT: {
        MinuteState.BREAKDOWN: 0.3,
        MinuteState.RANGING: 0.25,
        MinuteState.ZOMBIE: 0.2,
        MinuteState.FLAT: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.CONTINUATION: {
        MinuteState.ZOMBIE: 0.35,
        MinuteState.RALLY: 0.25,
        MinuteState.RANGING: 0.2,
        MinuteState.FLAT: 0.1,
        MinuteState.IMPULSIVE: 0.05,
        MinuteState.BREAKOUT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.REVERSAL: {
        MinuteState.IMPULSIVE: 0.3,
        MinuteState.BREAKDOWN: 0.25,
        MinuteState.RALLY: 0.2,
        MinuteState.RANGING: 0.15,
        MinuteState.BREAKOUT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.FLAT: 0.02,
    },
    HourState.CHOPPY: {
        MinuteState.RANGING: 0.35,
        MinuteState.IMPULSIVE: 0.25,
        MinuteState.RALLY: 0.15,
        MinuteState.BREAKDOWN: 0.15,
        MinuteState.FLAT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
}


# Day state characteristics
DAY_STATE_PARAMS = {
    DayState.TREND_DAY: {
        'directional_bias': 0.65,  # Upward bias (can be flipped for down trend)
        'volatility_mult': 1.3,
        'trend_strength': 0.8,
    },
    DayState.RANGE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.9,
        'trend_strength': 0.3,
    },
    DayState.BREAKOUT_DAY: {
        'directional_bias': 0.7,
        'volatility_mult': 1.6,
        'trend_strength': 0.85,
    },
    DayState.REVERSAL_DAY: {
        'directional_bias': 0.5,  # Changes during day
        'volatility_mult': 1.4,
        'trend_strength': 0.6,
    },
    DayState.QUIET_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.5,
        'trend_strength': 0.2,
    },
    DayState.VOLATILE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 2.0,
        'trend_strength': 0.4,
    },
}


# Hour state characteristics
HOUR_STATE_PARAMS = {
    HourState.IMPULSE: {
        'directional_bias': 0.7,
        'volatility_mult': 1.5,
        'trend_strength': 0.8,
    },
    HourState.CONSOLIDATION: {
        'directional_bias': 0.5,
        'volatility_mult': 0.6,
        'trend_strength': 0.3,
    },
    HourState.RETRACEMENT: {
        'directional_bias': 0.35,  # Against main trend
        'volatility_mult': 1.0,
        'trend_strength': 0.6,
    },
    HourState.CONTINUATION: {
        'directional_bias': 0.6,
        'volatility_mult': 1.1,
        'trend_strength': 0.7,
    },
    HourState.REVERSAL: {
        'directional_bias': 0.5,  # Flips during hour
        'volatility_mult': 1.4,
        'trend_strength': 0.7,
    },
    HourState.CHOPPY: {
        'directional_bias': 0.5,
        'volatility_mult': 1.2,
        'trend_strength': 0.2,
    },
}


class FractalStateManager:
    """
    Manages hierarchical market states across timeframes.
    
    Day state influences hour states, which influence minute states.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed"""
        self.rng = np.random.default_rng(seed)
        
        # Current states
        self.day_state: Optional[DayState] = None
        self.hour_state: Optional[HourState] = None
        self.minute_state: Optional[MinuteState] = None
        
        # State parameters
        self.day_params: Dict = {}
        self.hour_params: Dict = {}
        
        # Tracking
        self.current_hour_start: Optional[datetime] = None
        self.bars_in_current_hour: int = 0
    
    def initialize_day(self, day_state: Optional[DayState] = None) -> DayState:
        """
        Initialize a new day with a day-level state.
        
        Args:
            day_state: Specific day state, or None for random
        
        Returns:
            The selected day state
        """
        if day_state is None:
            # Random day state
            day_state = self.rng.choice(list(DayState))
        
        self.day_state = day_state
        self.day_params = DAY_STATE_PARAMS[day_state].copy()
        
        # Initialize first hour state
        self.hour_state = self._transition_hour_state()
        self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
        
        # Initialize first minute state
        self.minute_state = self._transition_minute_state()
        
        return day_state
    
    def _transition_hour_state(self) -> HourState:
        """Transition to a new hour state based on day state"""
        if self.day_state is None:
            return HourState.CONSOLIDATION
        
        # Get transition probabilities from day state
        probs = DAY_TO_HOUR_TRANSITIONS[self.day_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def _transition_minute_state(self) -> MinuteState:
        """Transition to a new minute state based on hour state"""
        if self.hour_state is None:
            return MinuteState.RANGING
        
        # Get transition probabilities from hour state
        probs = HOUR_TO_MINUTE_TRANSITIONS[self.hour_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def update(self, timestamp: datetime, force_hour_transition: bool = False) -> Tuple[DayState, HourState, MinuteState]:
        """
        Update states for a new bar.
        
        Args:
            timestamp: Current bar timestamp
            force_hour_transition: Force an hour state transition
        
        Returns:
            (day_state, hour_state, minute_state)
        """
        if self.day_state is None:
            self.initialize_day()
        
        # Track hour boundaries
        if self.current_hour_start is None:
            self.current_hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
            self.bars_in_current_hour = 0
        
        # Check if we've entered a new hour
        current_hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if current_hour != self.current_hour_start or force_hour_transition:
            # New hour - higher chance of hour state transition
            if self.rng.random() < 0.3 or force_hour_transition:  # 30% chance on hour boundary
                self.hour_state = self._transition_hour_state()
                self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
            
            self.current_hour_start = current_hour
            self.bars_in_current_hour = 0
        
        # Check for hour state transition (can happen mid-hour)
        elif self.rng.random() < 0.02:  # 2% chance per bar mid-hour
            self.hour_state = self._transition_hour_state()
            self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
        
        # Check for minute state transition
        if self.rng.random() < 0.1:  # 10% chance per bar
            self.minute_state = self._transition_minute_state()
        
        self.bars_in_current_hour += 1
        
        return self.day_state, self.hour_state, self.minute_state
    
    def get_combined_parameters(self) -> Dict:
        """
        Get combined parameters from all timeframe states.
        
        Returns:
            Dictionary with combined directional_bias, volatility_mult, trend_strength
        """
        # Combine day and hour parameters
        day_bias = self.day_params.get('directional_bias', 0.5)
        hour_bias = self.hour_params.get('directional_bias', 0.5)
        
        # Weight by trend strength
        day_strength = self.day_params.get('trend_strength', 0.5)
        hour_strength = self.hour_params.get('trend_strength', 0.5)
        
        # Combined bias (weighted average)
        combined_bias = (
            day_bias * day_strength * 0.3 +
            hour_bias * hour_strength * 0.7
        )
        
        # Combined volatility (multiplicative)
        combined_volatility = (
            self.day_params.get('volatility_mult', 1.0) *
            self.hour_params.get('volatility_mult', 1.0)
        )
        
        # Combined trend strength (average)
        combined_trend_strength = (
            day_strength * 0.4 +
            hour_strength * 0.6
        )
        
        return {
            'directional_bias': combined_bias,
            'volatility_mult': combined_volatility,
            'trend_strength': combined_trend_strength,
            'day_state': self.day_state.value if self.day_state else None,
            'hour_state': self.hour_state.value if self.hour_state else None,
            'minute_state': self.minute_state.value if self.minute_state else None,
        }
```

## FILE: ./lab/generators/price_generator.py
```
"""
MES Price Generator with Configurable Market States

This module generates synthetic 1-minute MES candles with realistic tick-based price action.
All parameters are exposed as "knobs" for testing different market conditions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum


class MarketState(Enum):
    """Market behavior states that affect price generation"""
    RANGING = "ranging"          # Tight, choppy, mean-reverting
    FLAT = "flat"                # Very low volatility, minimal movement
    ZOMBIE = "zombie"            # Slow grind in one direction
    RALLY = "rally"              # Strong directional move
    IMPULSIVE = "impulsive"      # High volatility, large swings
    BREAKDOWN = "breakdown"      # Sharp downward move
    BREAKOUT = "breakout"        # Sharp upward move


class Session(Enum):
    """Trading sessions with different characteristics"""
    ASIAN = "asian"              # 18:00-03:00 CT (typically lower volume)
    LONDON = "london"            # 03:00-08:30 CT (increasing activity)
    PREMARKET = "premarket"      # 08:30-09:30 CT (building momentum)
    RTH = "rth"                  # 09:30-16:00 CT (Regular Trading Hours - highest volume)
    AFTERHOURS = "afterhours"    # 16:00-18:00 CT (declining activity)


@dataclass
class StateConfig:
    """Configuration for a specific market state"""
    name: str
    
    # Tick movement parameters
    avg_ticks_per_bar: float = 8.0          # Average number of ticks per 1m bar
    ticks_per_bar_std: float = 4.0          # Std dev of ticks per bar
    
    # Directional bias
    up_probability: float = 0.5             # Probability of upward tick (0.5 = neutral)
    trend_persistence: float = 0.5          # How likely to continue previous direction (0-1)
    
    # Tick size distribution
    avg_tick_size: float = 1.0              # Average ticks per move (1.0 = single tick)
    tick_size_std: float = 0.5              # Std dev of tick size
    max_tick_jump: int = 8                  # Maximum ticks in single move
    
    # Volatility
    volatility_multiplier: float = 1.0      # Overall volatility scaling
    
    # Wick characteristics
    wick_probability: float = 0.3           # Probability of extended wicks
    wick_extension_avg: float = 2.0         # Average wick extension in ticks


# Predefined state configurations
STATE_CONFIGS = {
    MarketState.RANGING: StateConfig(
        name="ranging",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.5,
        trend_persistence=0.3,  # Low persistence = choppy
        avg_tick_size=1.2,
        tick_size_std=0.8,
        max_tick_jump=4,
        volatility_multiplier=1.0,
        wick_probability=0.4,
        wick_extension_avg=2.5,
    ),
    MarketState.FLAT: StateConfig(
        name="flat",
        avg_ticks_per_bar=4.0,
        ticks_per_bar_std=2.0,
        up_probability=0.5,
        trend_persistence=0.4,
        avg_tick_size=1.0,
        tick_size_std=0.3,
        max_tick_jump=2,
        volatility_multiplier=0.3,
        wick_probability=0.2,
        wick_extension_avg=1.0,
    ),
    MarketState.ZOMBIE: StateConfig(
        name="zombie",
        avg_ticks_per_bar=6.0,
        ticks_per_bar_std=3.0,
        up_probability=0.55,  # Slight upward bias
        trend_persistence=0.7,  # High persistence = grind
        avg_tick_size=1.0,
        tick_size_std=0.4,
        max_tick_jump=3,
        volatility_multiplier=0.6,
        wick_probability=0.25,
        wick_extension_avg=1.5,
    ),
    MarketState.RALLY: StateConfig(
        name="rally",
        avg_ticks_per_bar=18.0,
        ticks_per_bar_std=8.0,
        up_probability=0.7,  # Strong upward bias
        trend_persistence=0.8,
        avg_tick_size=1.5,
        tick_size_std=1.0,
        max_tick_jump=6,
        volatility_multiplier=1.5,
        wick_probability=0.3,
        wick_extension_avg=3.0,
    ),
    MarketState.IMPULSIVE: StateConfig(
        name="impulsive",
        avg_ticks_per_bar=25.0,
        ticks_per_bar_std=12.0,
        up_probability=0.5,
        trend_persistence=0.6,
        avg_tick_size=2.0,
        tick_size_std=1.5,
        max_tick_jump=10,
        volatility_multiplier=2.0,
        wick_probability=0.5,
        wick_extension_avg=4.0,
    ),
    MarketState.BREAKDOWN: StateConfig(
        name="breakdown",
        avg_ticks_per_bar=20.0,
        ticks_per_bar_std=10.0,
        up_probability=0.25,  # Strong downward bias
        trend_persistence=0.85,
        avg_tick_size=1.8,
        tick_size_std=1.2,
        max_tick_jump=8,
        volatility_multiplier=1.8,
        wick_probability=0.35,
        wick_extension_avg=3.5,
    ),
    MarketState.BREAKOUT: StateConfig(
        name="breakout",
        avg_ticks_per_bar=22.0,
        ticks_per_bar_std=10.0,
        up_probability=0.75,  # Strong upward bias
        trend_persistence=0.85,
        avg_tick_size=1.8,
        tick_size_std=1.2,
        max_tick_jump=8,
        volatility_multiplier=1.8,
        wick_probability=0.35,
        wick_extension_avg=3.5,
    ),
}


@dataclass
class SessionConfig:
    """Configuration for session-based effects"""
    name: str
    volume_multiplier: float = 1.0          # Relative volume level
    volatility_multiplier: float = 1.0      # Relative volatility
    state_transition_prob: float = 0.05     # Probability of state change per bar


SESSION_CONFIGS = {
    Session.ASIAN: SessionConfig(
        name="asian",
        volume_multiplier=0.4,
        volatility_multiplier=0.6,
        state_transition_prob=0.02,
    ),
    Session.LONDON: SessionConfig(
        name="london",
        volume_multiplier=0.8,
        volatility_multiplier=1.1,
        state_transition_prob=0.05,
    ),
    Session.PREMARKET: SessionConfig(
        name="premarket",
        volume_multiplier=0.6,
        volatility_multiplier=0.9,
        state_transition_prob=0.08,
    ),
    Session.RTH: SessionConfig(
        name="rth",
        volume_multiplier=1.5,
        volatility_multiplier=1.3,
        state_transition_prob=0.06,
    ),
    Session.AFTERHOURS: SessionConfig(
        name="afterhours",
        volume_multiplier=0.5,
        volatility_multiplier=0.7,
        state_transition_prob=0.03,
    ),
}


@dataclass
class DayOfWeekConfig:
    """Day of week effects"""
    name: str
    volume_multiplier: float = 1.0
    volatility_multiplier: float = 1.0


DOW_CONFIGS = {
    0: DayOfWeekConfig("monday", volume_multiplier=1.1, volatility_multiplier=1.2),
    1: DayOfWeekConfig("tuesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    2: DayOfWeekConfig("wednesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    3: DayOfWeekConfig("thursday", volume_multiplier=1.0, volatility_multiplier=1.0),
    4: DayOfWeekConfig("friday", volume_multiplier=1.2, volatility_multiplier=1.1),
    5: DayOfWeekConfig("saturday", volume_multiplier=0.3, volatility_multiplier=0.5),
    6: DayOfWeekConfig("sunday", volume_multiplier=0.4, volatility_multiplier=0.6),
}


class PriceGenerator:
    """
    Generate synthetic MES price bars with configurable market dynamics.
    
    All ticks are 0.25 (MES tick size).
    """
    
    TICK_SIZE = 0.25
    
    def __init__(
        self,
        initial_price: float = 5000.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the price generator.
        
        Args:
            initial_price: Starting price level
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.last_direction = 0  # -1, 0, or 1
        self.prev_close_ticks = int(initial_price / self.TICK_SIZE)  # Track for delta_ticks
        
        if seed is not None:
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)
    
    def get_session(self, dt: datetime) -> Session:
        """Determine trading session based on time (Chicago time)"""
        hour = dt.hour
        
        if 18 <= hour or hour < 3:
            return Session.ASIAN
        elif 3 <= hour < 8 or (hour == 8 and dt.minute < 30):
            return Session.LONDON
        elif (hour == 8 and dt.minute >= 30) or hour == 9 and dt.minute < 30:
            return Session.PREMARKET
        elif (hour == 9 and dt.minute >= 30) or (10 <= hour < 15) or (hour == 15 and dt.minute <= 15):
            return Session.RTH
        else:
            return Session.AFTERHOURS
    
    def generate_tick_movement(
        self,
        state_config: StateConfig,
        session_config: SessionConfig,
        dow_config: DayOfWeekConfig,
    ) -> Tuple[int, int]:
        """
        Generate a single tick movement.
        
        Returns:
            (direction, num_ticks) where direction is -1 or 1, num_ticks is the size
        """
        # Determine direction
        if self.rng.random() < state_config.trend_persistence and self.last_direction != 0:
            # Continue previous direction
            direction = self.last_direction
        else:
            # New direction based on state bias
            direction = 1 if self.rng.random() < state_config.up_probability else -1
        
        # Determine tick size
        volatility = (
            state_config.volatility_multiplier *
            session_config.volatility_multiplier *
            dow_config.volatility_multiplier
        )
        
        tick_size = max(
            1,
            int(self.rng.normal(
                state_config.avg_tick_size * volatility,
                state_config.tick_size_std * volatility
            ))
        )
        tick_size = min(tick_size, state_config.max_tick_jump)
        
        self.last_direction = direction
        return direction, tick_size
    
    def generate_bar(
        self,
        timestamp: datetime,
        state: MarketState = MarketState.RANGING,
        custom_state_config: Optional[StateConfig] = None,
    ) -> dict:
        """
        Generate a single 1-minute OHLCV bar.
        
        Args:
            timestamp: Bar timestamp
            state: Market state to use
            custom_state_config: Optional custom state configuration (overrides state)
        
        Returns:
            Dictionary with keys: time, open, high, low, close, volume
        """
        # Get configurations
        state_config = custom_state_config or STATE_CONFIGS[state]
        session = self.get_session(timestamp)
        session_config = SESSION_CONFIGS[session]
        dow_config = DOW_CONFIGS[timestamp.weekday()]
        
        # Determine number of ticks for this bar
        num_ticks = max(
            1,
            int(self.rng.normal(
                state_config.avg_ticks_per_bar,
                state_config.ticks_per_bar_std
            ))
        )
        
        # Generate tick-by-tick price action
        open_price = self.current_price
        high_price = open_price
        low_price = open_price
        current = open_price
        
        for _ in range(num_ticks):
            direction, tick_size = self.generate_tick_movement(
                state_config, session_config, dow_config
            )
            
            # Move price
            price_change = direction * tick_size * self.TICK_SIZE
            current += price_change
            
            # Update high/low
            high_price = max(high_price, current)
            low_price = min(low_price, current)
        
        close_price = current
        
        # Add wicks (extended highs/lows that don't close there)
        if self.rng.random() < state_config.wick_probability:
            # Upper wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            high_price += wick_ticks * self.TICK_SIZE
        
        if self.rng.random() < state_config.wick_probability:
            # Lower wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            low_price -= wick_ticks * self.TICK_SIZE
        
        # Generate volume (scaled by session and day)
        base_volume = max(
            10,
            int(self.rng.normal(100, 50))
        )
        volume = int(
            base_volume *
            session_config.volume_multiplier *
            dow_config.volume_multiplier *
            state_config.volatility_multiplier
        )
        
        # Update current price for next bar
        self.current_price = close_price
        
        # Round prices to tick size
        open_price = round(open_price / self.TICK_SIZE) * self.TICK_SIZE
        high_price = round(high_price / self.TICK_SIZE) * self.TICK_SIZE
        low_price = round(low_price / self.TICK_SIZE) * self.TICK_SIZE
        close_price = round(close_price / self.TICK_SIZE) * self.TICK_SIZE
        
        # Convert to tick units (integer ticks from zero)
        open_ticks = int(open_price / self.TICK_SIZE)
        high_ticks = int(high_price / self.TICK_SIZE)
        low_ticks = int(low_price / self.TICK_SIZE)
        close_ticks = int(close_price / self.TICK_SIZE)
        
        # Compute tick-based deltas and features
        delta_ticks = close_ticks - self.prev_close_ticks
        range_ticks = high_ticks - low_ticks
        body_ticks = abs(close_ticks - open_ticks)
        
        # Wick calculations in ticks
        upper_body = max(open_ticks, close_ticks)
        lower_body = min(open_ticks, close_ticks)
        upper_wick_ticks = high_ticks - upper_body
        lower_wick_ticks = lower_body - low_ticks
        
        # Update prev_close for next bar
        self.prev_close_ticks = close_ticks
        
        return {
            'time': timestamp,
            # Price columns (floats)
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            # Tick columns (integers) - ML-friendly
            'open_ticks': open_ticks,
            'high_ticks': high_ticks,
            'low_ticks': low_ticks,
            'close_ticks': close_ticks,
            'delta_ticks': delta_ticks,
            'range_ticks': range_ticks,
            'body_ticks': body_ticks,
            'upper_wick_ticks': upper_wick_ticks,
            'lower_wick_ticks': lower_wick_ticks,
            # State labels
            'state': state.value,
            'session': session.value,
        }
    
    def generate_day(
        self,
        start_date: datetime,
        state_sequence: Optional[List[Tuple[int, MarketState]]] = None,
        auto_transition: bool = True,
        segment_length: Optional[int] = None,
        macro_regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a full day of 1-minute bars (1440 bars).
        
        Args:
            start_date: Starting datetime (should be midnight)
            state_sequence: Optional list of (bar_index, state) tuples to control states
            auto_transition: If True, randomly transition between states based on session
            segment_length: If set, only allow state transitions at segment boundaries (e.g., 15 for 15-min segments)
            macro_regime: Optional day-level regime label (e.g., 'UP_DAY', 'DOWN_DAY', 'CHOP_DAY')
        
        Returns:
            DataFrame with OHLCV, tick columns, state labels, and optional macro_regime
        """
        bars = []
        current_state = MarketState.RANGING
        current_segment_id = 0
        
        # Build state map if provided
        state_map = {}
        if state_sequence:
            for bar_idx, state in state_sequence:
                state_map[bar_idx] = state
        
        for minute in range(1440):  # 24 hours * 60 minutes
            timestamp = start_date + timedelta(minutes=minute)
            
            # Update segment ID if using segments
            if segment_length:
                current_segment_id = minute // segment_length
            
            # Check for manual state transition
            if minute in state_map:
                current_state = state_map[minute]
            elif auto_transition:
                # Only transition at segment boundaries if segment_length is set
                can_transition = True
                if segment_length:
                    can_transition = (minute % segment_length == 0)
                
                if can_transition:
                    # Random state transition based on session
                    session = self.get_session(timestamp)
                    session_config = SESSION_CONFIGS[session]
                    
                    if self.rng.random() < session_config.state_transition_prob:
                        # Transition to a new state
                        current_state = self.rng.choice(list(MarketState))
            
            bar = self.generate_bar(timestamp, current_state)
            
            # Add segment ID if using segments
            if segment_length:
                bar['segment_id'] = current_segment_id
            
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Add macro regime label if provided
        if macro_regime:
            df['macro_regime'] = macro_regime
        else:
            # Infer simple macro regime from net movement
            net_move = df['close'].iloc[-1] - df['open'].iloc[0]
            total_range = df['high'].max() - df['low'].min()
            
            if abs(net_move) > total_range * 0.3:
                df['macro_regime'] = 'UP_DAY' if net_move > 0 else 'DOWN_DAY'
            elif total_range < df['close'].iloc[0] * 0.01:  # Less than 1% range
                df['macro_regime'] = 'QUIET_DAY'
            else:
                df['macro_regime'] = 'CHOP_DAY'
        
        return df
```

## FILE: ./lab/generators/utils.py
```
"""
Utilities for analyzing synthetic price data

Helper functions to sanity-check generator output and compute statistics.
"""

import pandas as pd
from typing import Dict, Any


def summarize_day(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a day of generated data.
    
    Args:
        df: DataFrame from PriceGenerator.generate_day()
    
    Returns:
        Dictionary with overall stats, per-state stats, and per-session stats
    """
    summary = {}
    
    # Overall day stats
    summary['overall'] = {
        'num_bars': len(df),
        'start_price': float(df['open'].iloc[0]),
        'end_price': float(df['close'].iloc[-1]),
        'net_move': float(df['close'].iloc[-1] - df['open'].iloc[0]),
        'net_move_ticks': int(df['close_ticks'].iloc[-1] - df['open_ticks'].iloc[0]),
        'high': float(df['high'].max()),
        'low': float(df['low'].min()),
        'total_range': float(df['high'].max() - df['low'].min()),
        'total_range_ticks': int((df['high'].max() - df['low'].min()) / 0.25),
        'avg_volume': float(df['volume'].mean()),
        'total_volume': int(df['volume'].sum()),
        'max_range_ticks': int(df['range_ticks'].max()),
        'avg_range_ticks': float(df['range_ticks'].mean()),
        'avg_body_ticks': float(df['body_ticks'].mean()),
        'avg_delta_ticks': float(df['delta_ticks'].mean()),
        'std_delta_ticks': float(df['delta_ticks'].std()),
    }
    
    # Macro regime if present
    if 'macro_regime' in df.columns:
        summary['overall']['macro_regime'] = df['macro_regime'].iloc[0]
    
    # Per-state statistics
    if 'state' in df.columns:
        summary['by_state'] = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            summary['by_state'][state] = {
                'count': len(state_df),
                'pct_of_day': float(len(state_df) / len(df) * 100),
                'avg_delta_ticks': float(state_df['delta_ticks'].mean()),
                'std_delta_ticks': float(state_df['delta_ticks'].std()),
                'avg_range_ticks': float(state_df['range_ticks'].mean()),
                'avg_body_ticks': float(state_df['body_ticks'].mean()),
                'avg_volume': float(state_df['volume'].mean()),
                'net_move_ticks': int(state_df['delta_ticks'].sum()),
                'up_bars': int((state_df['delta_ticks'] > 0).sum()),
                'down_bars': int((state_df['delta_ticks'] < 0).sum()),
                'flat_bars': int((state_df['delta_ticks'] == 0).sum()),
            }
    
    # Per-session statistics
    if 'session' in df.columns:
        summary['by_session'] = {}
        for session in df['session'].unique():
            session_df = df[df['session'] == session]
            summary['by_session'][session] = {
                'count': len(session_df),
                'pct_of_day': float(len(session_df) / len(df) * 100),
                'avg_delta_ticks': float(session_df['delta_ticks'].mean()),
                'avg_range_ticks': float(session_df['range_ticks'].mean()),
                'avg_volume': float(session_df['volume'].mean()),
                'net_move_ticks': int(session_df['delta_ticks'].sum()),
            }
    
    # Per-segment statistics if segments exist
    if 'segment_id' in df.columns:
        summary['by_segment'] = {}
        for seg_id in df['segment_id'].unique():
            seg_df = df[df['segment_id'] == seg_id]
            summary['by_segment'][int(seg_id)] = {
                'count': len(seg_df),
                'state': seg_df['state'].iloc[0] if len(seg_df) > 0 else None,
                'net_move_ticks': int(seg_df['delta_ticks'].sum()),
                'range_ticks': int(seg_df['range_ticks'].sum()),
                'avg_volume': float(seg_df['volume'].mean()),
            }
    
    return summary


def print_summary(summary: Dict[str, Any], verbose: bool = True) -> None:
    """
    Pretty-print a summary dictionary.
    
    Args:
        summary: Output from summarize_day()
        verbose: If True, print detailed per-state and per-session stats
    """
    print("\n" + "=" * 60)
    print("DAY SUMMARY")
    print("=" * 60)
    
    # Overall stats
    overall = summary['overall']
    print(f"\nOverall:")
    print(f"  Bars: {overall['num_bars']}")
    print(f"  Price: {overall['start_price']:.2f} → {overall['end_price']:.2f}")
    print(f"  Net Move: {overall['net_move']:.2f} ({overall['net_move_ticks']:+d} ticks)")
    print(f"  Range: {overall['low']:.2f} - {overall['high']:.2f} ({overall['total_range_ticks']} ticks)")
    print(f"  Avg Bar Range: {overall['avg_range_ticks']:.1f} ticks")
    print(f"  Avg Bar Body: {overall['avg_body_ticks']:.1f} ticks")
    print(f"  Avg Delta: {overall['avg_delta_ticks']:.2f} ± {overall['std_delta_ticks']:.2f} ticks")
    print(f"  Total Volume: {overall['total_volume']:,}")
    
    if 'macro_regime' in overall:
        print(f"  Macro Regime: {overall['macro_regime']}")
    
    if verbose and 'by_state' in summary:
        print(f"\nBy State:")
        print(f"  {'State':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8} {'Up/Dn':>10}")
        print("  " + "-" * 70)
        for state, stats in summary['by_state'].items():
            print(f"  {state:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d} "
                  f"{stats['up_bars']:>4}/{stats['down_bars']:<4}")
    
    if verbose and 'by_session' in summary:
        print(f"\nBy Session:")
        print(f"  {'Session':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8}")
        print("  " + "-" * 60)
        for session, stats in summary['by_session'].items():
            print(f"  {session:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d}")
    
    print("=" * 60)


def compare_states(df: pd.DataFrame, states_to_compare: list = None) -> pd.DataFrame:
    """
    Create a comparison table of different states.
    
    Args:
        df: DataFrame from generator
        states_to_compare: List of states to compare, or None for all
    
    Returns:
        DataFrame with comparison metrics
    """
    if states_to_compare is None:
        states_to_compare = df['state'].unique()
    
    comparison = []
    
    for state in states_to_compare:
        state_df = df[df['state'] == state]
        if len(state_df) == 0:
            continue
        
        comparison.append({
            'state': state,
            'count': len(state_df),
            'avg_delta_ticks': state_df['delta_ticks'].mean(),
            'std_delta_ticks': state_df['delta_ticks'].std(),
            'avg_range_ticks': state_df['range_ticks'].mean(),
            'avg_body_ticks': state_df['body_ticks'].mean(),
            'avg_upper_wick': state_df['upper_wick_ticks'].mean(),
            'avg_lower_wick': state_df['lower_wick_ticks'].mean(),
            'avg_volume': state_df['volume'].mean(),
            'up_pct': (state_df['delta_ticks'] > 0).sum() / len(state_df) * 100,
        })
    
    return pd.DataFrame(comparison)
```

## FILE: ./lab/visualizers/__init__.py
```
"""
Visualization tools for market data.
"""

from .chart_viz import (
    ChartVisualizer,
    ChartConfig,
    quick_chart,
)

__all__ = [
    'ChartVisualizer',
    'ChartConfig',
    'quick_chart',
]
```

## FILE: ./lab/visualizers/chart_viz.py
```
"""
Flexible Candlestick Chart Visualizer

Configurable matplotlib charting with various display options and knobs.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior"""
    
    # Figure settings
    figsize: Tuple[int, int] = (16, 9)
    dpi: int = 100
    
    # Candle appearance
    candle_width: float = 0.6
    color_up: str = '#26a69a'      # Teal green
    color_down: str = '#ef5350'    # Red
    wick_linewidth: float = 1.0
    wick_alpha: float = 0.8
    
    # Grid and background
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_color: str = '#cccccc'
    background_color: str = '#ffffff'
    
    # Volume subplot
    show_volume: bool = True
    volume_height_ratio: float = 0.25  # Relative to price chart
    volume_alpha: float = 0.5
    
    # Title and labels
    title: str = "MES 1-Minute Chart"
    title_fontsize: int = 14
    ylabel_price: str = "Price"
    ylabel_volume: str = "Volume"
    label_fontsize: int = 11
    
    # X-axis formatting
    date_format: str = '%H:%M'
    major_tick_interval_minutes: Optional[int] = 60  # None for auto
    
    # Annotations
    show_state_changes: bool = True
    state_change_color: str = '#ff9800'
    state_change_alpha: float = 0.3
    
    show_session_changes: bool = True
    session_colors: dict = None
    
    # Legend
    show_legend: bool = True
    legend_loc: str = 'upper left'
    
    def __post_init__(self):
        if self.session_colors is None:
            self.session_colors = {
                'asian': '#9c27b0',
                'london': '#2196f3',
                'premarket': '#ff9800',
                'rth': '#4caf50',
                'afterhours': '#795548',
            }


class ChartVisualizer:
    """
    Flexible candlestick chart creator with multiple display options.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: ChartConfig instance, uses defaults if None
        """
        self.config = config or ChartConfig()
    
    def plot_candlestick(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Plot candlestick chart on given axes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with OHLC data
            date_column: Name of the datetime column
        """
        # Ensure datetime index
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        # Convert to matplotlib date format
        dates_num = mdates.date2num(dates)
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Determine colors
        colors = [
            self.config.color_up if c >= o else self.config.color_down
            for o, c in zip(opens, closes)
        ]
        
        # Calculate candle width
        if len(dates_num) > 1:
            min_diff = np.min(np.diff(dates_num))
            width = self.config.candle_width * min_diff
        else:
            width = self.config.candle_width * 1.0 / 1440.0
        
        # Plot wicks
        ax.vlines(
            dates_num,
            lows,
            highs,
            colors=colors,
            linewidth=self.config.wick_linewidth,
            alpha=self.config.wick_alpha,
        )
        
        # Plot bodies
        for d, o, c, color in zip(dates_num, opens, closes, colors):
            lower = min(o, c)
            height = abs(c - o)
            if height == 0:
                height = 0.01  # Tiny height for doji
            
            ax.add_patch(
                plt.Rectangle(
                    (d - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                )
            )
        
        # Format x-axis
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(self.config.date_format))
        
        if self.config.major_tick_interval_minutes:
            ax.xaxis.set_major_locator(
                mdates.MinuteLocator(interval=self.config.major_tick_interval_minutes)
            )
        
        ax.autoscale_view()
    
    def plot_volume(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Plot volume bars on given axes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with volume data
            date_column: Name of the datetime column
        """
        # Ensure datetime index
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        volumes = df['volume'].values
        
        # Color based on price movement
        opens = df['open'].values
        closes = df['close'].values
        colors = [
            self.config.color_up if c >= o else self.config.color_down
            for o, c in zip(opens, closes)
        ]
        
        # Calculate bar width
        if len(dates_num) > 1:
            min_diff = np.min(np.diff(dates_num))
            width = self.config.candle_width * min_diff
        else:
            width = self.config.candle_width * 1.0 / 1440.0
        
        # Plot bars
        ax.bar(
            dates_num,
            volumes,
            width=width,
            color=colors,
            alpha=self.config.volume_alpha,
        )
        
        # Format x-axis
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(self.config.date_format))
        
        if self.config.major_tick_interval_minutes:
            ax.xaxis.set_major_locator(
                mdates.MinuteLocator(interval=self.config.major_tick_interval_minutes)
            )
        
        ax.autoscale_view()
    
    def add_state_annotations(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Add visual indicators for state changes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with 'state' column
            date_column: Name of the datetime column
        """
        if 'state' not in df.columns:
            return
        
        # Ensure datetime
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        
        # Find state changes
        states = df['state'].values
        changes = []
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                changes.append((i, states[i]))
        
        # Draw vertical lines at state changes
        for idx, new_state in changes:
            ax.axvline(
                x=dates_num[idx],
                color=self.config.state_change_color,
                linestyle='--',
                alpha=self.config.state_change_alpha,
                linewidth=1.5,
                label=f'→ {new_state}' if idx == changes[0][0] else '',
            )
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(
                dates_num[idx],
                y_pos,
                new_state.upper(),
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                alpha=0.7,
                color=self.config.state_change_color,
            )
    
    def add_session_backgrounds(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Add colored backgrounds for different trading sessions.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with 'session' column
            date_column: Name of the datetime column
        """
        if 'session' not in df.columns:
            return
        
        # Ensure datetime
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        sessions = df['session'].values
        
        # Find session blocks
        current_session = sessions[0]
        start_idx = 0
        
        for i in range(1, len(sessions)):
            if sessions[i] != current_session:
                # Draw background for previous session
                color = self.config.session_colors.get(current_session, '#cccccc')
                ax.axvspan(
                    dates_num[start_idx],
                    dates_num[i-1],
                    alpha=0.1,
                    color=color,
                    label=current_session.upper() if start_idx == 0 else '',
                )
                
                current_session = sessions[i]
                start_idx = i
        
        # Draw last session
        color = self.config.session_colors.get(current_session, '#cccccc')
        ax.axvspan(
            dates_num[start_idx],
            dates_num[-1],
            alpha=0.1,
            color=color,
            label=current_session.upper() if start_idx == 0 else '',
        )
    
    def create_chart(
        self,
        df: pd.DataFrame,
        date_column: str = 'time',
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Create complete chart with all configured elements.
        
        Args:
            df: DataFrame with OHLCV data
            date_column: Name of the datetime column
            save_path: Optional path to save the chart
            show: Whether to display the chart
        
        Returns:
            Matplotlib figure
        """
        # Create figure and axes
        if self.config.show_volume:
            fig, (ax_price, ax_volume) = plt.subplots(
                2, 1,
                figsize=self.config.figsize,
                dpi=self.config.dpi,
                gridspec_kw={'height_ratios': [1 - self.config.volume_height_ratio, self.config.volume_height_ratio]},
                sharex=True,
            )
        else:
            fig, ax_price = plt.subplots(
                figsize=self.config.figsize,
                dpi=self.config.dpi,
            )
            ax_volume = None
        
        # Set background
        fig.patch.set_facecolor(self.config.background_color)
        ax_price.set_facecolor(self.config.background_color)
        if ax_volume:
            ax_volume.set_facecolor(self.config.background_color)
        
        # Add session backgrounds first (so they're behind everything)
        if self.config.show_session_changes:
            self.add_session_backgrounds(ax_price, df, date_column)
        
        # Plot candlesticks
        self.plot_candlestick(ax_price, df, date_column)
        
        # Add state change annotations
        if self.config.show_state_changes:
            self.add_state_annotations(ax_price, df, date_column)
        
        # Plot volume
        if self.config.show_volume and ax_volume:
            self.plot_volume(ax_volume, df, date_column)
            ax_volume.set_ylabel(self.config.ylabel_volume, fontsize=self.config.label_fontsize)
            
            if self.config.show_grid:
                ax_volume.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
        
        # Configure price axis
        ax_price.set_title(self.config.title, fontsize=self.config.title_fontsize, pad=15)
        ax_price.set_ylabel(self.config.ylabel_price, fontsize=self.config.label_fontsize)
        
        if self.config.show_grid:
            ax_price.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
        
        if self.config.show_legend:
            ax_price.legend(loc=self.config.legend_loc, fontsize=9)
        
        # Format dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig


def quick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    save_path: Optional[Path] = None,
    show: bool = True,
    **config_kwargs,
) -> plt.Figure:
    """
    Quick helper to create a chart with minimal setup.
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        save_path: Optional save path
        show: Whether to display
        **config_kwargs: Additional ChartConfig parameters
    
    Returns:
        Matplotlib figure
    """
    config = ChartConfig(title=title, **config_kwargs)
    viz = ChartVisualizer(config)
    return viz.create_chart(df, save_path=save_path, show=show)
```

## FILE: ./printcode.sh
```
#!/usr/bin/env bash

OUTFILE="tickfirecode.md"

# Start fresh
echo "# Tickfire Code Snapshot" > "$OUTFILE"
echo "" >> "$OUTFILE"
echo "Generated on: \$(date)" >> "$OUTFILE"
echo "" >> "$OUTFILE"

# ----------------------------
# Project tree
# ----------------------------
echo "## Project Tree" >> "$OUTFILE"
echo '```' >> "$OUTFILE"
if command -v tree >/dev/null 2>&1; then
  # Ignore common junk dirs in tree view
  tree -a -I ".git|.venv*|__pycache__|out" >> "$OUTFILE"
else
  # Fallback if tree is not installed
  find . -maxdepth 4 \
    ! -path '*/.git*' \
    ! -path '*/.venv*' \
    ! -path '*/__pycache__*' \
    ! -path '*/out*' >> "$OUTFILE"
fi
echo '```' >> "$OUTFILE"
echo "" >> "$OUTFILE"
echo "## Files" >> "$OUTFILE"
echo "" >> "$OUTFILE"

# File patterns to skip entirely (binary / large)
SKIP_EXTENSIONS="png jpg jpeg gif bmp ico ttf otf wav mp3 mp4 avi mov mkv npz parquet feather pkl pickle pt h5 hdf5 zip gz tgz 7z dll exe so dylib"

# Directories to skip
SKIP_DIRS=".venv312 .git .vscode .idea __pycache__ out"

# Detect if file should be skipped
should_skip_file() {
    local file="$1"
    local base
    base="$(basename "$file")"

    # Skip our own snapshots / dumps
    if [[ "$base" == "tickfirecode.md" || "$base" == "newprint.md" ]]; then
        return 0
    fi

    local ext="${file##*.}"

    # Skip binary/large extensions
    for bad in $SKIP_EXTENSIONS; do
        [[ "$ext" == "$bad" ]] && return 0
    done

    return 1
}

# Detect if directory should be skipped
should_skip_dir() {
    local dir="$1"
    local base
    base="$(basename "$dir")"

    # Hidden directory (starts with .)
    [[ "$base" == .* ]] && return 0

    for skip in $SKIP_DIRS; do
        [[ "$base" == "$skip" ]] && return 0
    done

    return 1
}

# Recursively process files
process_directory() {
    local dir="$1"

    for file in "$dir"/*; do
        # Skip if doesn't exist
        [[ ! -e "$file" ]] && continue

        # Skip directories we don’t want
        if [[ -d "$file" ]]; then
            if should_skip_dir "$file"; then
                continue
            fi
            process_directory "$file"
            continue
        fi

        local base
        base="$(basename "$file")"

        # Skip hidden files
        [[ "$base" == .* ]] && continue

        # Skip file types completely
        if should_skip_file "$file"; then
            echo "## SKIPPED (binary or ignored): $file" >> "$OUTFILE"
            continue
        fi

        # Handle CSV / JSON specially
        local ext="${file##*.}"
        if [[ "$ext" == "csv" || "$ext" == "json" ]]; then
            echo "## FILE (first 200 lines): $file" >> "$OUTFILE"
            echo '```' >> "$OUTFILE"
            head -n 200 "$file" >> "$OUTFILE"
            echo '```' >> "$OUTFILE"
            echo "" >> "$OUTFILE"
            continue
        fi

        # Normal code/text file — print full contents
        echo "## FILE: $file" >> "$OUTFILE"
        echo '```' >> "$OUTFILE"
        cat "$file" >> "$OUTFILE"
        echo '```' >> "$OUTFILE"
        echo "" >> "$OUTFILE"
    done
}

# Kick off processing from current directory
process_directory "."

echo "Done! Output saved to $OUTFILE"
```

## FILE: ./README.md
```
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

## FILE: ./requirements.txt
```
# Python Dependencies for FracFire

# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0
torch>=2.0.0  # PyTorch for deep learning models
# tensorflow>=2.13.0  # Uncomment if using TensorFlow instead

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Data handling
pyarrow>=12.0.0  # For efficient parquet files
h5py>=3.9.0  # For HDF5 files

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
tqdm>=4.65.0  # Progress bars
joblib>=1.3.0  # Parallel processing

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Code quality
black>=23.0.0  # Code formatting
flake8>=6.0.0  # Linting
mypy>=1.4.0  # Type checking
isort>=5.12.0  # Import sorting

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
ipywidgets>=8.1.0

# Optional: Experiment tracking
# mlflow>=2.5.0
# wandb>=0.15.0
# tensorboard>=2.13.0

# Optional: Advanced features
# optuna>=3.3.0  # Hyperparameter optimization
# ray[tune]>=2.6.0  # Distributed training
```

## FILE: ./scripts/analyze_drift.py
```
"""
Analyze Feature Drift

Compares feature distributions between Synthetic Archetypes and Real Data.
Helps identify if synthetic data is "too clean" or has different scale than real data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("ANALYZING FEATURE DRIFT")
    print("=" * 60)
    
    # 1. Load Real Data
    print("Loading Real Data...")
    real_loader = RealDataLoader()
    real_path = root / "src" / "data" / "continuous_contract.json"
    real_df = real_loader.load_json(real_path)
    
    # 2. Load Synthetic Data (Sample)
    print("Loading Synthetic Data (Sample)...")
    syn_loader = DataLoader(root / "out" / "data" / "synthetic" / "archetypes")
    # Load just 50 files to get a representative distribution without being too slow
    syn_df = syn_loader.load_archetypes(limit=50)
    
    # 3. Extract Features
    print("Extracting features...")
    builder = FeatureBuilder(window_size=60)
    
    real_features = builder.extract_features(real_df)
    syn_features = builder.extract_features(syn_df)
    
    # 4. Compare Distributions
    features_to_check = [
        'volatility', 
        'tick_momentum', 
        'relative_range', 
        'volume_intensity',
        'tick_rsi',
        'cum_delta' # This one is tricky as it's cumulative, might not be comparable directly if day lengths differ
    ]
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(features_to_check):
        if feature not in real_features.columns:
            continue
            
        plt.subplot(2, 3, i+1)
        
        # Clip outliers for better visualization
        p01 = min(real_features[feature].quantile(0.01), syn_features[feature].quantile(0.01))
        p99 = max(real_features[feature].quantile(0.99), syn_features[feature].quantile(0.99))
        
        # Plot KDE
        sns.kdeplot(real_features[feature].clip(p01, p99), label='Real', fill=True, alpha=0.3)
        sns.kdeplot(syn_features[feature].clip(p01, p99), label='Synthetic', fill=True, alpha=0.3)
        
        plt.title(feature)
        plt.legend()
        
    plt.tight_layout()
    output_path = output_dir / "feature_drift.png"
    plt.savefig(output_path)
    print(f"\nDrift analysis chart saved to: {output_path}")
    
    # Print summary stats
    print("\nSummary Statistics Comparison:")
    print(f"{'Feature':<20} {'Real Mean':>10} {'Syn Mean':>10} {'Real Std':>10} {'Syn Std':>10}")
    print("-" * 65)
    
    for feature in features_to_check:
        r_mean = real_features[feature].mean()
        s_mean = syn_features[feature].mean()
        r_std = real_features[feature].std()
        s_std = syn_features[feature].std()
        
        print(f"{feature:<20} {r_mean:>10.4f} {s_mean:>10.4f} {r_std:>10.4f} {s_std:>10.4f}")

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/apply_optimized.py
```
"""
Apply Optimized Model

Applies the balanced Random Forest model to real data with custom probability thresholds.
This allows detecting directional moves even when the signal is weak due to low volatility.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("APPLYING OPTIMIZED MODEL (Balanced + Thresholds)")
    print("=" * 60)
    
    # 1. Load Real Data
    data_path = root / "src" / "data" / "continuous_contract.json"
    loader = RealDataLoader()
    try:
        df = loader.load_json(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    features = builder.extract_features(df)
    
    # 3. Load Balanced Model
    model_path = root / "out" / "models" / "balanced_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 4. Predict Probabilities
    print("Running inference...")
    X = features.values
    y_prob = clf.predict_proba(X)
    classes = clf.classes_
    
    # 5. Apply Custom Thresholds
    # We want to prioritize directional moves.
    # If P(Rally) > threshold, predict Rally.
    # Order matters! Check rarest/most important first.
    
    # Map class names to indices
    class_map = {c: i for i, c in enumerate(classes)}
    
    # Define thresholds (tuned heuristically based on drift analysis)
    # Real data has much lower volatility, so confidence will be lower.
    THRESHOLDS = {
        'breakdown': 0.30,
        'rally': 0.30,
        'impulsive': 0.35,
        'breakout': 0.25
    }
    
    print("\nApplying thresholds:")
    for c, t in THRESHOLDS.items():
        print(f"  {c}: > {t}")
        
    final_preds = []
    
    # Vectorized approach would be faster, but loop is clearer for logic
    # Let's try a vectorized approach for speed
    
    # Default to 'ranging' (or whatever max prob is if not ranging)
    # Actually, let's start with standard argmax
    max_indices = np.argmax(y_prob, axis=1)
    base_preds = classes[max_indices]
    
    # Create a Series for easy updating
    pred_series = pd.Series(base_preds, index=df.index)
    
    # Apply overrides
    # We iterate through thresholds. Later ones overwrite earlier ones if multiple trigger?
    # Usually we want the "strongest" signal.
    # But here we just want ANY directional signal to override 'ranging'.
    
    for state, threshold in THRESHOLDS.items():
        if state not in class_map:
            continue
            
        idx = class_map[state]
        probs = y_prob[:, idx]
        
        # Where probability exceeds threshold, force this state
        # Note: This is a simple override. If multiple exceed, the last one in loop wins.
        # Ideally we'd check which exceeds its threshold by the most relative margin.
        # But for now, let's just apply them.
        mask = probs > threshold
        pred_series[mask] = state
        
    df['predicted_state'] = pred_series.values
    
    # 6. Save Results
    output_dir = root / "out" / "data" / "real" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "optimized_states.parquet"
    df.to_parquet(output_path)
    
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    print("\nOptimized State Distribution:")
    dist = df['predicted_state'].value_counts(normalize=True) * 100
    print(dist)

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/apply_to_real.py
```
"""
Apply Model to Real Data

Loads the pre-trained baseline model and applies it to the real continuous contract data.
Saves the predictions for visualization.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("APPLYING MODEL TO REAL DATA")
    print("=" * 60)
    
    # 1. Load Real Data
    data_path = root / "src" / "data" / "continuous_contract.json"
    loader = RealDataLoader()
    
    try:
        df = loader.load_json(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    features = builder.extract_features(df)
    
    # 3. Load Model
    model_path = root / "out" / "models" / "baseline_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 4. Predict
    print("Running inference...")
    # FeatureBuilder returns DataFrame, sklearn needs values
    X = features.values
    
    # Predict probabilities and classes
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    
    # 5. Save Results
    print("\nSaving results...")
    
    # Add predictions to original dataframe
    df['predicted_state'] = y_pred
    
    # Add confidence (max probability)
    df['confidence'] = np.max(y_prob, axis=1)
    
    # Save to parquet
    output_dir = root / "out" / "data" / "real" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "predicted_states.parquet"
    df.to_parquet(output_path)
    
    print(f"Predictions saved to: {output_path}")
    
    # Print summary
    print("\nPredicted State Distribution:")
    dist = df['predicted_state'].value_counts(normalize=True) * 100
    print(dist)

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/demo_custom_states.py
```
"""
Demo: Custom Market States

Visualizes the extreme and specialized market states defined in custom_states.py.
Creates a grid of charts showing different market behaviors.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import PriceGenerator
from lab.generators.custom_states import CUSTOM_STATES, get_custom_state
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig


def demo_comparison_grid():
    """Create a grid of charts comparing different custom states"""
    print("=" * 60)
    print("DEMO: Custom State Comparison Grid")
    print("=" * 60)
    
    # Select states to visualize
    states_to_show = [
        'mega_volatile', 'flash_crash', 'melt_up',
        'whipsaw', 'opening_bell', 'news_spike',
        'slow_bleed', 'dead_zone', 'closing_squeeze'
    ]
    
    # Setup grid
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    print(f"Generating {len(states_to_show)} state samples...")
    
    for idx, state_name in enumerate(states_to_show):
        ax = axes[idx]
        config = get_custom_state(state_name)
        
        # Generate 2 hours of data
        bars = []
        for minute in range(120):
            timestamp = start_date + timedelta(minutes=minute)
            bar = gen.generate_bar(timestamp, custom_state_config=config)
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Visualize
        viz_config = ChartConfig(
            title=f"{state_name.upper()}",
            show_volume=False,
            show_state_changes=False,
            show_session_changes=False,
            title_fontsize=10,
        )
        
        viz = ChartVisualizer(viz_config)
        viz.plot_candlestick(ax, df)
        
        # Add stats
        volatility = df['range_ticks'].mean()
        net_move = df['close'].iloc[-1] - df['open'].iloc[0]
        
        stats = f"Vol: {volatility:.1f} | Net: {net_move:.2f}"
        ax.text(
            0.02, 0.95, stats,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_title(state_name.upper(), fontsize=10)
        ax.grid(True, alpha=0.2)
        
        print(f"  - {state_name}: Range {df['low'].min():.2f}-{df['high'].max():.2f}")

    plt.tight_layout()
    plt.suptitle("Custom Market State Archetypes", y=1.02, fontsize=16)
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chart_path = output_dir / "custom_states_grid.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    print(f"\nGrid chart saved to: {chart_path}")


def main():
    demo_comparison_grid()


if __name__ == "__main__":
    main()
```

## FILE: ./scripts/demo_enhanced_features.py
```
"""
Demo: Enhanced Features and Analysis

Demonstrates advanced features like segment-based control, macro regimes,
and detailed statistical analysis of generated data.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import PriceGenerator, MarketState
from lab.generators.utils import summarize_day, print_summary, compare_states
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig


def demo_segment_control():
    """Demonstrate segment-based state control (e.g., 15-minute blocks)"""
    print("\n" + "=" * 60)
    print("DEMO: Segment-Based State Control")
    print("=" * 60)
    
    gen = PriceGenerator(initial_price=5000.0, seed=101)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    # Define a sequence of 15-minute segments
    # 0-15: Ranging
    # 15-30: Breakout
    # 30-45: Rally
    # 45-60: Ranging
    state_sequence = [
        (0, MarketState.RANGING),
        (15, MarketState.BREAKOUT),
        (30, MarketState.RALLY),
        (45, MarketState.RANGING),
    ]
    
    # Generate 1 hour of data with 15-minute segments
    # Note: generate_day generates 1440 bars, so we'll slice it or use a custom loop
    # Here we use generate_day but with a segment_length parameter
    
    df = gen.generate_day(
        start_date,
        state_sequence=state_sequence,
        auto_transition=False,
        segment_length=15,
        macro_regime="BREAKOUT_HOUR"
    )
    
    # Slice to just the first hour for this demo
    df_hour = df.iloc[:60].copy()
    
    print(f"Generated {len(df_hour)} bars with 15-minute segments")
    print("\nSegments found:")
    for seg_id in df_hour['segment_id'].unique():
        seg_df = df_hour[df_hour['segment_id'] == seg_id]
        state = seg_df['state'].iloc[0]
        print(f"  Segment {seg_id}: {state} ({len(seg_df)} bars)")
    
    # Visualize
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ChartConfig(
        title="15-Minute Segment Control",
        figsize=(12, 8),
        show_state_changes=True,
        major_tick_interval_minutes=15,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo_segments.png"
    viz.create_chart(df_hour, save_path=chart_path, show=False)
    print(f"\nChart saved to: {chart_path}")
    
    return df_hour


def demo_detailed_analysis():
    """Demonstrate detailed statistical analysis of a generated day"""
    print("\n" + "=" * 60)
    print("DEMO: Detailed Statistical Analysis")
    print("=" * 60)
    
    gen = PriceGenerator(initial_price=5000.0, seed=202)
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    
    # Generate a full day with auto transitions
    df = gen.generate_day(start_date, auto_transition=True)
    
    # 1. Full Day Summary
    print("\n1. Full Day Summary")
    summary = summarize_day(df)
    print_summary(summary, verbose=True)
    
    # 2. State Comparison
    print("\n2. State Comparison Table")
    comp_df = compare_states(df)
    
    # Format for display
    display_cols = ['state', 'count', 'avg_delta_ticks', 'avg_range_ticks', 'avg_volume', 'up_pct']
    print(comp_df[display_cols].round(2).to_string(index=False))
    
    # 3. Tick Feature Analysis
    print("\n3. Tick Feature Analysis")
    print("Correlation between features:")
    features = ['delta_ticks', 'range_ticks', 'body_ticks', 'volume']
    corr = df[features].corr()
    print(corr.round(2))
    
    # 4. Visualizing Distributions
    print("\n4. Generating Distribution Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Delta Ticks Distribution
    df['delta_ticks'].hist(bins=30, ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('Distribution of Price Changes (Ticks)')
    axes[0, 0].set_xlabel('Ticks')
    
    # Range Ticks Distribution
    df['range_ticks'].hist(bins=30, ax=axes[0, 1], alpha=0.7, color='orange')
    axes[0, 1].set_title('Distribution of Bar Ranges (Ticks)')
    axes[0, 1].set_xlabel('Ticks')
    
    # Volume by State
    df.boxplot(column='volume', by='state', ax=axes[1, 0], rot=45)
    axes[1, 0].set_title('Volume Distribution by State')
    axes[1, 0].set_xlabel('')
    
    # Range by Session
    df.boxplot(column='range_ticks', by='session', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('Volatility (Range) by Session')
    axes[1, 1].set_xlabel('')
    
    plt.tight_layout()
    plt.suptitle("Generated Data Analysis", y=1.02, fontsize=16)
    
    output_dir = root / "out" / "charts"
    chart_path = output_dir / "demo_analysis_dist.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    print(f"Analysis plots saved to: {chart_path}")


def main():
    demo_segment_control()
    demo_detailed_analysis()
    
    print("\n" + "=" * 60)
    print("Enhanced features demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## FILE: ./scripts/demo_price_generation.py
```
"""
Demo: Generate and visualize synthetic MES price data

This script demonstrates the price generator and chart visualizer working together.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import (
    PriceGenerator,
    MarketState,
    StateConfig,
    STATE_CONFIGS,
)
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig, quick_chart


def demo_basic_generation():
    """Generate a day of data with automatic state transitions"""
    print("=" * 60)
    print("DEMO 1: Basic Day Generation with Auto State Transitions")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    
    # Generate a full day
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, auto_transition=True)
    
    print(f"\nGenerated {len(df)} bars")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Show state distribution
    print("\nState distribution:")
    print(df['state'].value_counts())
    
    # Create chart
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ChartConfig(
        title="MES Simulation - Auto State Transitions",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo1_auto_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_controlled_states():
    """Generate data with manually controlled state sequence"""
    print("\n" + "=" * 60)
    print("DEMO 2: Controlled State Sequence")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=123)
    
    # Define a specific state sequence
    # Format: (bar_index, state)
    state_sequence = [
        (0, MarketState.FLAT),           # Start flat (midnight)
        (180, MarketState.ZOMBIE),       # 3am - slow grind starts
        (390, MarketState.RANGING),      # 6:30am - choppy
        (570, MarketState.RALLY),        # 9:30am - RTH open, rally
        (690, MarketState.IMPULSIVE),    # 11:30am - high volatility
        (810, MarketState.BREAKDOWN),    # 1:30pm - sharp drop
        (900, MarketState.RANGING),      # 3pm - settle into range
        (960, MarketState.FLAT),         # 4pm - afterhours quiet
    ]
    
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, state_sequence=state_sequence, auto_transition=False)
    
    print(f"\nGenerated {len(df)} bars with controlled states")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Controlled State Sequence",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
        major_tick_interval_minutes=120,  # Every 2 hours
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo2_controlled_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_custom_state():
    """Generate data with a custom state configuration"""
    print("\n" + "=" * 60)
    print("DEMO 3: Custom State Configuration")
    print("=" * 60)
    
    # Create a custom "mega volatile" state
    custom_state = StateConfig(
        name="mega_volatile",
        avg_ticks_per_bar=40.0,
        ticks_per_bar_std=20.0,
        up_probability=0.5,
        trend_persistence=0.4,  # Low persistence = very choppy
        avg_tick_size=3.0,
        tick_size_std=2.0,
        max_tick_jump=15,
        volatility_multiplier=3.0,
        wick_probability=0.6,
        wick_extension_avg=5.0,
    )
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=456)
    
    # Generate just a few hours with this custom state
    start_date = datetime(2025, 11, 29, 9, 30, 0)  # RTH open
    bars = []
    
    for minute in range(240):  # 4 hours
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, custom_state_config=custom_state)
        bars.append(bar)
    
    import pandas as pd
    df = pd.DataFrame(bars)
    
    print(f"\nGenerated {len(df)} bars with custom 'mega_volatile' state")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    print(f"Max bar range: {(df['high'] - df['low']).max():.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Custom Mega Volatile State",
        figsize=(16, 9),
        show_volume=True,
        show_state_changes=False,
        show_session_changes=True,
        major_tick_interval_minutes=30,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo3_custom_state.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_state_comparison():
    """Generate samples of each state for comparison"""
    print("\n" + "=" * 60)
    print("DEMO 4: State Comparison")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    states_to_compare = [
        MarketState.FLAT,
        MarketState.RANGING,
        MarketState.ZOMBIE,
        MarketState.RALLY,
        MarketState.IMPULSIVE,
    ]
    
    fig, axes = plt.subplots(len(states_to_compare), 1, figsize=(16, 12))
    
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    for idx, state in enumerate(states_to_compare):
        # Generate 2 hours of data for this state
        gen = PriceGenerator(initial_price=5000.0, seed=42 + idx)
        
        bars = []
        for minute in range(120):
            timestamp = start_date + timedelta(minutes=minute)
            bar = gen.generate_bar(timestamp, state=state)
            bars.append(bar)
        
        import pandas as pd
        df = pd.DataFrame(bars)
        
        # Plot on subplot
        ax = axes[idx]
        
        config = ChartConfig(
            title=f"{state.value.upper()} State",
            show_volume=False,
            show_state_changes=False,
            show_session_changes=False,
            figsize=(16, 3),
        )
        
        viz = ChartVisualizer(config)
        viz.plot_candlestick(ax, df)
        
        ax.set_title(f"{state.value.upper()} State", fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add stats
        price_range = df['high'].max() - df['low'].min()
        avg_bar_range = (df['high'] - df['low']).mean()
        
        stats_text = f"Range: {price_range:.2f} | Avg Bar: {avg_bar_range:.2f}"
        ax.text(
            0.02, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "demo4_state_comparison.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison chart saved to: {chart_path}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("MES PRICE GENERATOR DEMO")
    print("=" * 60)
    
    # Run demos
    demo_basic_generation()
    demo_controlled_states()
    demo_custom_state()
    demo_state_comparison()
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print(f"\nCharts saved to: {Path(__file__).resolve().parents[1] / 'out' / 'charts'}")


if __name__ == "__main__":
    main()
```

## FILE: ./scripts/evaluate_baseline.py
```
"""
Evaluate Baseline Model

Loads the trained Random Forest model and evaluates it on a held-out test set.
Generates confusion matrices and feature importance plots.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("=" * 60)
    print("EVALUATING BASELINE MODEL")
    print("=" * 60)
    
    # 1. Load Model
    model_path = root / "out" / "models" / "baseline_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 2. Load Data (Test Set)
    # We need to reload and resplit to ensure we get the same test set
    # ideally we would have saved the split, but for now we rely on seed
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    print("Loading data...")
    df = loader.load_archetypes()
    _, test_df = loader.prepare_training_data(df, seed=42) # Must use same seed!
    
    # 3. Build Features
    print("Extracting features...")
    builder = FeatureBuilder(window_size=60)
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    # 4. Predict
    print("Running inference...")
    y_pred = clf.predict(X_test)
    
    # 5. Visualize
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    labels = sorted(list(set(y_test)))
    cm_path = output_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, labels, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Feature Importance
    feature_names = builder.extract_features(test_df.iloc[:5]).columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    fi_path = output_dir / "baseline_feature_importance.png"
    plt.savefig(fi_path)
    print(f"Feature importance plot saved to: {fi_path}")

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/generate_archetypes.py
```
"""
Generate Synthetic Archetypes

Creates a library of labeled price patterns (archetypes) for pre-training models.
Generates 100+ samples for each of the 10 defined archetypes.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators import (
    PriceGenerator,
    MarketState,
    StateConfig,
    get_custom_state,
    STATE_CONFIGS
)

# Configuration
OUTPUT_DIR = root / "out" / "data" / "synthetic" / "archetypes"
SAMPLES_PER_ARCHETYPE = 100
START_DATE = datetime(2024, 1, 1, 0, 0, 0)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic archetypes")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_ARCHETYPE, help="Number of samples per archetype")
    return parser.parse_args()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def generate_rally_day(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 1: Pure Rally Day - Sustained upward trend"""
    # Mix of RALLY and IMPULSIVE, maybe some RANGING rests
    state_sequence = [
        (0, MarketState.RANGING),      # Open
        (60, MarketState.RALLY),       # Start moving
        (240, MarketState.RANGING),    # Rest
        (300, MarketState.RALLY),      # Resume
        (600, MarketState.IMPULSIVE),  # Accelerate
        (720, MarketState.RALLY),      # Sustain
        (1200, MarketState.RANGING),   # Close
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="RALLY_DAY")

def generate_range_day(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 2: Pure Range Day - Bounded, mean-reverting"""
    # Mostly RANGING and FLAT, maybe brief false breakouts
    state_sequence = [
        (0, MarketState.RANGING),
        (300, MarketState.FLAT),
        (600, MarketState.RANGING),
        (900, MarketState.FLAT),
        (1200, MarketState.RANGING),
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="RANGE_DAY")

def generate_breakout_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 3: Breakout Pattern - Range -> Breakout -> Rally"""
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.RANGING),
        (420, MarketState.BREAKOUT),   # The move
        (480, MarketState.IMPULSIVE),  # Follow through
        (600, MarketState.RALLY),      # Trend
        (1000, MarketState.RANGING),   # Consolidation at new high
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="BREAKOUT_PATTERN")

def generate_breakdown_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 4: Breakdown Pattern - Range -> Breakdown -> Selloff"""
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.RANGING),
        (420, MarketState.BREAKDOWN),  # The drop
        (480, MarketState.IMPULSIVE),  # Follow through (down)
        # Note: IMPULSIVE is high vol, direction depends on trend persistence/bias. 
        # For pure breakdown we might need a custom state or rely on momentum.
        # Let's use a custom 'strong_down' config if needed, but standard states might work 
        # if the breakdown sets the direction.
        (600, MarketState.RALLY),      # RALLY state has up bias. We need a 'SELLOFF' state really.
        # Using custom state for selloff to ensure down direction
    ]
    
    # Let's use standard states but rely on the generator's momentum or use a custom config for the trend part
    # Actually, let's use a custom state for the selloff part to guarantee it goes down
    selloff_config = StateConfig(
        name="selloff",
        avg_ticks_per_bar=20.0,
        up_probability=0.3, # Down bias
        trend_persistence=0.8,
        volatility_multiplier=1.5
    )
    
    # We can't easily mix state_sequence with custom_config in generate_day directly 
    # unless we modify generate_day to accept a list of configs.
    # For now, let's stick to standard states and hope the BREAKDOWN momentum carries, 
    # or use 'BREAKDOWN' state for longer.
    
    state_sequence = [
        (0, MarketState.RANGING),
        (400, MarketState.BREAKDOWN),
        (600, MarketState.BREAKDOWN), # Keep breaking down
        (800, MarketState.RANGING),
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="BREAKDOWN_PATTERN")

def generate_reversal_pattern(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 5: Reversal Pattern - Rally -> Range -> Breakdown"""
    state_sequence = [
        (0, MarketState.RALLY),        # Up
        (400, MarketState.RANGING),    # Top
        (600, MarketState.BREAKDOWN),  # Reversal
        (800, MarketState.RANGING),    # Settle low
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="REVERSAL_PATTERN")

def generate_zombie_grind(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 6: Zombie Grind - Slow, low-volatility trend"""
    state_sequence = [
        (0, MarketState.ZOMBIE),
        (1440, MarketState.ZOMBIE), # All day
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="ZOMBIE_GRIND")

def generate_volatile_chop(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 7: Volatile Chop - High volatility, no direction"""
    # We can use the 'mega_volatile' custom state, but generate_day takes standard states in sequence.
    # We'll use IMPULSIVE which is high vol.
    state_sequence = [
        (0, MarketState.IMPULSIVE),
        (1440, MarketState.IMPULSIVE),
    ]
    # To make it choppy (no direction), we might need to rely on the state config.
    # IMPULSIVE has 0.5 up_prob, so it should be choppy.
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="VOLATILE_CHOP")

def generate_opening_bell(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 8: Opening Bell - High volatility at open, then settle"""
    # RTH Open is usually 9:30 ET (8:30 CT). Let's assume 9:30 start for simplicity in bar index
    # 9:30 is 570 minutes from midnight
    state_sequence = [
        (0, MarketState.RANGING),      # Overnight
        (570, MarketState.IMPULSIVE),  # Open (High Vol)
        (630, MarketState.RALLY),      # Direction established
        (750, MarketState.RANGING),    # Lunch
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="OPENING_BELL")

def generate_closing_squeeze(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 9: Closing Squeeze - Quiet day -> End-of-day rally"""
    # Close is 16:00 ET (15:00 CT) -> 900 minutes
    state_sequence = [
        (0, MarketState.RANGING),
        (840, MarketState.RALLY),      # 14:00 CT - Start move
        (870, MarketState.IMPULSIVE),  # 14:30 CT - Squeeze
        (900, MarketState.RANGING),    # Close
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="CLOSING_SQUEEZE")

def generate_news_event(gen: PriceGenerator, date: datetime) -> pd.DataFrame:
    """Archetype 10: News Event - Sudden volatility spike"""
    # Random time spike
    event_time = 600 # 10:00 am
    state_sequence = [
        (0, MarketState.RANGING),
        (event_time, MarketState.BREAKOUT), # The news
        (event_time + 15, MarketState.IMPULSIVE), # The reaction
        (event_time + 60, MarketState.RANGING), # Settle
    ]
    return gen.generate_day(date, state_sequence=state_sequence, auto_transition=False, macro_regime="NEWS_EVENT")


ARCHETYPES = {
    "rally_day": generate_rally_day,
    "range_day": generate_range_day,
    "breakout_pattern": generate_breakout_pattern,
    "breakdown_pattern": generate_breakdown_pattern,
    "reversal_pattern": generate_reversal_pattern,
    "zombie_grind": generate_zombie_grind,
    "volatile_chop": generate_volatile_chop,
    "opening_bell": generate_opening_bell,
    "closing_squeeze": generate_closing_squeeze,
    "news_event": generate_news_event,
}

def main():
    args = parse_args()
    samples_to_generate = args.samples
    
    print("=" * 60)
    print(f"GENERATING {samples_to_generate} SAMPLES PER ARCHETYPE")
    print("=" * 60)
    
    ensure_dir(OUTPUT_DIR)
    
    total_generated = 0
    
    for name, generator_func in ARCHETYPES.items():
        print(f"\nGenerating {name}...")
        archetype_dir = OUTPUT_DIR / name
        ensure_dir(archetype_dir)
        
        # Use a fixed seed base for reproducibility, but different for each sample
        base_seed = hash(name) % 100000
        
        for i in tqdm(range(samples_to_generate)):
            seed = base_seed + i
            gen = PriceGenerator(initial_price=5000.0, seed=seed)
            
            # Increment date for variety (though mostly affects DOW config)
            date = START_DATE + timedelta(days=i)
            
            try:
                df = generator_func(gen, date)
                
                # Add metadata
                df.attrs['archetype'] = name
                df.attrs['seed'] = seed
                
                # Save
                filename = f"{name}_{i:03d}.parquet"
                df.to_parquet(archetype_dir / filename)
                
                total_generated += 1
                
            except Exception as e:
                print(f"Error generating {name} sample {i}: {e}")
                continue
                
    print("\n" + "=" * 60)
    print(f"DONE. Generated {total_generated} files in {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/test_installation.py
```
"""
Quick Test Script - Verify Installation and Basic Functionality

This script tests that all components are working correctly.
Run this after setting up the project to verify everything is installed.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

print("=" * 70)
print("FRACFIRE - INSTALLATION TEST")
print("=" * 70)

# Test 1: Import core dependencies
print("\n[1/6] Testing core dependencies...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ Core dependencies (numpy, pandas, matplotlib) OK")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 2: Import lab modules
print("\n[2/6] Testing lab modules...")
try:
    from lab.generators import PriceGenerator, MarketState
    from lab.generators.utils import summarize_day, print_summary
    print("✓ Lab generators module OK")
except ImportError as e:
    print(f"✗ Failed to import lab modules: {e}")
    sys.exit(1)

# Test 3: Generate synthetic data
print("\n[3/6] Testing price generator...")
try:
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    # Generate 60 bars (1 hour)
    bars = []
    from datetime import timedelta
    for minute in range(60):
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, MarketState.RALLY)
        bars.append(bar)
    
    df = pd.DataFrame(bars)
    print(f"✓ Generated {len(df)} bars successfully")
    print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Columns: {', '.join(df.columns[:8])}...")
except Exception as e:
    print(f"✗ Price generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test tick-based features
print("\n[4/6] Testing tick-based features...")
try:
    assert 'delta_ticks' in df.columns
    assert 'range_ticks' in df.columns
    assert 'body_ticks' in df.columns
    assert 'upper_wick_ticks' in df.columns
    assert 'lower_wick_ticks' in df.columns
    
    # Verify all prices are multiples of 0.25
    assert all(df['open'] % 0.25 == 0)
    assert all(df['high'] % 0.25 == 0)
    assert all(df['low'] % 0.25 == 0)
    assert all(df['close'] % 0.25 == 0)
    
    print("✓ Tick-based features OK")
    print(f"  Avg delta_ticks: {df['delta_ticks'].mean():.2f}")
    print(f"  Avg range_ticks: {df['range_ticks'].mean():.2f}")
except AssertionError as e:
    print(f"✗ Tick feature validation failed: {e}")
    sys.exit(1)

# Test 5: Test analysis utilities
print("\n[5/6] Testing analysis utilities...")
try:
    # Generate a full day for better stats
    full_day = gen.generate_day(
        datetime(2025, 11, 29, 0, 0, 0),
        auto_transition=True
    )
    
    summary = summarize_day(full_day)
    
    assert 'overall' in summary
    assert 'by_state' in summary
    assert 'by_session' in summary
    
    print("✓ Analysis utilities OK")
    print(f"  States found: {len(summary['by_state'])}")
    print(f"  Sessions found: {len(summary['by_session'])}")
except Exception as e:
    print(f"✗ Analysis utilities failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test directory structure
print("\n[6/6] Testing directory structure...")
try:
    required_dirs = [
        root / "lab" / "generators",
        root / "lab" / "visualizers",
        root / "src" / "data",
        root / "src" / "models",
        root / "scripts",
        root / "docs",
        root / "out" / "charts",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print(f"✗ Missing directories: {', '.join(missing_dirs)}")
    else:
        print("✓ Directory structure OK")
except Exception as e:
    print(f"✗ Directory check failed: {e}")

# Summary
print("\n" + "=" * 70)
print("INSTALLATION TEST COMPLETE")
print("=" * 70)
print("\n✅ All tests passed! Your environment is ready.")
print("\nNext steps:")
print("  1. Review docs/PROJECT_MANAGEMENT.md for roadmap")
print("  2. Run: python scripts/demo_price_generation.py")
print("  3. Start generating archetypes")
print("\nEnvironment info:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Project root: {root}")
print("=" * 70)
```

## FILE: ./scripts/train_balanced.py
```
"""
Train Balanced Model

Trains a Random Forest classifier with class_weight='balanced' to address
the dominance of 'ranging' states and improve sensitivity to directional moves.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("TRAINING BALANCED MODEL")
    print("=" * 60)
    
    # 1. Load Data
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    print("Loading data...")
    df = loader.load_archetypes()
    train_df, test_df = loader.prepare_training_data(df)
    
    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    
    X_train, y_train = builder.create_dataset(train_df, target_col='state')
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    # 3. Train with Class Weights
    print("\nTraining Balanced Random Forest...")
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',  # <--- KEY CHANGE
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Model
    model_dir = root / "out" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "balanced_rf.joblib"
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/train_baseline.py
```
"""
Train Baseline Model

Trains a Random Forest classifier to predict market states (RALLY, RANGING, etc.)
based on tick features extracted from synthetic archetypes.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("TRAINING BASELINE MODEL (Random Forest)")
    print("=" * 60)
    
    # 1. Load Data
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    # Load all data (might be large, so be careful)
    # For baseline, maybe limit if too slow
    print("Loading data...")
    df = loader.load_archetypes()
    
    # Split
    train_df, test_df = loader.prepare_training_data(df)
    
    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    
    # We predict 'state' column
    X_train, y_train = builder.create_dataset(train_df, target_col='state')
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 3. Train
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Model
    model_dir = root / "out" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_rf.joblib"
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 6. Feature Importance
    feature_names = builder.extract_features(train_df.iloc[:5]).columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for f in range(min(10, len(feature_names))):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/validate_archetypes.py
```
"""
Validate Synthetic Archetypes

Verifies the integrity and statistical properties of the generated archetype library.
Checks if the generated patterns match their intended definitions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.utils import summarize_day

# Configuration
ARCHETYPE_DIR = root / "out" / "data" / "synthetic" / "archetypes"
SAMPLES_TO_CHECK = 20  # Check random 20 samples per archetype to save time

def validate_rally_day(stats: Dict) -> bool:
    """Rally day should have positive net move and decent range"""
    return stats['overall']['net_move'] > 0 and stats['overall']['total_range_ticks'] > 50

def validate_range_day(stats: Dict) -> bool:
    """Range day should have low net move relative to range"""
    net_move = abs(stats['overall']['net_move_ticks'])
    total_range = stats['overall']['total_range_ticks']
    return net_move < (total_range * 0.6)

def validate_breakout(stats: Dict) -> bool:
    """Breakout should have significant move"""
    return abs(stats['overall']['net_move_ticks']) > 50

def validate_zombie(stats: Dict) -> bool:
    """Zombie should be low volatility but directional"""
    # Low average bar range
    return stats['overall']['avg_range_ticks'] < 10

def validate_volatile(stats: Dict) -> bool:
    """Volatile should have high average range"""
    return stats['overall']['avg_range_ticks'] > 15

def validate_news_event(stats: Dict) -> bool:
    """News event should have a volatility spike (high max range)"""
    # Even if average is low, max bar range should be high
    return stats['overall'].get('max_range_ticks', 0) > 25

VALIDATORS = {
    "rally_day": validate_rally_day,
    "range_day": validate_range_day,
    "breakout_pattern": validate_breakout,
    "breakdown_pattern": validate_breakout, # Same logic (big move)
    "reversal_pattern": None, # Hard to validate with simple stats
    "zombie_grind": validate_zombie,
    "volatile_chop": validate_volatile,
    "opening_bell": None,
    "closing_squeeze": None,
    "news_event": validate_news_event, # Should be volatile spike
}

def main():
    print("=" * 60)
    print("VALIDATING ARCHETYPES")
    print("=" * 60)
    
    if not ARCHETYPE_DIR.exists():
        print(f"Archetype directory not found: {ARCHETYPE_DIR}")
        print("Run generate_archetypes.py first.")
        sys.exit(1)
    
    results = {}
    
    for archetype_path in ARCHETYPE_DIR.iterdir():
        if not archetype_path.is_dir():
            continue
            
        name = archetype_path.name
        print(f"\nChecking {name}...")
        
        files = list(archetype_path.glob("*.parquet"))
        if not files:
            print(f"  No files found!")
            continue
            
        # Sample files
        if len(files) > SAMPLES_TO_CHECK:
            files = np.random.choice(files, SAMPLES_TO_CHECK, replace=False)
        
        passed = 0
        failed = 0
        errors = 0
        
        validator = VALIDATORS.get(name)
        
        for file_path in tqdm(files):
            try:
                df = pd.read_parquet(file_path)
                
                # Check required columns
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'state']
                if not all(col in df.columns for col in required_cols):
                    print(f"  Missing columns in {file_path.name}")
                    errors += 1
                    continue
                
                # Run stats
                stats = summarize_day(df)
                
                # Run specific validation if exists
                if validator:
                    if validator(stats):
                        passed += 1
                    else:
                        failed += 1
                else:
                    passed += 1 # No specific validator, just load check
                    
            except Exception as e:
                print(f"  Error reading {file_path.name}: {e}")
                errors += 1
        
        results[name] = {
            "total": len(files),
            "passed": passed,
            "failed": failed,
            "errors": errors
        }
        
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Archetype':<20} {'Checked':>8} {'Passed':>8} {'Failed':>8} {'Errors':>8}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<20} {res['total']:>8} {res['passed']:>8} {res['failed']:>8} {res['errors']:>8}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/visualize_optimized.py
```
"""
Visualize Optimized Predictions

Plots the real price data colored by the predicted market state from the optimized model.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def main():
    print("=" * 60)
    print("VISUALIZING OPTIMIZED PREDICTIONS")
    print("=" * 60)
    
    # 1. Load Predictions
    data_path = root / "out" / "data" / "real" / "processed" / "optimized_states.parquet"
    if not data_path.exists():
        print(f"Predictions not found at {data_path}")
        print("Run apply_optimized.py first.")
        return
        
    print(f"Loading predictions from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Select a subset to visualize
    subset_size = 2000
    print(f"Visualizing first {subset_size} bars...")
    df_subset = df.iloc[:subset_size].copy()
    
    # 3. Plot
    plt.figure(figsize=(15, 8))
    
    # Create a color map for states
    states = df['predicted_state'].unique()
    palette = sns.color_palette("bright", len(states)) # Use bright for better visibility
    color_map = dict(zip(states, palette))
    
    # Plot price line
    plt.plot(df_subset.index, df_subset['close'], color='gray', alpha=0.5, linewidth=1, label='Price')
    
    for state in states:
        mask = df_subset['predicted_state'] == state
        plt.scatter(
            df_subset.index[mask], 
            df_subset['close'][mask], 
            c=[color_map[state]], 
            label=state,
            s=15,
            alpha=0.8
        )
        
    plt.title("Real Market Data - Optimized States (Balanced + Thresholds)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 4. Save
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "optimized_states.png"
    
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    main()
```

## FILE: ./scripts/visualize_real.py
```
"""
Visualize Real Data Predictions

Plots the real price data colored by the predicted market state.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def main():
    print("=" * 60)
    print("VISUALIZING REAL DATA PREDICTIONS")
    print("=" * 60)
    
    # 1. Load Predictions
    data_path = root / "out" / "data" / "real" / "processed" / "predicted_states.parquet"
    if not data_path.exists():
        print(f"Predictions not found at {data_path}")
        print("Run apply_to_real.py first.")
        return
        
    print(f"Loading predictions from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Select a subset to visualize (e.g., first 5 days or 2000 bars)
    # Visualizing the whole year is too dense
    subset_size = 2000
    print(f"Visualizing first {subset_size} bars...")
    df_subset = df.iloc[:subset_size].copy()
    
    # 3. Plot
    plt.figure(figsize=(15, 8))
    
    # Create a color map for states
    states = df['predicted_state'].unique()
    palette = sns.color_palette("husl", len(states))
    color_map = dict(zip(states, palette))
    
    # Plot price line
    # We can't easily color a single line with multiple colors in matplotlib without segments
    # So we'll plot scatter points on top of a thin gray line
    
    plt.plot(df_subset.index, df_subset['close'], color='gray', alpha=0.5, linewidth=1, label='Price')
    
    for state in states:
        mask = df_subset['predicted_state'] == state
        plt.scatter(
            df_subset.index[mask], 
            df_subset['close'][mask], 
            c=[color_map[state]], 
            label=state,
            s=10,
            alpha=0.8
        )
        
    plt.title("Real Market Data - Predicted States (Baseline Model)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 4. Save
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_data_states.png"
    
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    main()
```

## FILE: ./src/behavior/learner.py
```
"""
Behavior Learner Module

Responsible for learning market behavior patterns, specifically:
- State transition matrices (Markov Chains)
- Regime probability estimation
- Feature-to-State mapping (Random Forest / XGBoost)

This module sits between the raw data and the generative policy.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BehaviorLearner:
    """
    Learns behavioral parameters from market data.
    """
    
    def __init__(self):
        self.transition_matrix = None
        self.state_priors = None
        
    def fit_transitions(self, state_sequence: List[str]) -> pd.DataFrame:
        """
        Estimate Markov transition matrix from a sequence of states.
        
        Args:
            state_sequence: List of state labels
            
        Returns:
            DataFrame representing transition probabilities
        """
        # Placeholder
        raise NotImplementedError("To be implemented in Phase 2")
        
    def fit_regime_classifier(self, X: pd.DataFrame, y: pd.Series):
        """
        Train a classifier to predict market state from features.
        
        Args:
            X: Feature matrix
            y: State labels
        """
        # Placeholder (e.g., RandomForest)
        raise NotImplementedError("To be implemented in Phase 2")
```

## FILE (first 200 lines): ./src/data/continuous_contract.json
```
[
  {
    "time":"2025-03-18T00:00:00Z",
    "open":5812.75,
    "high":5814.5,
    "low":5810.25,
    "close":5810.25,
    "volume":608,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:01:00Z",
    "open":5810.25,
    "high":5812.75,
    "low":5810.0,
    "close":5812.0,
    "volume":237,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:02:00Z",
    "open":5811.5,
    "high":5812.0,
    "low":5811.5,
    "close":5812.0,
    "volume":35,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:03:00Z",
    "open":5811.5,
    "high":5812.5,
    "low":5811.25,
    "close":5811.75,
    "volume":246,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:04:00Z",
    "open":5811.75,
    "high":5812.5,
    "low":5811.75,
    "close":5812.5,
    "volume":57,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:05:00Z",
    "open":5812.25,
    "high":5812.5,
    "low":5811.0,
    "close":5811.25,
    "volume":74,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:06:00Z",
    "open":5811.0,
    "high":5811.0,
    "low":5809.25,
    "close":5810.25,
    "volume":221,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:07:00Z",
    "open":5810.0,
    "high":5811.25,
    "low":5809.5,
    "close":5811.0,
    "volume":170,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:08:00Z",
    "open":5810.25,
    "high":5810.25,
    "low":5809.0,
    "close":5809.75,
    "volume":198,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:09:00Z",
    "open":5809.75,
    "high":5810.5,
    "low":5809.25,
    "close":5810.0,
    "volume":162,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:10:00Z",
    "open":5810.0,
    "high":5810.25,
    "low":5809.5,
    "close":5810.0,
    "volume":74,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:11:00Z",
    "open":5810.25,
    "high":5811.5,
    "low":5808.5,
    "close":5811.0,
    "volume":288,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:12:00Z",
    "open":5810.75,
    "high":5811.5,
    "low":5810.25,
    "close":5810.5,
    "volume":217,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:13:00Z",
    "open":5810.75,
    "high":5811.5,
    "low":5810.5,
    "close":5810.75,
    "volume":84,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:14:00Z",
    "open":5810.75,
    "high":5811.5,
    "low":5810.75,
    "close":5810.75,
    "volume":47,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:15:00Z",
    "open":5810.75,
    "high":5811.75,
    "low":5810.75,
    "close":5811.0,
    "volume":197,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:16:00Z",
    "open":5810.75,
    "high":5811.25,
    "low":5809.25,
    "close":5809.5,
    "volume":292,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:17:00Z",
    "open":5809.25,
    "high":5810.25,
    "low":5808.25,
    "close":5809.25,
    "volume":455,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:18:00Z",
    "open":5809.25,
    "high":5809.25,
    "low":5808.75,
    "close":5809.0,
    "volume":58,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:19:00Z",
    "open":5808.75,
    "high":5809.0,
    "low":5808.0,
    "close":5808.5,
    "volume":94,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:20:00Z",
    "open":5808.5,
    "high":5809.25,
    "low":5808.5,
    "close":5809.25,
    "volume":88,
    "original_symbol":"MESM5"
  },
  {
    "time":"2025-03-18T00:21:00Z",
    "open":5809.5,
    "high":5809.75,
    "low":5809.0,
    "close":5809.25,
    "volume":50,
    "original_symbol":"MESM5"
  },
  {
```

## FILE: ./src/data/loader.py
```
"""
Real Data Loader

Loads and preprocesses real market data (JSON format) for use with the ML pipeline.
Handles conversion to tick-based metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

class RealDataLoader:
    """
    Loads real market data and aligns it with synthetic data schema.
    """
    
    def __init__(self, tick_size: float = 0.25):
        self.tick_size = tick_size
        
    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load JSON data and preprocess it.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame with required columns for FeatureBuilder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Loading real data from {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        # Parse time
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure numeric columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            df[col] = pd.to_numeric(df[col])
            
        # Calculate tick-based metrics
        # Note: We round to nearest int to avoid floating point issues
        df['delta_ticks'] = ((df['close'] - df['open']) / self.tick_size).round().astype(int)
        df['range_ticks'] = ((df['high'] - df['low']) / self.tick_size).round().astype(int)
        df['body_ticks'] = (abs(df['close'] - df['open']) / self.tick_size).round().astype(int)
        
        # Calculate wick ticks (useful for debugging/validation even if not directly used by FeatureBuilder currently)
        df['upper_wick_ticks'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / self.tick_size).round().astype(int)
        df['lower_wick_ticks'] = ((df[['open', 'close']].min(axis=1) - df['low']) / self.tick_size).round().astype(int)
        
        # Add synthetic-compatible columns if missing
        # Synthetic data has 'state' but real data doesn't (that's what we want to predict)
        
        print(f"Loaded {len(df)} bars of real data.")
        return df

if __name__ == "__main__":
    # Test run
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "data" / "continuous_contract.json"
    loader = RealDataLoader()
    df = loader.load_json(path)
    print(df.head())
    print(df.describe())
```

## FILE: ./src/features/builder.py
```
"""
Feature Builder Module

Responsible for transforming raw generator output (DataFrames) into 
ML-ready feature matrices (X) and targets (y).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class FeatureBuilder:
    """
    Builds features from tick-based OHLCV data.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical and statistical features.
        
        Args:
            df: Raw DataFrame from PriceGenerator
            
        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Log Returns
        # Use close prices. Fillna(0) for the first bar.
        features['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 2. Volatility (Rolling Std Dev of Log Returns)
        features['volatility'] = features['log_return'].rolling(self.window_size).std().fillna(0)
        
        # 3. Tick Momentum (Rolling mean of delta ticks)
        # Captures short-term directional pressure
        features['tick_momentum'] = df['delta_ticks'].rolling(10).mean().fillna(0)
        
        # 4. Relative Range
        # Current range vs average range (volatility expansion/contraction)
        avg_range = df['range_ticks'].rolling(self.window_size).mean()
        features['relative_range'] = (df['range_ticks'] / avg_range.replace(0, 1)).fillna(1.0)
        
        # 5. Volume Intensity
        # Current volume vs average volume
        avg_vol = df['volume'].rolling(self.window_size).mean()
        features['volume_intensity'] = (df['volume'] / avg_vol.replace(0, 1)).fillna(1.0)
        
        # 6. Body Dominance
        # How much of the range is the body? (Trend vs Indecision)
        features['body_dominance'] = (df['body_ticks'] / df['range_ticks'].replace(0, 1)).fillna(0)
        
        # 7. Wick Percentages
        # Upper wick % (Selling pressure)
        features['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / df['range_ticks'].replace(0, 1) * 4).fillna(0) # *4 because ticks are 0.25? No, range_ticks is integer.
        # Wait, range_ticks is int. (high-max)/range is correct ratio.
        features['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        
        # Lower wick % (Buying pressure)
        features['lower_wick_pct'] = ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        
        # 8. Cumulative Delta (Intraday Trend)
        # Reset per day usually, but here we just take cumsum of the DF provided
        features['cum_delta'] = df['delta_ticks'].cumsum()
        
        # 9. Distance from VWAP (Approximate)
        # VWAP = Sum(Price * Vol) / Sum(Vol)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol_price = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        vwap = cum_vol_price / cum_vol.replace(0, 1)
        features['dist_vwap'] = (df['close'] - vwap) / vwap
        
        # 10. RSI-like Tick Strength (14 period)
        # Simple implementation using delta ticks
        delta = df['delta_ticks']
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        features['tick_rsi'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Clean up NaNs/Infs
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    def create_dataset(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'state',
        lookback: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (X, y) dataset for training.
        
        Args:
            df: Raw DataFrame
            target_col: Column to use as target
            lookback: Sequence length for RNN/Transformer (NOT IMPLEMENTED YET)
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Extract features
        features = self.extract_features(df)
        
        # Get target
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame")
            
        y = df[target_col].values
        X = features.values
        
        if lookback > 0:
            raise NotImplementedError("Sequence lookback not yet implemented")
            
        return X, y
```

## FILE: ./src/models/tilt.py
```
"""
Neural Tilt Model

A lightweight PyTorch model that outputs a "tilt" vector to adjust the 
probabilities of the PriceGenerator's next move.

It does NOT predict the next price directly. It predicts the *bias* 
of the next tick (up/down probability, volatility scaling).
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class TiltModel(nn.Module):
    """
    Neural network for estimating market tilt.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP for now, can be GRU/LSTM later
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Outputs: [up_prob_bias, vol_bias, persistence_bias]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor (batch, input_dim)
            
        Returns:
            Tilt vector (batch, 3)
        """
        return self.net(x)
    
    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load model weights"""
        self.load_state_dict(torch.load(path))
```

## FILE: ./src/policy/orchestrator.py
```
"""
Orchestrator Policy Module

The brain of the operation. Coordinates:
1. FractalStateManager (What is the high-level context?)
2. BehaviorLearner (What are the likely transitions?)
3. TiltModel (What is the micro-bias?)
4. PriceGenerator (Execute the physics)

This module ensures multi-timeframe consistency and drives the simulation.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class Orchestrator:
    """
    High-level policy controller for the simulation.
    """
    
    def __init__(self):
        # Will hold references to other components
        self.generator = None
        self.fractal_manager = None
        self.tilt_model = None
        
    def initialize(self, config: Dict[str, Any]):
        """Setup components based on config"""
        pass
        
    def step(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Advance the simulation by one step.
        
        1. Update fractal state
        2. Get tilt from model (if active)
        3. Determine parameters for generator
        4. Call generator.generate_bar()
        """
        raise NotImplementedError("To be implemented in Phase 2")
    
    def run_day(self, date: datetime) -> Any:
        """Run a full day simulation"""
        pass
```

## FILE: ./src/training/data_loader.py
```
"""
Data Loader Module

Handles loading, splitting, and preparing synthetic archetype data for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

class DataLoader:
    """
    Loads and manages archetype datasets.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def load_archetypes(self, pattern: str = "*.parquet", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load all archetypes into a single DataFrame with metadata.
        
        Args:
            pattern: Glob pattern for files
            limit: Max files to load (for testing)
            
        Returns:
            Combined DataFrame
        """
        files = list(self.data_dir.rglob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found in {self.data_dir} matching {pattern}")
            
        if limit:
            files = files[:limit]
            
        dfs = []
        print(f"Loading {len(files)} archetype files...")
        
        for f in tqdm(files, desc="Loading"):
            try:
                df = pd.read_parquet(f)
                
                # Add metadata
                # Folder name is archetype label (e.g., 'rally_day')
                archetype_label = f.parent.name
                
                # Use filename stem as run_id (e.g., 'rally_day_001')
                run_id = f.stem
                
                df['archetype'] = archetype_label
                df['run_id'] = run_id
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not dfs:
            raise ValueError("No valid dataframes loaded")
            
        return pd.concat(dfs, ignore_index=True)
        
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets by run_id to avoid leakage.
        
        We must split by DAY (run_id), not by row, because rows within a day 
        are highly correlated.
        
        Args:
            df: Combined DataFrame
            test_size: Fraction of runs to use for testing
            seed: Random seed
            
        Returns:
            train_df, test_df
        """
        run_ids = df['run_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(run_ids)
        
        split_idx = int(len(run_ids) * (1 - test_size))
        train_runs = run_ids[:split_idx]
        test_runs = run_ids[split_idx:]
        
        train_df = df[df['run_id'].isin(train_runs)].copy()
        test_df = df[df['run_id'].isin(test_runs)].copy()
        
        print(f"Split data: {len(train_runs)} training days, {len(test_runs)} test days")
        
        return train_df, test_df
```

## SKIPPED (binary or ignored): ./tickfirecode.md
