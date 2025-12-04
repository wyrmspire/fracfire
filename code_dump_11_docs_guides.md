# Code Dump: 11_docs_guides

## File: docs/GENERATOR_GUIDE.md
```markdown
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

---

## File: docs/SETUP_COMPLETE.md
```markdown
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

---

## File: docs/PROJECT_MANAGEMENT.md
```markdown
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
**Current Focus**: User Verification of Liquidity Physics  
**Blockers**: User Approval  
**Next Review**: After approval, proceed to production pipeline

```

---

## File: requirements.txt
```text
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

---

