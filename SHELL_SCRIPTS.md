# Fracfire Shell Scripts Guide

This document describes all available shell scripts in the fracfire project.

## Overview

The project includes both **shell scripts** for git operations and documentation generation, and **Python scripts** for data generation, analysis, and ML training.

## Shell Scripts

### `printcode.sh`
**Purpose:** Generate code dump files for documentation and LLM context loading.

**Usage:**
```bash
bash printcode.sh
```

**Output:** Creates markdown files named `code_dump_*.md` in the project root, organized by functionality:
- `code_dump_01_core_generator.md` - Price generator engine
- `code_dump_02_core_detector.md` - Setup detection and features
- `code_dump_03_data_loaders.md` - Data loading utilities
- `code_dump_04_ml_features.md` - ML feature builders
- `code_dump_05_ml_training.md` - Training pipelines
- `code_dump_06_ml_models.md` - Model definitions
- `code_dump_07_agent_modules.md` - Agent modules
- `code_dump_08_runner_scripts.md` - Setup runners
- `code_dump_09_demo_scripts.md` - Demonstration scripts
- `code_dump_10_docs_core.md` - Core documentation
- `code_dump_11_docs_guides.md` - Guide documentation
- `code_dump_12_agent_instructions.md` - Agent instructions

**Dependencies:** Python 3.12, `.venv312` virtual environment

---

### `gitr.sh`
**Purpose:** Git commit and push with automatic rebase on conflicts.

**Usage:**
```bash
bash gitr.sh "Your commit message here"
bash gitr.sh  # Uses default: "chore: update"
```

**Behavior:**
1. Adds all changes (`git add -A`)
2. Commits with provided message
3. Pushes to remote
4. If push fails, automatically attempts `git pull --rebase --autostash`
5. Retries push after rebase

**Features:**
- Automatic branch detection
- Upstream branch auto-configuration
- Error messages if manual intervention needed

---

### `gitp.sh`
**Purpose:** Hard-reset local branch to match remote (destructive operation).

**Usage:**
```bash
bash gitp.sh
```

**Behavior:**
1. Shows current branch
2. Sets upstream if not already set
3. Fetches latest from remote
4. Hard-resets to upstream (discards all local changes)
5. Confirms completion

**⚠️ WARNING:** This command discards all local changes. Use with caution!

---

## Python Scripts

### Core Runner Scripts

#### `scripts/run_orb_trade_runner.py`
**Purpose:** Generate synthetic 1-minute price data and detect Opening Range Breakout (ORB) trades.

**Usage:**
```bash
python scripts/run_orb_trade_runner.py \
  --seed 100 \
  --max-iterations 5 \
  --days-per-iter 1 \
  --or-minutes 30 \
  --buffer-ticks 2 \
  --min-drive-rr 0.5 \
  --max-counter-rr 0.3 \
  --target-rr 2.0 \
  --out-dir out/my_run
```

**Key Features:**
- 1-minute candle generation using synthetic price generator
- ORB detection with 5-minute decision anchors
- Feature extraction (30+ ML-friendly metrics)
- Bad-trade injection (hesitation/chase variants)
- Auto-relaxing knobs if no trades found
- Full JSON/CSV output with diagnostics

**Outputs:**
- `synthetic_1m.csv` - 1-minute OHLCV data
- `synthetic_5m.csv` - 5-minute OHLCV data (resampled)
- `summary.json` - Entries, outcomes, features, diagnostics
- `trades_features.csv` - Feature matrix for ML training

---

#### `scripts/plot_trade_setups.py`
**Purpose:** Visualize 1-minute candlesticks with trade overlays.

**Usage:**
```bash
python scripts/plot_trade_setups.py --data-dir out/my_run
```

**Output:** `trade_plot.png` showing:
- 1-minute candlesticks (wicks + bodies)
- Entry markers (green=win, red=loss, yellow=open)
- Stop/target levels (dashed lines)
- Trade annotations (kind, R-multiple)

---

### Analysis & Validation Scripts

#### `scripts/test_installation.py`
**Purpose:** Verify all dependencies and module imports are working.

**Usage:**
```bash
python scripts/test_installation.py
```

**Tests:**
1. Core dependencies (numpy, pandas, matplotlib)
2. Core generator and detector modules
3. Price generator functionality
4. Tick-based features
5. Analysis utilities
6. Directory structure

---

#### `scripts/compare_real_vs_generator.py`
**Purpose:** Compare real MES data against synthetic generator output.

**Usage:**
```bash
python scripts/compare_real_vs_generator.py \
  --week-days 14 \
  --timeframes 1min,4min,15min,1H,4H,1D
```

**Outputs:** Side-by-side candlestick charts at multiple timeframes

---

#### `scripts/analyze_real_vs_synth.py`
**Purpose:** Compute statistical comparisons (returns, ranges, wicks, volatility).

**Usage:**
```bash
python scripts/analyze_real_vs_synth.py \
  --week-days 14 \
  --timeframes 1min,4min,15min,1H,4H,1D
```

**Outputs:** CSV files with side-by-side statistics

---

### Generator & Demo Scripts

#### `scripts/demo_custom_states.py`
**Purpose:** Visualize extreme and specialized market states.

**Usage:**
```bash
python scripts/demo_custom_states.py
```

#### `scripts/demo_enhanced_features.py`
**Purpose:** Demonstrate feature extraction pipeline.

**Usage:**
```bash
python scripts/demo_enhanced_features.py
```

#### `scripts/demo_price_generation.py`
**Purpose:** Simple demo of price generation.

**Usage:**
```bash
python scripts/demo_price_generation.py
```

---

## Comprehensive Testing

### `scripts_test.sh`
**Purpose:** Run all syntax checks and import validation.

**Usage:**
```bash
bash scripts_test.sh
```

**Tests:**
- Shell script syntax (bash -n)
- Python script syntax (py_compile)
- Module imports (python -c import)

**Output:**
```
✓ printcode.sh (syntax OK)
✓ gitr.sh (syntax OK)
✓ gitp.sh (syntax OK)
✓ run_orb_trade_runner.py (syntax OK)
...
TEST RESULTS: 12 passed, 0 failed
✅ All tests passed!
```

---

## Module Structure (Updated for Current Build)

All imports have been updated to use the new modular structure:

```
src/
├── core/
│   ├── generator/      # Price generation engine
│   │   ├── engine.py
│   │   ├── states.py
│   │   ├── custom_states.py
│   │   └── utils.py
│   └── detector/       # Setup detection & features
│       ├── models.py
│       ├── indicators.py
│       ├── library.py
│       ├── features.py
│       └── engine.py
├── data/              # Data loading
├── ml/                # ML training & models
│   ├── features/
│   ├── training/
│   └── models/
└── agent/            # AI agent modules
```

---

## Execution Environment

All scripts expect:
- **Python 3.12** via `.venv312` virtual environment
- **Project root:** `/C/fracfire` (Windows) or `/path/to/fracfire` (Linux/Mac)
- **Working directory:** Project root

To activate the environment:
```bash
source .venv312/Scripts/activate  # Linux/Mac
.venv312\Scripts\activate.bat     # Windows
```

---

## Quick Start Examples

### Generate and analyze a single ORB trade:
```bash
cd /C/fracfire
.venv312/Scripts/python.exe scripts/run_orb_trade_runner.py \
  --seed 42 \
  --max-iterations 3 \
  --out-dir out/quick_test
.venv312/Scripts/python.exe scripts/plot_trade_setups.py \
  --data-dir out/quick_test
```

### Verify everything is working:
```bash
bash scripts_test.sh
bash printcode.sh
```

### Generate code dumps for LLM context:
```bash
bash printcode.sh
# Creates code_dump_*.md files in project root
```

---

## Troubleshooting

### "No module named 'src.core.generator'"
- Ensure you're running from project root: `/C/fracfire`
- Check that `src/core/generator/__init__.py` exists
- Verify `.venv312` is activated

### Shell script permission denied
```bash
chmod +x *.sh scripts/*.sh
```

### Python import errors
Run the test script to diagnose:
```bash
bash scripts_test.sh
python scripts/test_installation.py
```

---

## Configuration

All scripts use reasonable defaults but accept CLI arguments for customization:
- `--seed`: Random seed for reproducibility
- `--out-dir`: Output directory
- `--max-iterations`: Max generation attempts
- `--inject-hesitation`, `--inject-chase`: Bad-trade variants

See individual script headers for full argument lists.

---

**Last Updated:** December 3, 2025
**Status:** ✅ All scripts tested and working with current build
