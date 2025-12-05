# Code Organization Guide

## Current Active Code (As of 2024-12-04)

This document helps distinguish between actively used code and legacy/experimental code that may need cleanup.

### Core Price Generation (Active)

**Primary Components:**
- `src/core/generator/engine.py` - Main physics-based price generator
- `src/core/generator/states.py` - Fractal state management (Day/Hour/Minute states)
- `src/core/generator/sampler.py` - Physics sampling for realistic tick movements
- `src/core/generator/fractal_planner.py` - ML-based trajectory planning (optional, requires TensorFlow)
- `src/core/generator/custom_states.py` - Custom extreme market states (Flash Crash, Melt Up, etc.)
- `src/core/generator/utils.py` - Utility functions

**Status:** ✅ **ACTIVE** - This is the current generation system used for all recent trades (12/4/2024)

**Key Features:**
- Tick-based simulation (0.25 tick size)
- Hierarchical state machines
- Session awareness (Asian/London/RTH)
- Configurable via `PhysicsConfig`

### Trade Detection System (Active)

**Primary Components:**
- `src/core/detector/engine.py` - Detection orchestration
- `src/core/detector/library.py` - Setup detection functions (ORB, EMA200, Breakout, Reversal, etc.)
- `src/core/detector/indicators.py` - Technical indicators (RSI, ATR, MACD, Bollinger Bands, Volume)
- `src/core/detector/features.py` - Feature extraction and trade evaluation
- `src/core/detector/models.py` - Data models (SetupEntry, SetupOutcome)
- `src/core/detector/styles.py` - Styling utilities
- `src/core/detector/sweep.py` - Liquidity sweep detection

**Status:** ✅ **ACTIVE** - Used for all trade setup detection

**Key Setup Types:**
1. `orb` - Opening Range Breakout
2. `ema200_continuation` - EMA200 continuation
3. `breakout` - Momentum breakout
4. `reversal` - Mean reversion
5. `opening_push` - First 30-min directional move
6. `moc` - Market on Close

### Visualization Scripts (Active - Recently Updated)

**Primary Scripts:**
- `scripts/place_trades_and_plot_3day.py` ✅ **UPDATED 12/4/2024**
  - Generates 3-day windows with trades
  - Now includes shaded profit/loss zones
  - SL/TP lines limited to trade duration
  - Exports summary.csv with exit_time
  
- `scripts/generate_plotly_representatives.py` ✅ **UPDATED 12/4/2024**
  - Interactive Plotly charts with HTML output
  - Shaded profit/loss zones
  - Duration-limited SL/TP lines
  - Requires: plotly, kaleido (optional)

- `scripts/plot_trade_setups.py` ✅ **UPDATED 12/4/2024**
  - Plots trades from scenario folders
  - Updated with new visualization style

- `scripts/place_trades_and_plot_3d.py` ✅ **ACTIVE**
  - 3D visualization (Time × Price × P&L)
  - Good for seeing cumulative P&L trajectory

**Status:** ✅ **ACTIVE** - Primary visualization tools

**Recent Changes (12/4/2024):**
- Added shaded boxes for profit (green) and stop loss (red) zones
- Limited SL/TP lines to actual trade duration (entry to exit)
- Added exit_time to CSV output
- Improved Plotly interactivity

### Analysis & Validation Scripts (Active)

**Comparison & Validation:**
- `scripts/compare_real_vs_generator.py` - Compare synthetic vs real data
- `scripts/validate_archetypes.py` - Validate generated archetypes
- `scripts/calibrate_generator.py` - Calibration utilities

**Analysis:**
- `scripts/analyze_drift.py` - Check for price drift issues
- `scripts/analyze_real_vs_synth.py` - Statistical comparison
- `scripts/analyze_regimes.py` - Regime detection
- `scripts/analyze_volatility.py` - Volatility analysis
- `scripts/analyze_wicks.py` - Wick distribution analysis

**Status:** ✅ **ACTIVE** - Used for validation and tuning

### Demo & Testing Scripts (Active)

- `scripts/demo_all_setups.py` - Demonstrates all setup types
- `scripts/demo_orb_setup.py` - ORB-specific demo
- `scripts/run_orb_trade_runner.py` - ORB trade simulation

**Status:** ✅ **ACTIVE** - Good for testing and demonstrations

## Legacy/Research Code (Potentially Unused)

### Lab Directory (`lab/`)

**Status:** ⚠️ **REVIEW NEEDED**

The `lab/` directory likely contains experimental or older code. Needs investigation:

- `lab/generators/` - May contain old generator implementations
- `lab/visualizers/` - May contain old visualization code

**Action Required:**
1. Compare with current `src/core/generator/` implementation
2. Identify duplicated functionality
3. Archive or remove if superseded by current code

### Old Visualization Scripts

**Potentially Outdated:**
- Check for any scripts in `scripts/` with dates or "old" in the name
- Compare functionality with recently updated scripts

**Status:** ⚠️ **NEEDS AUDIT**

### Data Pipeline (`src/data/`)

**Components:**
- `src/data/loader.py` - Data loading utilities
- `src/data/` - May contain data processing code

**Status:** ⚠️ **REVIEW NEEDED** - Determine if actively used

### ML/Models (`src/models/`, `src/behavior/`)

**Components:**
- `src/models/` - Neural network models
- `src/behavior/` - Behavior learning
- `src/features/` - Feature engineering
- `src/policy/` - Policy orchestration

**Status:** ⚠️ **REVIEW NEEDED** - Determine integration with current system

## Recommended Cleanup Actions

### Phase 1: Audit (Do First)
1. ✅ Document current active code (DONE - this file)
2. ⚠️ Review `lab/` directory contents
3. ⚠️ Check for duplicate implementations
4. ⚠️ Identify scripts with overlapping functionality

### Phase 2: Archive
1. Create `archive/` directory for old code
2. Move superseded implementations to archive
3. Update README to reflect current structure

### Phase 3: Document
1. ✅ Add inline documentation to generators (DONE - see PRICE_GENERATION_METHODOLOGY.md)
2. Add inline comments explaining key algorithms
3. Update docstrings for public APIs

### Phase 4: Deprecate
1. Mark deprecated functions with warnings
2. Provide migration paths to new code
3. Set removal timeline

## Dependencies Status

### Required (Currently Used)
```python
numpy>=1.24.0          # Core math
pandas>=2.0.0          # Data handling
matplotlib>=3.7.0      # Static charts
plotly>=5.18.0         # Interactive charts (NEW)
kaleido>=0.2.1         # Plotly PNG export (NEW)
scipy>=1.10.0          # Scientific computing
scikit-learn>=1.3.0    # ML utilities
```

### Optional (Feature-Specific)
```python
tensorflow>=2.13.0     # For FractalPlanner ML model (optional)
torch>=2.0.0           # For neural tilt models
```

### Development Only
```python
pytest>=7.4.0          # Testing
black>=23.0.0          # Formatting
flake8>=6.0.0          # Linting
```

## Import Conventions

### Current Active Imports

**Price Generation:**
```python
from src.core.generator.engine import PhysicsConfig, PriceGenerator
from src.core.generator.states import DayState, HourState, MinuteState
```

**Trade Detection:**
```python
from src.core.detector.library import (
    find_opening_orb_continuations,
    find_ema200_continuation,
    find_breakout,
    find_reversal,
    find_opening_push,
    find_moc,
)
from src.core.detector.models import SetupEntry, SetupOutcome
from src.core.detector.indicators import add_5m_indicators, IndicatorConfig
```

### ⚠️ Imports to Audit

If you see imports from these paths, check if they're still needed:
- `from lab.generators.*` - May be old generator code
- `from lab.visualizers.*` - May be old viz code
- Any imports with "old", "legacy", or "deprecated" in the path

## File Naming Conventions

### Current Active Files
- Use descriptive names: `place_trades_and_plot_3day.py`
- Include action in name: `generate_`, `analyze_`, `demo_`
- No version numbers in primary scripts

### Files to Review
- Files with numbers: `script_v1.py`, `script_v2.py`
- Files with dates: `generator_2024_01.py`
- Files with "test" but not in tests/: May be temporary
- Files with "old", "backup", "copy": Should be removed

## Questions to Ask When Reviewing Code

1. **Is this used by any active script?**
   - Search codebase for imports
   - Check git history for last use

2. **Does it duplicate current functionality?**
   - Compare with `src/core/generator/` and `src/core/detector/`
   - Check for similar function names

3. **Is it documented?**
   - If no docstring and no clear purpose → likely old

4. **Does it have tests?**
   - Untested code may be experimental

5. **Is it mentioned in README or docs?**
   - If not documented → may be superseded

## Suggestions from External Review (12/4/2024)

The following suggestions were made regarding the price generation system:

### ✅ Already Implemented
- Tick integrity (0.25 increments)
- Wick physics
- Fractal state management
- Session awareness
- Markov chain transitions

### ✅ Now Available as Optional Features (12/5/2024)
- **Fat-tailed distribution** (Student-t) - Enable with `use_fat_tails=True`
- **Volatility clustering** (GARCH-like) - Enable with `use_volatility_clustering=True`

**Important**: These are **optional configuration flags** that maintain backward compatibility. Default behavior is unchanged (uses Gaussian distribution).

### 🔧 Future Improvements to Consider

#### 1. **Fat-Tailed Distribution (Student-t)**
**Status**: ✅ **IMPLEMENTED AS OPTION**

Enable via `PhysicsConfig`:
```python
config = PhysicsConfig(
    use_fat_tails=True,      # Enable Student-t (default: False)
    fat_tail_df=3.0          # Degrees of freedom
)
```

**What changed:**
- Added `_sample_distribution()` method that switches between Gaussian and Student-t
- All three uses of `rng.normal()` now use this method
- Backward compatible: default is `use_fat_tails=False` (uses Gaussian)

**Benefit:** Captures "Black Swan" events and realistic market spikes

**Priority:** ✅ COMPLETE - Available as optional feature

#### 2. **Add Volatility Clustering (GARCH-like)**
**Status**: ✅ **IMPLEMENTED AS OPTION**

Enable via `PhysicsConfig`:
```python
config = PhysicsConfig(
    use_volatility_clustering=True,  # Enable (default: False)
    volatility_persistence=0.3       # Clustering strength
)
```

**What changed:**
- Added `_apply_volatility_clustering()` method
- Tracks `recent_volatility` as exponential moving average
- Updates after each bar generation
- Backward compatible: default is `use_volatility_clustering=False`

**Benefit:** More realistic volatility patterns

**Priority:** ✅ COMPLETE - Available as optional feature

#### 3. **Self-Exciting Processes (Hawkes Process)**
**Status**: 🔜 **NOT YET IMPLEMENTED**

**Current Issue:** Events don't trigger more events

**Proposed Implementation:**
```python
if abs(recent_move) > threshold:
    volatility_boost = event_intensity * decay_factor
```

**Benefit:** Captures cascade effects in real markets

**Priority:** MEDIUM

#### 4. **Dynamic Transition Probabilities**
**Status**: 🔜 **NOT YET IMPLEMENTED**

**Current Issue:** State transition probabilities are fixed

**Proposed Enhancement:** Make probabilities context-aware:
```python
if recent_vix_proxy > 30:
    increase_probability(HourState.IMPULSIVE)
```

**Benefit:** More realistic regime shifts

**Priority:** LOW - Current fixed probabilities work well

#### 5. **Hybrid GAN Approach**
**Status**: 🔜 **NOT YET IMPLEMENTED**

**Concept:** Use current generator for structure, GAN for texture

**Workflow:**
1. Generate clean structured data with current system
2. Train GAN on real MES data
3. Apply GAN as "style transfer" to add realistic noise

**Benefit:** Best of both worlds

**Priority:** LOW - Future research direction

## Summary

### Current System Status
- ✅ Price generation is sophisticated and working well
- ✅ Trade detection is comprehensive (6 setup types)
- ✅ Visualization just upgraded with shaded zones
- ✅ Documentation added (PRICE_GENERATION_METHODOLOGY.md)

### Immediate Action Items
- ⚠️ Audit `lab/` directory for old code
- ⚠️ Review and document data pipeline usage
- ⚠️ Check for duplicate implementations

### Future Enhancements (Optional)
- Consider fat-tailed distributions (Student-t) for more realistic spikes
- Add volatility clustering
- Implement self-exciting processes

---

**Last Updated:** 2024-12-04  
**Author:** FracFire Development Team
