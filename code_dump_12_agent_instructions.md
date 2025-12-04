# Code Dump: 12_agent_instructions

## File: docs/agent_instructions/phase_1_tasks.md
```markdown
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

---

## File: docs/agent_instructions/legacy_components_directive.md
```markdown
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

---

## File: docs/agent_instructions/style_rules.md
```markdown
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

---

## File: docs/agent_instructions/bootstrapping_checklist.md
```markdown
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

---

## File: docs/agent_instructions/file_replacement_rules.md
```markdown
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

---

## File: docs/agent_instructions/reverse_engineering_directive.md
```markdown
# 🧬 Reverse-Engineering the Price Generator from Real Data

## 🔍 Context: What went wrong

* The current `PriceGenerator` produces **unrealistic multi-week behavior**:
  * On 15m charts over a month, price **stair-steps almost monotonically upward** to extreme levels.
  * Macro “fractal” regimes are effectively **too long and too biased** (trend states persist for days instead of a few hours).
* Real MES / ES behavior is more like:
  * A few strong hours after the open
  * A slower, choppy/muted middle
  * A distinct closing phase
  * Up and down swings over weeks, not a one-way staircase.

Your mission is to **use real market data to reverse engineer the generator**, so that multi-week 15m/1h structure looks like real futures, not a cartoon.

We will **not** immediately add more states.
First, we will **calibrate and re-structure** the ones we have.

---

## 🎯 High-Level Objective

1. Use **real MES/ES data** to discover recurring **2-hour regimes** (patterns).
2. Map those regimes onto our **Day / Hour / Minute states** and adjust:
   * State parameters (bias, volatility, persistence)
   * Transition structure (what kinds of hours follow what)
3. Re-run the generator to produce **3 months of synthetic data**.
4. Render a **15m candlestick chart** and **ask the human for visual approval** before doing any new pretraining.

---

## 1️⃣ Use 2-Hour Windows as Building Blocks

We care most about **2-hour regimes** that match how the user trades and experiences the market.

You must:

1. Work with real 1m (or similar) MES/ES data over a decent period (e.g. a few recent months).
2. Slice the data into overlapping **2-hour patches**:
   * 2 hours **after RTH open** (e.g. 9:30–11:30 ET)
   * 2 hours **before RTH close** (e.g. 14:00–16:00 ET)
   * Multiple 2-hour windows in between (late morning, lunch, early afternoon)
3. For each 2-hour patch, compute a **feature fingerprint** capturing:
   * Net move and drift (directionality)
   * Volatility (avg & max range per bar)
   * Monotonicity (how many bars go with the main move)
   * “Choppiness” (number of swings / reversals)
   * Wickiness vs body (spiky vs smooth)
   * Volume level and volume skew (front-loaded, flat, back-loaded)

These fingerprints will be the basis for comparing patches.

---

## 2️⃣ Cluster Similar 2-Hour Patches (“Match the Patch”)

Now you must **play match-the-patch**:

1. Treat each 2-hour fingerprint as a point in feature space.
2. Use a clustering method (e.g. k-means or similar) to group patches into **regimes**:
   * Identify clusters such as:
     * “Open impulse drive”
     * “Open fake-out then revert”
     * “Lunch drift / low-vol chop”
     * “Symmetric range”
     * “Slow grind up”
     * “Slow bleed down”
     * “Late-day ramp / squeeze”
     * “Late-day liquidation / selloff”
3. For each cluster:
   * Inspect a few example 2-hour windows
   * Give the cluster a **human-readable name** and description
   * Record its typical stats (net move, vol, choppiness, wickiness, volume profile)

These clusters are our **empirical 2-hour states**.

---

## 3️⃣ Map 2-Hour Regimes to Our State Hierarchy

Now integrate these empirical regimes into the existing architecture:

1. Map each cluster to:
   * A **DayState** flavor (trend_day, range_day, quiet_day, etc.)
   * One or more **HourState** flavors (impulse, consolidation, retracement, reversal, choppy, etc.)
   * A compatible **MarketState** / minute-level mix (rally, ranging, breakdown, flat, zombie, impulsive, etc.)
2. Adjust:
   * **State parameters**:
     * For trending regimes: directional_bias, volatility_mult, trend_persistence, up_probability (but remember, direction can be up *or* down).
     * For choppy/lunch regimes: lower volatility, more mean reversion, smaller monotonicity, more swing reversals.
   * **Transition structure**:
     * RTH open 2-hour block should mostly sample from “open” clusters (impulse, fakeout, early trend).
     * Midday blocks should mostly sample from “lunch / chop / drift” clusters.
     * Final 2-hour blocks should sample from “close continuation / close reversal / squeeze / liquidation” clusters.
3. Ensure both **bullish and bearish** versions of trending regimes exist:
   * Do not encode “always up” physics.
   * Allow the same structural state to have up or down direction depending on day bias.

Goal: over many days, the generator should produce **realistic mixtures** of these 2-hour regime types, not endless trend stairs.

---

## 4️⃣ Neutralize Long-Term Drift

The current generator’s month-long staircase means there is a **net positive bias** in the micro states.

You must:

1. Introduce a **day_direction** sign (up or down) at the day level.
   * For each day, draw a direction (e.g. +1 / −1 with roughly balanced frequencies, or slightly bullish).
   * Mirror the bias of trend states based on this sign so we get both trend-up and trend-down days.
2. Adjust per-state parameters (`up_probability`, directional_bias) so that:
   * Across all states and realistic regime mixes, the **expected drift per bar over months is near zero**.
   * The generator can create up-months and down-months, not just one-way ramps.

You must ensure that multi-month 15m charts do **not** look like monotone staircases.

---

## 5️⃣ Generate 3 Months of Synthetic Data and Visualize

After you have:
* Integrated 2-hour regimes and transition logic
* Calibrated state parameters so drift is reasonable

You must:

1. Use `PriceGenerator` to generate **3 months of continuous 1m synthetic data**.
2. Resample to **15-minute candles** (OHLCV).
3. Use `ChartVisualizer` to save a **3-month 15m candlestick chart** to the charts folder.

This chart should show:
* Realistic swings over weeks (up / down legs)
* Mixed days (trend days, range days, chop)
* Recognizable open, midday, and close behavior

---

## 6️⃣ Ask for Human Verification *Before* Any New Pretraining

**Do not** retrain or pretrain any ML models until the human signs off.

After generating the 3-month chart, you must:

1. Present the 15m 3-month chart(s) to the human.
2. Explicitly ask:

> “Does this 3-month synthetic 15m chart look realistic enough to you to use for pretraining?
> If not, tell me what still looks off (e.g. trends too clean, lunch too active, closes too quiet, etc.), and I will tweak the generator physics again.”

3. Only after receiving human approval may you:
   * Generate a fresh set of synthetic data
   * Rebuild archetypes if needed
   * And **then** run any pretraining / retraining scripts.

```

---

