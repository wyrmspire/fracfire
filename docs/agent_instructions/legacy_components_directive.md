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
