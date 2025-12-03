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
