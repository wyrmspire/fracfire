# Fracfire Agent Summary (2025-12-04)

This document explains the agent’s capabilities/tools, what we implemented today, and how the new scripts and charts work.

## Tools & Capabilities
- **File editing:** Patch existing files and create new ones within the workspace.
- **Terminal execution:** Run Python scripts and other commands in the project’s venv (`.venv312`).
- **Plotly visualization:** Generate interactive HTML candlestick charts and open them in the browser.
- **Pandas processing:** Load real market minute data (`data/continuous_contract.json`), filter windows, aggregate to timeframes.
- **PyTorch synthesis (GPU if available):** Generate synthetic minute prices quickly on CUDA/CPU and align with real trading-hour timestamps.
- **Project scripts:** Leverage existing calibration helpers (e.g., `scripts/calibrate_generator.py`) and add new utilities.

## What We Did Today
1. **Canonical comparison flow**
   - Implemented a clean script to compare real vs synthetic prices over the last 3 months.
   - Ensured synthetic generator produces data only for real timestamps (no weekends or off-hours that need post-filtering).
   - Aggregated both datasets to 15m by default for lighter, responsive charts.

2. **Robust Plotly chart**
   - Two stacked candlestick charts: top = real, bottom = synthetic.
   - Fixed axis behavior and timeframe switching (15m, 1h, 4h) via Plotly updatemenus.
   - Enabled consistent vertical zoom on both charts and reset ranges per timeframe.

3. **Amplitude calibration tools**
   - Added a `--vol-scale` parameter to control synthetic noise amplitude.
   - Printed 15m mean candle range metrics to quantify amplitude mismatch.
   - Created a sweep script to recommend a volatility scale that best matches real amplitude.

## How It Works

### Real vs Synthetic Comparison: `scripts/compare_real_synth.py`
- **Load real (last 90 days):**
  - Reads `data/continuous_contract.json`, sorts by time, filters to the latest 90 days.
- **Generate synthetic on real timestamps:**
  - Uses PyTorch to create minute-level synthetic prices only for the exact real timestamps.
  - Scales noise by `--vol-scale` (default 1.0) to tune amplitude.
- **Aggregate:**
  - Resamples to OHLCV for chosen timeframes (default display is 15m; controls allow 1h/4h).
  - Drops gaps automatically via resampling.
- **Plotly chart:**
  - Two stacked candlestick traces (real/synth) with sequential x indices to avoid weekend gaps.
  - Updatemenus buttons switch timeframes and reset both X/Y ranges to the selected TF baseline.
  - Native Plotly zoom allows stretching/shrinking vertically and horizontally.
- **Metrics:**
  - Prints: `15m mean range: real=... synth=... ratio=... (vol-scale=...)` to assess amplitude.

Run it:
```
bash
cd c:/fracfire
python scripts/compare_real_synth.py --vol-scale 0.8
```
- Saves `charts/compare_real_synth.html` and opens the browser.

### Volatility Scale Sweep: `scripts/sweep_volatility_scale.py`
- **Purpose:** Find a `vol-scale` value that makes synthetic candle amplitude match real (ratio ~ 1.0).
- **Process:**
  - Runs `compare_real_synth.py` for scales `[0.5 … 1.8]`.
  - Parses the printed 15m mean range metrics.
  - Picks the scale with ratio closest to 1.0.
  - Saves results to `charts/vol_scale_sweep.json`.

Run it:
```
bash
cd c:/fracfire
python scripts/sweep_volatility_scale.py
```

### Calibration Helpers: `scripts/calibrate_generator.py`
- Contains day-state tests and a gap-fill experiment used to validate generator behavior.
- Not modified today, but available for deeper tuning of generator parameters beyond simple amplitude scaling.

## Notes & Next Steps
- If sweep recommends a scale (e.g., `0.80`), we can set it as the default in `compare_real_synth.py`.
- Extend analysis beyond mean range: add ATR, percentile distributions, session-aware scaling (RTH vs Globex) to match intra-day amplitude profiles.
- Add presets or buttons for Y-scale multipliers if desired (currently native zoom + reset per timeframe covers most needs).

## Quick Commands
- Compare (15m default):
```
bash
python scripts/compare_real_synth.py
```
- Compare with tuned amplitude:
```
bash
python scripts/compare_real_synth.py --vol-scale 0.8
```
- Sweep volatility scale:
```
bash
python scripts/sweep_volatility_scale.py
```