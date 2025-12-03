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
