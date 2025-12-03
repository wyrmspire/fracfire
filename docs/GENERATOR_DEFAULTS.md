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
