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
