# Midnight Strategy Experiment Results

## Overview
We tested 6 variations of trading strategies using a CNN trained on synthetic data and tested on real market data.

## Strategy 1: 1:1 Risk/Reward (ATR Based)
- **Synth Test EV:** +0.08 R
- **Real Test EV:** +0.06 R
- **Verdict:** Marginally profitable.

## Strategy 2: 1:2 Risk/Reward (Target=2*ATR, Stop=1*ATR)
- **Synth Test EV:** +0.55 R
- **Real Test EV:** +0.68 R
- **Verdict:** **Highly Profitable**. The 1:2 ratio captures the trend effectively while capping risk. This is the best performing strategy.

## Strategy 3: Dynamic Risk (Stop=0.66*Range, Target=1.4*Stop)
- **Synth Test EV:** +0.25 R
- **Real Test EV:** +0.20 R
- **Verdict:** Profitable, but fewer trades triggered due to tighter constraints.

## Strategy 4: Time-Based Exit (Hold until 10:30)
- **Synth Test EV:** +0.99 R (Very High)
- **Real Test EV:** -9.20 R (Large Loss)
- **Verdict:** **Dangerous on Real Data**. Accuracy is good (53%), but uncapped risk leads to massive losses on wrong days. Real markets have "fat tails" that punish this approach.

## Strategy 5: Box Breakout (Real Data Only)
- **Setup:** Midnight-05:00 Range (Scaled 0.66). Breakout entries with 1.4R Target.
- **Base Performance (No AI):** +60.57 R (EV +0.03 R). **Profitable**.
- **AI Performance:** The CNN could not improve upon the base strategy (no high-confidence signals found).
- **Verdict:** The mechanical strategy itself has a positive edge on real data.

## Strategy 6: Reversal Pattern (Real Data Only)
- **Setup:** 15m Reversal (Swing Failure + Break). 10m confirmation. 1.4R Target.
- **Base Performance (No AI):** -58.32 R (EV -0.02 R). **Unprofitable**.
- **AI Performance:** The CNN could not improve performance.
- **Verdict:** This specific pattern does not have an edge on this dataset.

## Conclusion
The **1:2 Risk/Reward strategy (Strategy 2)** is the most robust, showing consistent profitability on both synthetic and real data. The **Box Breakout (Strategy 5)** also showed a mechanical edge on real data without AI assistance.
