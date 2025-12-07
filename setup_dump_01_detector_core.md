# Setup Dump: 01_detector_core

## File: src/core/detector/__init__.py
```python
"""Detector module: setup detection and feature extraction."""

from .models import SetupEntry, SetupOutcome, Direction, SetupKind
from .indicators import IndicatorConfig, DecisionAnchorConfig, add_1m_indicators, add_5m_indicators, mark_decision_points_1m
from .features import compute_trade_features, BadTradeConfig, inject_bad_trade_variants

__all__ = [
    'SetupEntry',
    'SetupOutcome',
    'Direction',
    'SetupKind',
    'IndicatorConfig',
    'DecisionAnchorConfig',
    'add_1m_indicators',
    'add_5m_indicators',
    'mark_decision_points_1m',
    'compute_trade_features',
    'BadTradeConfig',
    'inject_bad_trade_variants',
]

```

---

## File: src/core/detector/engine.py
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from .indicators import IndicatorConfig, add_5m_indicators
from .models import SetupOutcome
from . import library


@dataclass
class SetupFamilyConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupConfig:
    """Top-level config for all setup families and indicators."""

    indicator_cfg: IndicatorConfig = field(default_factory=IndicatorConfig)
    orb: SetupFamilyConfig = field(default_factory=lambda: SetupFamilyConfig(enabled=True))
    level_scalp: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    ema20_vwap_revert: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    ema200_continuation: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    breakout: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    reversal: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    opening_push: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)
    moc: SetupFamilyConfig = field(default_factory=SetupFamilyConfig)


def run_setups(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    cfg: SetupConfig,
) -> List[SetupOutcome]:
    """Run all enabled setup families on the given data.

    Returns a flat list of SetupOutcome objects for all playbooks.
    """

    df_5m_ind = add_5m_indicators(df_5m, cfg.indicator_cfg)
    outcomes: List[SetupOutcome] = []

    if cfg.orb.enabled:
        outcomes.extend(
            library.run_orb_family(df_5m_ind, **cfg.orb.params)
        )

    # Placeholders for future families; they can be wired in as implemented
    if cfg.level_scalp.enabled and hasattr(library, "run_level_scalp_family"):
        outcomes.extend(
            library.run_level_scalp_family(df_5m_ind, **cfg.level_scalp.params)
        )

    if cfg.ema20_vwap_revert.enabled and hasattr(library, "run_ema20_vwap_revert_family"):
        outcomes.extend(
            library.run_ema20_vwap_revert_family(df_5m_ind, **cfg.ema20_vwap_revert.params)
        )

    if cfg.ema200_continuation.enabled:
        outcomes.extend(
            library.run_ema200_continuation_family(df_5m_ind, **cfg.ema200_continuation.params)
        )
    
    if cfg.breakout.enabled:
        outcomes.extend(
            library.run_breakout_family(df_5m_ind, **cfg.breakout.params)
        )
    
    if cfg.reversal.enabled:
        outcomes.extend(
            library.run_reversal_family(df_5m_ind, **cfg.reversal.params)
        )
    
    if cfg.opening_push.enabled:
        outcomes.extend(
            library.run_opening_push_family(df_5m_ind, **cfg.opening_push.params)
        )
    
    if cfg.moc.enabled:
        outcomes.extend(
            library.run_moc_family(df_5m_ind, **cfg.moc.params)
        )

    return outcomes

```

---

## File: src/core/detector/models.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

import pandas as pd

Direction = Literal["long", "short"]
SetupKind = Literal[
    "orb", 
    "level_scalp", 
    "ema20_vwap_revert", 
    "ema200_continuation",
    "breakout",
    "reversal",
    "opening_push",
    "moc"
]


@dataclass
class SetupEntry:
    """Generic setup entry usable by all playbooks.

    All concrete detectors (ORB, scalp, EMA reversion, continuation, etc.)
    should emit this type so downstream code and agents have a single
    interface.
    """

    time: pd.Timestamp
    direction: Direction
    kind: SetupKind
    entry_price: float
    stop_price: float
    target_price: float
    context: Dict[str, Any]


@dataclass
class SetupOutcome:
    """Evaluation of a setup entry against future bars."""

    entry: SetupEntry
    hit_target: bool
    hit_stop: bool
    exit_time: pd.Timestamp
    r_multiple: float
    mfe: float
    mae: float

```

---

## File: src/core/detector/indicators.py
```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IndicatorConfig:
    """Shared indicator knobs for 5m/1m data.

    This is intentionally minimal; we can extend as needed without
    touching setup logic.
    """

    ema_fast: int = 20
    ema_slow: int = 200
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_sma_period: int = 20


def add_5m_indicators(df_5m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 5m indicators (EMA fast/slow, VWAP, RSI, BB, ATR, MACD, Volume).

    All setup detectors should expect these columns if they need them.
    """
    df = df_5m.copy()
    
    # EMA indicators
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    # VWAP
    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
    
    # RSI
    df["rsi"] = _calculate_rsi(df["close"], cfg.rsi_period)
    
    # Bollinger Bands
    bb_sma = df["close"].rolling(window=cfg.bb_period).mean()
    bb_std = df["close"].rolling(window=cfg.bb_period).std()
    df["bb_upper"] = bb_sma + (cfg.bb_std * bb_std)
    df["bb_middle"] = bb_sma
    df["bb_lower"] = bb_sma - (cfg.bb_std * bb_std)
    
    # ATR (Average True Range)
    df["atr"] = _calculate_atr(df, cfg.atr_period)
    
    # MACD
    macd_line, signal_line, histogram = _calculate_macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    
    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=cfg.volume_sma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)
    
    return df


def add_1m_indicators(df_1m: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Attach common 1m indicators (EMA fast/slow, VWAP, RSI, BB, ATR, MACD, Volume)."""
    df = df_1m.copy()
    
    # EMA indicators
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    
    # VWAP
    vwap_num = (df["close"] * df["volume"]).cumsum()
    vwap_den = df["volume"].cumsum().replace(0, 1)
    df["vwap"] = vwap_num / vwap_den
    
    # RSI
    df["rsi"] = _calculate_rsi(df["close"], cfg.rsi_period)
    
    # Bollinger Bands
    bb_sma = df["close"].rolling(window=cfg.bb_period).mean()
    bb_std = df["close"].rolling(window=cfg.bb_period).std()
    df["bb_upper"] = bb_sma + (cfg.bb_std * bb_std)
    df["bb_middle"] = bb_sma
    df["bb_lower"] = bb_sma - (cfg.bb_std * bb_std)
    
    # ATR
    df["atr"] = _calculate_atr(df, cfg.atr_period)
    
    # MACD
    macd_line, signal_line, histogram = _calculate_macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    
    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=cfg.volume_sma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)
    
    return df


@dataclass
class DecisionAnchorConfig:
    """Defines decision points on 1m where setups are allowed to trigger.

    By default, we trigger at the close of each 5th minute (minute % 5 == 4).
    """

    modulo: int = 5
    offset: int = 4


def mark_decision_points_1m(df_1m: pd.DataFrame, cfg: DecisionAnchorConfig) -> pd.Series:
    """Return a boolean Series marking decision points on 1m timeline.

    True indicates this 1m bar is the decision boundary (e.g., 5-min close).
    """
    idx = df_1m.index
    # Assume DatetimeIndex with minute granularity
    marks = ((idx.minute % cfg.modulo) == cfg.offset)
    return pd.Series(marks, index=idx)


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range (ATR).
    
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = SMA of True Range over period
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def _calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

```

---

## File: src/core/detector/library.py
```python
"""Setup detection utilities (starting with Opening Range Breakout continuation).

All functions operate on OHLCV DataFrames and are designed to be fully
knob-driven so an agent can control behavior without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd

from .models import Direction, SetupEntry, SetupKind, SetupOutcome
from .indicators import DecisionAnchorConfig, mark_decision_points_1m


@dataclass
class ORBConfig:
    """Knobs for Opening Range Breakout + continuation on 5m bars.

    Times are assumed to be in the DataFrame index timezone (e.g. UTC with
    session_start corresponding to RTH open in that timezone).
    """

    session_start_hour: int = 14  # e.g. 14:30 UTC for 09:30 ET
    session_start_minute: int = 30
    or_minutes: int = 30          # length of opening range window
    buffer_ticks: float = 2.0     # extra ticks beyond OR high/low for a valid break
    tick_size: float = 0.25

    min_drive_rr: float = 0.5     # OR height multiples price should move in break direction
    max_counter_rr: float = 0.3   # max counter move against drive during OR

    stop_type: Literal["or", "bar"] = "or"  # stop below OR low or entry bar extreme
    target_rr: float = 2.0        # take-profit in multiples of risk
    max_hold_bars: int = 48       # safety cap after entry


@dataclass
class LevelScalpConfig:
    """Config for simple level-based scalp setups on 5m bars.

    This is intentionally simple for now: it looks for price
    touching/breaching a reference level with a rejecting wick.
    """

    tick_size: float = 0.25
    proximity_ticks: float = 4.0        # how close to level counts as touch
    stop_ticks: float = 8.0             # stop distance
    target_rr: float = 1.5              # scalp target in R
    max_hold_bars: int = 24


@dataclass
class ORBEntry:
    """Internal ORB-specific view; converted to generic SetupEntry."""

    time: pd.Timestamp
    direction: Direction
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    or_high: float = 0.0
    or_low: float = 0.0
    or_start: Optional[pd.Timestamp] = None
    or_end: Optional[pd.Timestamp] = None
    kind: SetupKind = "orb"


@dataclass
class ORBOutcome:
    """Evaluation of a single ORB entry over future bars."""

    entry: ORBEntry
    hit_target: bool
    hit_stop: bool
    exit_time: pd.Timestamp
    r_multiple: float  # realized R
    mfe: float          # max favorable excursion in price (price distance)
    mae: float          # max adverse excursion in price (price distance)


@dataclass
class ORBMissInfo:
    """Diagnostics for ORB attempts that did not qualify as entries.

    This helps explain *why* we didn't get a trade on a given day.
    """

    has_or_window: bool
    or_height: float
    drive_up: float
    drive_down: float
    up_rr: float
    down_rr: float
    reason: str


def explain_orb_missing(df_5m: pd.DataFrame, cfg: Optional[ORBConfig] = None) -> ORBMissInfo:
    """Return diagnostics describing why no ORB entry was found.

    Looks only at the OR window and drive metrics; does not inspect
    post-OR price action.
    """
    if cfg is None:
        cfg = ORBConfig()

    if df_5m.empty:
        return ORBMissInfo(
            has_or_window=False,
            or_height=0.0,
            drive_up=0.0,
            drive_down=0.0,
            up_rr=0.0,
            down_rr=0.0,
            reason="empty_5m_data",
        )

    or_slice = _find_session_or(df_5m, cfg)
    if or_slice is None or or_slice.empty:
        return ORBMissInfo(
            has_or_window=False,
            or_height=0.0,
            drive_up=0.0,
            drive_down=0.0,
            up_rr=0.0,
            down_rr=0.0,
            reason="no_or_window_for_session_start",
        )

    or_high = float(or_slice["high"].max())
    or_low = float(or_slice["low"].min())
    or_height = or_high - or_low
    or_closes = or_slice["close"]
    drive_up = float(or_closes.max() - or_closes.iloc[0])
    drive_down = float(or_closes.iloc[0] - or_closes.min())
    up_rr = drive_up / or_height if or_height > 0 else 0.0
    down_rr = drive_down / or_height if or_height > 0 else 0.0

    reason = []
    if or_height <= 0:
        reason.append("zero_or_height")
    if up_rr < cfg.min_drive_rr and down_rr < cfg.min_drive_rr:
        reason.append("weak_drive_both_directions")
    if up_rr >= cfg.min_drive_rr and down_rr > cfg.max_counter_rr:
        reason.append("long_drive_but_too_much_counter")
    if down_rr >= cfg.min_drive_rr and up_rr > cfg.max_counter_rr:
        reason.append("short_drive_but_too_much_counter")

    if not reason:
        reason.append("no_breakout_bar_found")

    return ORBMissInfo(
        has_or_window=True,
        or_height=or_height,
        drive_up=drive_up,
        drive_down=drive_down,
        up_rr=up_rr,
        down_rr=down_rr,
        reason=",".join(reason),
    )


def _find_session_or(df_5m: pd.DataFrame, cfg: ORBConfig) -> Optional[pd.DataFrame]:
    """Slice the DataFrame to the opening range window for the first session found.

    Currently we just take the first calendar day present and define the OR
    window as [session_start, session_start + or_minutes).
    """
    if df_5m.empty:
        return None

    # Use the date of the first bar as the session day
    first_ts = df_5m.index[0]
    session_start = first_ts.replace(
        hour=cfg.session_start_hour,
        minute=cfg.session_start_minute,
        second=0,
        microsecond=0,
    )
    or_end = session_start + pd.Timedelta(minutes=cfg.or_minutes)

    or_slice = df_5m[(df_5m.index >= session_start) & (df_5m.index < or_end)]
    if or_slice.empty:
        return None
    return or_slice


def find_opening_orb_continuations(
    df_5m: pd.DataFrame,
    cfg: Optional[ORBConfig] = None,
) -> List[ORBEntry]:
    """Find Opening Range Breakout continuation setups on 5m data.

    Strategy (long side, short is symmetric):
    1. Define an opening range (OR) from session_start for `or_minutes`.
    2. Compute OR high/low and height.
    3. After OR ends, watch for a close above OR_high + buffer to define a
       breakout direction and candidate drive.
    4. Ensure that during the OR the net movement in that direction is at
       least `min_drive_rr` * OR_height and max counter move is within
       `max_counter_rr` * OR_height.
    5. Entry is at the first bar close beyond the buffer.
    6. Stop is either beyond the OR extreme or the entry bar extreme.
    7. Target is `target_rr` multiples of risk.
    """
    if cfg is None:
        cfg = ORBConfig()

    df_5m = df_5m.sort_index().copy()

    or_slice = _find_session_or(df_5m, cfg)
    if or_slice is None:
        return []

    or_high = float(or_slice["high"].max())
    or_low = float(or_slice["low"].min())
    or_start = or_slice.index[0]
    or_end = or_slice.index[-1]
    or_height = or_high - or_low
    if or_height <= 0:
        return []

    buffer = cfg.buffer_ticks * cfg.tick_size

    # Price action during OR: measure drive vs counter in both directions
    or_closes = or_slice["close"]
    drive_up = float(or_closes.max() - or_closes.iloc[0])
    drive_down = float(or_closes.iloc[0] - or_closes.min())

    # Determine candidate direction based on OR drive
    up_rr = drive_up / or_height
    down_rr = drive_down / or_height

    direction: Optional[Direction] = None
    if up_rr >= cfg.min_drive_rr and down_rr <= cfg.max_counter_rr:
        direction = "long"
    elif down_rr >= cfg.min_drive_rr and up_rr <= cfg.max_counter_rr:
        direction = "short"

    if direction is None:
        return []

    # Look for first close beyond OR with buffer after OR window
    post_or = df_5m[df_5m.index > or_slice.index[-1]]
    if post_or.empty:
        return []

    entries: List[ORBEntry] = []

    if direction == "long":
        trigger = post_or[post_or["close"] > or_high + buffer]
    else:
        trigger = post_or[post_or["close"] < or_low - buffer]

    if trigger.empty:
        return []

    # For now we only take the first breakout; later we can extend to multiple
    first_bar = trigger.iloc[0]
    entry_time = trigger.index[0]
    
    # Restrict entry to within 60 minutes of OR end
    if entry_time > or_end + pd.Timedelta(minutes=60):
        return []
        
    entry_price = float(first_bar["close"])

    if direction == "long":
        if cfg.stop_type == "or":
            stop_price = or_low - buffer
        else:
            stop_price = float(min(first_bar["low"], or_low)) - buffer
    else:
        if cfg.stop_type == "or":
            stop_price = or_high + buffer
        else:
            stop_price = float(max(first_bar["high"], or_high)) + buffer

    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return []

    if direction == "long":
        target_price = entry_price + cfg.target_rr * risk
    else:
        target_price = entry_price - cfg.target_rr * risk

    entries.append(
        ORBEntry(
            time=entry_time,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            or_high=or_high,
            or_low=or_low,
            or_start=or_start,
            or_end=or_end,
        )
    )

    return entries


def evaluate_orb_entry(df_5m: pd.DataFrame, entry: ORBEntry, cfg: Optional[ORBConfig] = None) -> Optional[ORBOutcome]:
    """Evaluate a single ORB entry on future 5m bars.

    We step forward from the entry bar and see which is hit first: target or
    stop, with a cap of `max_hold_bars`.
    """
    if cfg is None:
        cfg = ORBConfig()

    df_5m = df_5m.sort_index()
    future = df_5m[df_5m.index > entry.time]
    if future.empty:
        return None

    future = future.iloc[: cfg.max_hold_bars]

    hit_target = False
    hit_stop = False
    exit_time = future.index[-1]

    # Track MFE/MAE in price space relative to entry
    mfe = 0.0
    mae = 0.0

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if entry.direction == "long":
            # Update excursions
            mfe = max(mfe, high - entry.entry_price)
            mae = min(mae, low - entry.entry_price)

            if low <= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if high >= entry.target_price:
                hit_target = True
                exit_time = ts
                break
        else:
            mfe = max(mfe, entry.entry_price - low)
            mae = min(mae, entry.entry_price - high)

            if high >= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if low <= entry.target_price:
                hit_target = True
                exit_time = ts
                break

    risk = abs(entry.entry_price - entry.stop_price)
    if risk <= 0:
        r_mult = 0.0
    else:
        if hit_target:
            r_mult = cfg.target_rr
        elif hit_stop:
            r_mult = -1.0
        else:
            # Neither hit; mark to market at last close
            last_close = float(future.iloc[-1]["close"])
            if entry.direction == "long":
                r_mult = (last_close - entry.entry_price) / risk
            else:
                r_mult = (entry.entry_price - last_close) / risk

    return ORBOutcome(
        entry=entry,
        hit_target=hit_target,
        hit_stop=hit_stop,
        exit_time=exit_time,
        r_multiple=r_mult,
        mfe=mfe,
        mae=mae,
    )


def summarize_orb_day(
    df_5m: pd.DataFrame,
    cfg: Optional[ORBConfig] = None,
) -> Tuple[List[ORBEntry], List[ORBOutcome], ORBMissInfo]:
    """Run ORB detection + evaluation and return entries, outcomes, and miss info.

    - entries/outcomes will be empty if no qualifying setups.
    - miss will always be populated to explain the no-trade or pre-trade context.
    """

    from .library import find_opening_orb_continuations  # type: ignore  # local import to avoid circular

    if cfg is None:
        cfg = ORBConfig()

    entries = find_opening_orb_continuations(df_5m, cfg)
    miss = explain_orb_missing(df_5m, cfg)

    outcomes: List[ORBOutcome] = []
    for e in entries:
        outcome = evaluate_orb_entry(df_5m, e, cfg)
        if outcome is not None:
            outcomes.append(outcome)

    return entries, outcomes, miss


# 1m-based detection with 5-min decision anchors
def find_opening_orb_continuations_1m(
    df_1m: pd.DataFrame,
    cfg: Optional[ORBConfig] = None,
    anchor: Optional[DecisionAnchorConfig] = None,
) -> List[ORBEntry]:
    if cfg is None:
        cfg = ORBConfig()
    if anchor is None:
        anchor = DecisionAnchorConfig()

    df_1m = df_1m.sort_index().copy()

    # OR window on 1m
    first_ts = df_1m.index[0]
    session_start = first_ts.replace(
        hour=cfg.session_start_hour,
        minute=cfg.session_start_minute,
        second=0,
        microsecond=0,
    )
    or_end = session_start + pd.Timedelta(minutes=cfg.or_minutes)
    or_slice = df_1m[(df_1m.index >= session_start) & (df_1m.index < or_end)]
    if or_slice.empty:
        return []

    or_high = float(or_slice["high"].max())
    or_low = float(or_slice["low"].min())
    or_start = or_slice.index[0]
    or_end_ts = or_slice.index[-1]
    or_height = or_high - or_low
    if or_height <= 0:
        return []

    buffer = cfg.buffer_ticks * cfg.tick_size
    closes = or_slice["close"]
    drive_up = float(closes.max() - closes.iloc[0])
    drive_down = float(closes.iloc[0] - closes.min())
    up_rr = drive_up / or_height
    down_rr = drive_down / or_height

    direction: Optional[Direction] = None
    if up_rr >= cfg.min_drive_rr and down_rr <= cfg.max_counter_rr:
        direction = "long"
    elif down_rr >= cfg.min_drive_rr and up_rr <= cfg.max_counter_rr:
        direction = "short"
    if direction is None:
        return []

    post_or = df_1m[df_1m.index > or_end_ts]
    if post_or.empty:
        return []

    decision_marks = mark_decision_points_1m(post_or, anchor)
    candidates = post_or[decision_marks]
    if candidates.empty:
        return []

    if direction == "long":
        trigger = candidates[candidates["close"] > or_high + buffer]
    else:
        trigger = candidates[candidates["close"] < or_low - buffer]
    if trigger.empty:
        return []

    first_bar = trigger.iloc[0]
    entry_time = trigger.index[0]
    entry_price = float(first_bar["close"])

    if direction == "long":
        stop_price = or_low - buffer if cfg.stop_type == "or" else float(min(first_bar["low"], or_low)) - buffer
        target_price = entry_price + cfg.target_rr * abs(entry_price - stop_price)
    else:
        stop_price = or_high + buffer if cfg.stop_type == "or" else float(max(first_bar["high"], or_high)) + buffer
        target_price = entry_price - cfg.target_rr * abs(entry_price - stop_price)

    return [
        ORBEntry(
            time=entry_time,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            or_high=or_high,
            or_low=or_low,
            or_start=or_start,
            or_end=or_end_ts,
        )
    ]


def evaluate_orb_entry_1m(
    df_1m: pd.DataFrame,
    entry: ORBEntry,
    cfg: Optional[ORBConfig] = None,
) -> Optional[ORBOutcome]:
    if cfg is None:
        cfg = ORBConfig()
    df_1m = df_1m.sort_index()
    future = df_1m[df_1m.index > entry.time]
    if future.empty:
        return None

    # Max hold in minutes approximated by 5m bars * 5
    max_minutes = cfg.max_hold_bars * 5
    future = future.iloc[: max_minutes]

    hit_target = False
    hit_stop = False
    exit_time = future.index[-1]
    mfe = 0.0
    mae = 0.0

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if entry.direction == "long":
            mfe = max(mfe, high - entry.entry_price)
            mae = min(mae, low - entry.entry_price)
            if low <= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if high >= entry.target_price:
                hit_target = True
                exit_time = ts
                break
        else:
            mfe = max(mfe, entry.entry_price - low)
            mae = min(mae, entry.entry_price - high)
            if high >= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if low <= entry.target_price:
                hit_target = True
                exit_time = ts
                break

    risk = abs(entry.entry_price - entry.stop_price)
    if risk <= 0:
        r_mult = 0.0
    else:
        if hit_target:
            r_mult = cfg.target_rr
        elif hit_stop:
            r_mult = -1.0
        else:
            last_close = float(future.iloc[-1]["close"])
            r_mult = ((last_close - entry.entry_price) / risk) if entry.direction == "long" else ((entry.entry_price - last_close) / risk)

    return ORBOutcome(
        entry=entry,
        hit_target=hit_target,
        hit_stop=hit_stop,
        exit_time=exit_time,
        r_multiple=r_mult,
        mfe=mfe,
        mae=mae,
    )


def summarize_orb_day_1m(
    df_1m: pd.DataFrame,
    cfg: Optional[ORBConfig] = None,
    anchor: Optional[DecisionAnchorConfig] = None,
) -> Tuple[List[ORBEntry], List[ORBOutcome], ORBMissInfo]:
    if cfg is None:
        cfg = ORBConfig()
    if anchor is None:
        anchor = DecisionAnchorConfig()

    entries = find_opening_orb_continuations_1m(df_1m, cfg, anchor)
    # For miss info, reuse 5m method logic by aggregating quickly if needed, or compute directly from 1m OR slice
    miss = explain_orb_missing(df_1m.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(), cfg)

    outcomes: List[ORBOutcome] = []
    for e in entries:
        out = evaluate_orb_entry_1m(df_1m, e, cfg)
        if out is not None:
            outcomes.append(out)

    return entries, outcomes, miss


def evaluate_generic_entry_1m(
    df_1m: pd.DataFrame,
    entry: SetupEntry,
    max_minutes: int = 240,
) -> SetupOutcome:
    """Evaluate a generic SetupEntry on 1m data.

    Walk forward minute-by-minute up to max_minutes and determine
    whether stop or target hits first; compute R, MFE/MAE.
    """
    df_1m = df_1m.sort_index()
    future = df_1m[df_1m.index > entry.time]
    if future.empty:
        return SetupOutcome(
            entry=entry,
            hit_target=False,
            hit_stop=False,
            exit_time=entry.time,
            r_multiple=0.0,
            mfe=0.0,
            mae=0.0,
        )

    future = future.iloc[: max_minutes]

    hit_target = False
    hit_stop = False
    exit_time = future.index[-1]
    mfe = 0.0
    mae = 0.0

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if entry.direction == "long":
            mfe = max(mfe, high - entry.entry_price)
            mae = min(mae, low - entry.entry_price)
            if low <= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if high >= entry.target_price:
                hit_target = True
                exit_time = ts
                break
        else:
            mfe = max(mfe, entry.entry_price - low)
            mae = min(mae, entry.entry_price - high)
            if high >= entry.stop_price:
                hit_stop = True
                exit_time = ts
                break
            if low <= entry.target_price:
                hit_target = True
                exit_time = ts
                break

    risk = abs(entry.entry_price - entry.stop_price)
    if risk <= 0:
        r_mult = 0.0
    else:
        if hit_target:
            r_mult = (entry.target_price - entry.entry_price) / risk if entry.direction == "long" else (entry.entry_price - entry.target_price) / risk
        elif hit_stop:
            r_mult = -1.0
        else:
            last_close = float(future.iloc[-1]["close"])
            r_mult = ((last_close - entry.entry_price) / risk) if entry.direction == "long" else ((entry.entry_price - last_close) / risk)

    return SetupOutcome(
        entry=entry,
        hit_target=hit_target,
        hit_stop=hit_stop,
        exit_time=exit_time,
        r_multiple=r_mult,
        mfe=mfe,
        mae=mae,
    )


def run_orb_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """
    Wrapper to run ORB setup detection and return a list of outcomes.
    Compatible with the engine's expectation.
    """
    cfg = ORBConfig(**kwargs)
    _, outcomes, _ = summarize_orb_day(df_5m, cfg)
    return outcomes  # type: ignore


# ============================================================================
# EMA200 CONTINUATION SETUP
# ============================================================================

@dataclass
class EMA200ContinuationConfig:
    """Config for EMA200 trend continuation setup on 5m bars.
    
    Looks for price pullback to EMA200 in a trending market with
    RSI confirmation and volume support.
    """
    
    tick_size: float = 0.25
    ema_proximity_ticks: float = 4.0  # how close to EMA200 counts as pullback
    rsi_oversold: float = 35.0        # RSI level for long setups
    rsi_overbought: float = 65.0      # RSI level for short setups
    min_volume_ratio: float = 0.8     # volume vs SMA threshold
    stop_atr_mult: float = 1.5        # stop distance in ATR
    target_rr: float = 2.0
    max_hold_bars: int = 48


def find_ema200_continuation(
    df_5m: pd.DataFrame,
    cfg: Optional[EMA200ContinuationConfig] = None,
) -> List[SetupEntry]:
    """Find EMA200 continuation setups on 5m data with indicators."""
    if cfg is None:
        cfg = EMA200ContinuationConfig()
    
    df = df_5m.sort_index().copy()
    required_cols = ["close", "ema_slow", "rsi", "atr", "volume_ratio"]
    if not all(col in df.columns for col in required_cols):
        return []
    
    entries: List[SetupEntry] = []
    proximity = cfg.ema_proximity_ticks * cfg.tick_size
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        close = float(row["close"])
        ema200 = float(row["ema_slow"])
        rsi = float(row["rsi"])
        atr = float(row["atr"])
        vol_ratio = float(row["volume_ratio"])
        
        if pd.isna(ema200) or pd.isna(rsi) or pd.isna(atr) or atr <= 0:
            continue
        
        # Long setup: price near EMA200 from above, RSI oversold, volume support
        if (close > ema200 and abs(close - ema200) <= proximity and
            rsi < cfg.rsi_oversold and vol_ratio >= cfg.min_volume_ratio):
            
            entry_price = close
            stop_price = close - (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            target_price = entry_price + (cfg.target_rr * risk)
            
            entries.append(SetupEntry(
                time=df.index[i],
                direction="long",
                kind="ema200_continuation",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                context={
                    "ema200": ema200,
                    "rsi": rsi,
                    "atr": atr,
                    "vol_ratio": vol_ratio,
                }
            ))
        
        # Short setup: price near EMA200 from below, RSI overbought, volume support
        elif (close < ema200 and abs(close - ema200) <= proximity and
              rsi > cfg.rsi_overbought and vol_ratio >= cfg.min_volume_ratio):
            
            entry_price = close
            stop_price = close + (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            target_price = entry_price - (cfg.target_rr * risk)
            
            entries.append(SetupEntry(
                time=df.index[i],
                direction="short",
                kind="ema200_continuation",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                context={
                    "ema200": ema200,
                    "rsi": rsi,
                    "atr": atr,
                    "vol_ratio": vol_ratio,
                }
            ))
    
    return entries


def run_ema200_continuation_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """Wrapper to run EMA200 continuation setup detection."""
    cfg = EMA200ContinuationConfig(**kwargs)
    entries = find_ema200_continuation(df_5m, cfg)
    outcomes: List[SetupOutcome] = []
    for entry in entries:
        outcome = evaluate_generic_entry_1m(
            df_5m.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
            entry,
            max_minutes=cfg.max_hold_bars * 5
        )
        outcomes.append(outcome)
    return outcomes


# ============================================================================
# BREAKOUT SETUP
# ============================================================================

@dataclass
class BreakoutConfig:
    """Config for breakout setup on 5m bars.
    
    Looks for price breaking above recent highs or below recent lows
    with volume confirmation and momentum (MACD).
    """
    
    tick_size: float = 0.25
    lookback_bars: int = 20           # bars to find high/low
    buffer_ticks: float = 2.0         # extra ticks for valid breakout
    min_volume_ratio: float = 1.2     # volume vs SMA for confirmation
    macd_threshold: float = 0.0       # MACD histogram threshold
    stop_atr_mult: float = 1.5
    target_rr: float = 2.5
    max_hold_bars: int = 48


def find_breakout(
    df_5m: pd.DataFrame,
    cfg: Optional[BreakoutConfig] = None,
) -> List[SetupEntry]:
    """Find breakout setups on 5m data with volume and MACD confirmation."""
    if cfg is None:
        cfg = BreakoutConfig()
    
    df = df_5m.sort_index().copy()
    required_cols = ["close", "high", "low", "atr", "volume_ratio", "macd_histogram"]
    if not all(col in df.columns for col in required_cols):
        return []
    
    entries: List[SetupEntry] = []
    buffer = cfg.buffer_ticks * cfg.tick_size
    
    for i in range(cfg.lookback_bars, len(df)):
        row = df.iloc[i]
        lookback_slice = df.iloc[i-cfg.lookback_bars:i]
        
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])
        vol_ratio = float(row["volume_ratio"])
        macd_hist = float(row["macd_histogram"])
        
        if pd.isna(atr) or atr <= 0 or pd.isna(vol_ratio) or pd.isna(macd_hist):
            continue
        
        recent_high = float(lookback_slice["high"].max())
        recent_low = float(lookback_slice["low"].min())
        
        # Long breakout: close above recent high with volume and MACD
        if (close > recent_high + buffer and 
            vol_ratio >= cfg.min_volume_ratio and
            macd_hist > cfg.macd_threshold):
            
            entry_price = close
            stop_price = recent_high - buffer
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price + (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=df.index[i],
                    direction="long",
                    kind="breakout",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "breakout_level": recent_high,
                        "vol_ratio": vol_ratio,
                        "macd_histogram": macd_hist,
                    }
                ))
        
        # Short breakout: close below recent low with volume and MACD
        elif (close < recent_low - buffer and
              vol_ratio >= cfg.min_volume_ratio and
              macd_hist < -cfg.macd_threshold):
            
            entry_price = close
            stop_price = recent_low + buffer
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price - (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=df.index[i],
                    direction="short",
                    kind="breakout",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "breakout_level": recent_low,
                        "vol_ratio": vol_ratio,
                        "macd_histogram": macd_hist,
                    }
                ))
    
    return entries


def run_breakout_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """Wrapper to run breakout setup detection."""
    cfg = BreakoutConfig(**kwargs)
    entries = find_breakout(df_5m, cfg)
    outcomes: List[SetupOutcome] = []
    for entry in entries:
        outcome = evaluate_generic_entry_1m(
            df_5m.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
            entry,
            max_minutes=cfg.max_hold_bars * 5
        )
        outcomes.append(outcome)
    return outcomes


# ============================================================================
# REVERSAL SETUP
# ============================================================================

@dataclass
class ReversalConfig:
    """Config for reversal setup on 5m bars.
    
    Looks for price rejecting Bollinger Band extremes with RSI divergence
    and candlestick patterns (wicks).
    """
    
    tick_size: float = 0.25
    bb_touch_ticks: float = 2.0       # proximity to BB for touch
    rsi_extreme_long: float = 30.0    # RSI oversold for long
    rsi_extreme_short: float = 70.0   # RSI overbought for short
    min_wick_ratio: float = 0.4       # wick size vs range for rejection
    stop_atr_mult: float = 1.5
    target_rr: float = 2.0
    max_hold_bars: int = 36


def find_reversal(
    df_5m: pd.DataFrame,
    cfg: Optional[ReversalConfig] = None,
) -> List[SetupEntry]:
    """Find reversal setups at BB extremes with RSI and wick confirmation."""
    if cfg is None:
        cfg = ReversalConfig()
    
    df = df_5m.sort_index().copy()
    required_cols = ["open", "close", "high", "low", "bb_upper", "bb_lower", "rsi", "atr"]
    if not all(col in df.columns for col in required_cols):
        return []
    
    entries: List[SetupEntry] = []
    proximity = cfg.bb_touch_ticks * cfg.tick_size
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        open_price = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        bb_upper = float(row["bb_upper"])
        bb_lower = float(row["bb_lower"])
        rsi = float(row["rsi"])
        atr = float(row["atr"])
        
        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(rsi) or pd.isna(atr) or atr <= 0:
            continue
        
        bar_range = high - low
        if bar_range <= 0:
            continue
        
        # Long reversal: touch lower BB, RSI oversold, rejection wick
        lower_wick = min(open_price, close) - low
        wick_ratio = lower_wick / bar_range
        
        if (low <= bb_lower + proximity and 
            rsi < cfg.rsi_extreme_long and
            wick_ratio >= cfg.min_wick_ratio):
            
            entry_price = close
            stop_price = low - (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price + (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=df.index[i],
                    direction="long",
                    kind="reversal",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "bb_level": bb_lower,
                        "rsi": rsi,
                        "wick_ratio": wick_ratio,
                    }
                ))
        
        # Short reversal: touch upper BB, RSI overbought, rejection wick
        upper_wick = high - max(open_price, close)
        wick_ratio = upper_wick / bar_range
        
        if (high >= bb_upper - proximity and
            rsi > cfg.rsi_extreme_short and
            wick_ratio >= cfg.min_wick_ratio):
            
            entry_price = close
            stop_price = high + (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price - (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=df.index[i],
                    direction="short",
                    kind="reversal",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "bb_level": bb_upper,
                        "rsi": rsi,
                        "wick_ratio": wick_ratio,
                    }
                ))
    
    return entries


def run_reversal_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """Wrapper to run reversal setup detection."""
    cfg = ReversalConfig(**kwargs)
    entries = find_reversal(df_5m, cfg)
    outcomes: List[SetupOutcome] = []
    for entry in entries:
        outcome = evaluate_generic_entry_1m(
            df_5m.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
            entry,
            max_minutes=cfg.max_hold_bars * 5
        )
        outcomes.append(outcome)
    return outcomes


# ============================================================================
# OPENING PUSH SETUP
# ============================================================================

@dataclass
class OpeningPushConfig:
    """Config for opening push setup on 5m bars.
    
    Captures momentum in the first 15-30 minutes of the session with
    volume and directional indicators.
    """
    
    session_start_hour: int = 14      # 09:30 ET in UTC
    session_start_minute: int = 30
    push_window_minutes: int = 30     # duration to look for push
    min_move_ticks: float = 8.0       # minimum move from open
    min_volume_ratio: float = 1.5     # volume vs average
    tick_size: float = 0.25
    stop_atr_mult: float = 2.0
    target_rr: float = 2.0
    max_hold_bars: int = 48


def find_opening_push(
    df_5m: pd.DataFrame,
    cfg: Optional[OpeningPushConfig] = None,
) -> List[SetupEntry]:
    """Find opening push setups in the first 30 minutes of session."""
    if cfg is None:
        cfg = OpeningPushConfig()
    
    df = df_5m.sort_index().copy()
    required_cols = ["open", "close", "high", "low", "atr", "volume_ratio"]
    if not all(col in df.columns for col in required_cols):
        return []
    
    entries: List[SetupEntry] = []
    
    # Find session start
    if df.empty:
        return []
    
    first_ts = df.index[0]
    session_start = first_ts.replace(
        hour=cfg.session_start_hour,
        minute=cfg.session_start_minute,
        second=0,
        microsecond=0,
    )
    push_end = session_start + pd.Timedelta(minutes=cfg.push_window_minutes)
    
    # Get opening range
    open_slice = df[(df.index >= session_start) & (df.index < push_end)]
    if open_slice.empty or len(open_slice) < 3:
        return []
    
    open_price = float(open_slice.iloc[0]["open"])
    min_move = cfg.min_move_ticks * cfg.tick_size
    
    # Look for strong directional move with volume
    for i in range(2, len(open_slice)):
        row = open_slice.iloc[i]
        close = float(row["close"])
        atr = float(row["atr"])
        vol_ratio = float(row["volume_ratio"])
        
        if pd.isna(atr) or atr <= 0 or pd.isna(vol_ratio):
            continue
        
        move_from_open = close - open_price
        
        # Long push: strong upward move with volume
        if (move_from_open >= min_move and vol_ratio >= cfg.min_volume_ratio):
            entry_price = close
            stop_price = entry_price - (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price + (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=open_slice.index[i],
                    direction="long",
                    kind="opening_push",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "session_open": open_price,
                        "move_ticks": move_from_open / cfg.tick_size,
                        "vol_ratio": vol_ratio,
                    }
                ))
            # Take first valid signal
            break
        
        # Short push: strong downward move with volume
        elif (move_from_open <= -min_move and vol_ratio >= cfg.min_volume_ratio):
            entry_price = close
            stop_price = entry_price + (cfg.stop_atr_mult * atr)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price - (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=open_slice.index[i],
                    direction="short",
                    kind="opening_push",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "session_open": open_price,
                        "move_ticks": move_from_open / cfg.tick_size,
                        "vol_ratio": vol_ratio,
                    }
                ))
            # Take first valid signal
            break
    
    return entries


def run_opening_push_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """Wrapper to run opening push setup detection."""
    cfg = OpeningPushConfig(**kwargs)
    entries = find_opening_push(df_5m, cfg)
    outcomes: List[SetupOutcome] = []
    for entry in entries:
        outcome = evaluate_generic_entry_1m(
            df_5m.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
            entry,
            max_minutes=cfg.max_hold_bars * 5
        )
        outcomes.append(outcome)
    return outcomes


# ============================================================================
# MOC (MARKET ON CLOSE) SETUP
# ============================================================================

@dataclass
class MOCConfig:
    """Config for Market on Close setup on 5m bars.
    
    Captures trend continuation or reversal behavior in the last hour
    of the session, often driven by institutional flows.
    """
    
    session_end_hour: int = 20        # 16:00 ET in UTC (approx)
    session_end_minute: int = 0
    moc_window_minutes: int = 60      # last hour before close
    min_directional_bars: int = 3     # consecutive bars in same direction
    min_volume_ratio: float = 1.2     # volume confirmation
    tick_size: float = 0.25
    stop_atr_mult: float = 1.5
    target_rr: float = 1.5            # shorter target for EOD
    max_hold_bars: int = 24


def find_moc(
    df_5m: pd.DataFrame,
    cfg: Optional[MOCConfig] = None,
) -> List[SetupEntry]:
    """Find Market on Close setups in the last hour of session."""
    if cfg is None:
        cfg = MOCConfig()
    
    df = df_5m.sort_index().copy()
    required_cols = ["open", "close", "high", "low", "atr", "volume_ratio"]
    if not all(col in df.columns for col in required_cols):
        return []
    
    entries: List[SetupEntry] = []
    
    if df.empty:
        return []
    
    # Find session end
    last_ts = df.index[-1]
    session_end = last_ts.replace(
        hour=cfg.session_end_hour,
        minute=cfg.session_end_minute,
        second=0,
        microsecond=0,
    )
    moc_start = session_end - pd.Timedelta(minutes=cfg.moc_window_minutes)
    
    # Get MOC window
    moc_slice = df[(df.index >= moc_start) & (df.index < session_end)]
    if moc_slice.empty or len(moc_slice) < cfg.min_directional_bars + 1:
        return []
    
    # Look for directional run with volume
    for i in range(cfg.min_directional_bars, len(moc_slice)):
        lookback = moc_slice.iloc[i-cfg.min_directional_bars:i]
        row = moc_slice.iloc[i]
        
        closes = lookback["close"].values
        atr = float(row["atr"])
        vol_ratio = float(row["volume_ratio"])
        
        if pd.isna(atr) or atr <= 0 or pd.isna(vol_ratio):
            continue
        
        # Check for directional move
        up_bars = sum(1 for j in range(len(closes)-1) if closes[j+1] > closes[j])
        down_bars = sum(1 for j in range(len(closes)-1) if closes[j+1] < closes[j])
        
        close = float(row["close"])
        
        # Long setup: consistent upward momentum
        if (up_bars >= cfg.min_directional_bars - 1 and 
            vol_ratio >= cfg.min_volume_ratio):
            
            entry_price = close
            stop_price = float(lookback["low"].min()) - (cfg.stop_atr_mult * atr * 0.5)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price + (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=moc_slice.index[i],
                    direction="long",
                    kind="moc",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "up_bars": up_bars,
                        "vol_ratio": vol_ratio,
                        "time_to_close": (session_end - moc_slice.index[i]).total_seconds() / 60,
                    }
                ))
            # Take first valid signal
            break
        
        # Short setup: consistent downward momentum
        elif (down_bars >= cfg.min_directional_bars - 1 and
              vol_ratio >= cfg.min_volume_ratio):
            
            entry_price = close
            stop_price = float(lookback["high"].max()) + (cfg.stop_atr_mult * atr * 0.5)
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = entry_price - (cfg.target_rr * risk)
                
                entries.append(SetupEntry(
                    time=moc_slice.index[i],
                    direction="short",
                    kind="moc",
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    context={
                        "down_bars": down_bars,
                        "vol_ratio": vol_ratio,
                        "time_to_close": (session_end - moc_slice.index[i]).total_seconds() / 60,
                    }
                ))
            # Take first valid signal
            break
    
    return entries


def run_moc_family(df_5m: pd.DataFrame, **kwargs) -> List[SetupOutcome]:
    """Wrapper to run MOC setup detection."""
    cfg = MOCConfig(**kwargs)
    entries = find_moc(df_5m, cfg)
    outcomes: List[SetupOutcome] = []
    for entry in entries:
        outcome = evaluate_generic_entry_1m(
            df_5m.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
            entry,
            max_minutes=cfg.max_hold_bars * 5
        )
        outcomes.append(outcome)
    return outcomes

```

---

## File: src/core/detector/features.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from .models import SetupOutcome
from .models import SetupEntry
from .library import evaluate_generic_entry_1m


@dataclass
class FeatureConfig:
    tick_size: float = 0.25


@dataclass
class BadTradeConfig:
    """Knobs to synthesize labeled 'bad trades' for ML practice."""

    enable_hesitation: bool = False
    hesitation_minutes: int = 5
    enable_chase: bool = False
    chase_window_minutes: int = 15
    max_variants_per_trade: int = 2


def nearest_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    if ts in df.index:
        return df.loc[ts]
    # fallback: nearest by time
    idx = df.index.get_indexer([ts], method="nearest")[0]
    return df.iloc[idx]


def compute_trade_features(
    outcome: SetupOutcome,
    df_ind: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> Dict[str, Any]:
    """Compute a compact set of trade features suitable for ML.

    Inputs:
      - outcome: SetupOutcome (contains entry, exit, R, mfe/mae)
      - df_5m_ind: 5m DataFrame with indicators (ema_fast, ema_slow, vwap)
    Returns:
      - dict of numeric/categorical features describing context and execution
    """
    if cfg is None:
        cfg = FeatureConfig()

    e = outcome.entry
    entry_ts = e.time
    exit_ts = outcome.exit_time

    entry_row = nearest_row(df_ind, entry_ts)
    exit_row = nearest_row(df_ind, exit_ts)

    # Price distances
    dist_ema_fast = float(entry_row["close"] - entry_row.get("ema_fast", entry_row["close"]))
    dist_ema_slow = float(entry_row["close"] - entry_row.get("ema_slow", entry_row["close"]))
    dist_vwap = float(entry_row["close"] - entry_row.get("vwap", entry_row["close"]))

    # Candle geometry at entry
    rng = float(entry_row["high"] - entry_row["low"]) if "high" in entry_row else 0.0
    body = abs(float(entry_row["close"] - entry_row["open"])) if "open" in entry_row else 0.0
    wick_up = float(entry_row["high"] - max(entry_row["open"], entry_row["close"])) if "open" in entry_row else 0.0
    wick_dn = float(min(entry_row["open"], entry_row["close"]) - entry_row["low"]) if "open" in entry_row else 0.0
    body_frac = body / rng if rng > 0 else 0.0
    wick_up_frac = wick_up / rng if rng > 0 else 0.0
    wick_dn_frac = wick_dn / rng if rng > 0 else 0.0

    # Timing features
    bars_to_exit = int(df_ind.index.get_indexer([exit_ts], method="nearest")[0]) - int(
        df_ind.index.get_indexer([entry_ts], method="nearest")[0]
    )
    hour = int(entry_ts.hour)
    minute = int(entry_ts.minute)

    # Risk & outcome
    risk = abs(e.entry_price - e.stop_price)
    mfe = float(outcome.mfe)
    mae = float(outcome.mae)
    mfe_mae_ratio = (mfe / max(1e-9, abs(mae))) if mae != 0 else 0.0

    # Simple flags for behavior labeling (placeholders for ML):
    low_rr_flag = 1 if outcome.r_multiple < 0.5 else 0
    chase_flag = 1 if (abs(dist_vwap) < 0.5 and body_frac > 0.7) else 0
    hesitation_flag = 1 if (wick_dn_frac > 0.6 and e.direction == "long") or (wick_up_frac > 0.6 and e.direction == "short") else 0

    features: Dict[str, Any] = {
        "kind": e.kind,
        "direction": e.direction,
        "entry_time": str(entry_ts),
        "exit_time": str(exit_ts),
        "hour": hour,
        "minute": minute,
        "entry_price": e.entry_price,
        "stop_price": e.stop_price,
        "target_price": e.target_price,
        "risk": risk,
        "r_multiple": outcome.r_multiple,
        "mfe": mfe,
        "mae": mae,
        "mfe_mae_ratio": mfe_mae_ratio,
        "dist_ema_fast": dist_ema_fast,
        "dist_ema_slow": dist_ema_slow,
        "dist_vwap": dist_vwap,
        "body_frac": body_frac,
        "wick_up_frac": wick_up_frac,
        "wick_dn_frac": wick_dn_frac,
        "bars_to_exit": bars_to_exit,
        "low_rr_flag": low_rr_flag,
        "chase_flag": chase_flag,
        "hesitation_flag": hesitation_flag,
    }

    # Include context if present (e.g., OR bounds, level names)
    for k, v in (e.context or {}).items():
        features[f"ctx_{k}"] = v

    return features


def inject_bad_trade_variants(
    df_1m_ind: pd.DataFrame,
    base_outcomes: list[SetupOutcome],
    cfg: BadTradeConfig,
) -> list[SetupOutcome]:
    """Create synthetic 'bad trade' variants (hesitation/chase) from outcomes.

    Returns a list of additional SetupOutcome objects labeled via entry.context["variant"].
    """
    variants: list[SetupOutcome] = []
    for o in base_outcomes:
        e = o.entry
        produced = 0

        # Hesitation: enter later at a decision point (or next minute), keep stop, recompute target by R
        if cfg.enable_hesitation and produced < cfg.max_variants_per_trade:
            later_idx = df_1m_ind.index.get_indexer([e.time + pd.Timedelta(minutes=cfg.hesitation_minutes)], method="nearest")[0]
            if later_idx < len(df_1m_ind):
                later_ts = df_1m_ind.index[later_idx]
                later_close = float(df_1m_ind.iloc[later_idx]["close"])
                new_risk = abs(later_close - e.stop_price)
                if new_risk > 0:
                    target_price = later_close + (o.r_multiple if e.direction == "long" else -o.r_multiple) * new_risk
                    entry2 = SetupEntry(
                        time=later_ts,
                        direction=e.direction,
                        kind=e.kind,
                        entry_price=later_close,
                        stop_price=e.stop_price,
                        target_price=target_price,
                        context={**(e.context or {}), "variant": "hesitation"},
                    )
                    out2 = evaluate_generic_entry_1m(df_1m_ind, entry2, max_minutes=240)
                    variants.append(out2)
                    produced += 1

        # Chase: pick a later bar within window with max |close - vwap|
        if cfg.enable_chase and produced < cfg.max_variants_per_trade:
            window_end = e.time + pd.Timedelta(minutes=cfg.chase_window_minutes)
            window = df_1m_ind[(df_1m_ind.index > e.time) & (df_1m_ind.index <= window_end)]
            if not window.empty and "vwap" in window.columns:
                diffs = (window["close"] - window["vwap"]).abs()
                idxmax = int(diffs.idxmax() == diffs.index).bit_length()  # safe guard but not used
                ts_max = diffs.idxmax()
                close_max = float(window.loc[ts_max, "close"])
                new_risk = abs(close_max - e.stop_price)
                if new_risk > 0:
                    target_price = close_max + (o.r_multiple if e.direction == "long" else -o.r_multiple) * new_risk
                    entry2 = SetupEntry(
                        time=ts_max,
                        direction=e.direction,
                        kind=e.kind,
                        entry_price=close_max,
                        stop_price=e.stop_price,
                        target_price=target_price,
                        context={**(e.context or {}), "variant": "chase"},
                    )
                    out2 = evaluate_generic_entry_1m(df_1m_ind, entry2, max_minutes=240)
                    variants.append(out2)
                    produced += 1

    return variants

```

---

