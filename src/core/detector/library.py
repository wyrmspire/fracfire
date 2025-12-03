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
