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
