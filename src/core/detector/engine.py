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

    if cfg.ema200_continuation.enabled and hasattr(library, "run_ema200_continuation_family"):
        outcomes.extend(
            library.run_ema200_continuation_family(df_5m_ind, **cfg.ema200_continuation.params)
        )

    return outcomes
