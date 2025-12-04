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
