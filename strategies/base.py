"""
strategies/base.py — Classe abstraite pour toutes les stratégies.

Pour créer une nouvelle stratégie :
  1. Hériter de Strategy
  2. Implémenter on_tick() → retourne Signal ou None
  3. Optionnel : on_window_start(), on_window_end()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class Direction(Enum):
    UP   = "UP"    # acheter token UP
    DOWN = "DOWN"  # acheter token DOWN (= vendre UP)


@dataclass
class Signal:
    """Signal de trading émis par une stratégie (market order)."""
    direction:   Direction
    size_usdc:   float              # montant à investir en USDC
    confidence:  float = 1.0        # [0,1] — utilisé pour sizing
    reason:      str   = ""         # description du signal
    stop_loss:   Optional[float] = None   # prix up_mid stop (ex: 0.20)
    take_profit: Optional[float] = None  # prix up_mid take profit


@dataclass
class LimitSignal:
    """Signal de trading avec limit order (maker, 0 frais)."""
    direction:   Direction
    size_usdc:   float
    limit_price: float              # prix auquel on pose le limit
    queue_model: str   = "conservative"  # "conservative" ou "optimistic"
    confidence:  float = 1.0
    reason:      str   = ""
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class ExitSignal:
    """Signal de sortie anticipée d'une position."""
    slug:   str
    reason: str = "signal_exit"


class Strategy(ABC):
    """
    Classe abstraite pour toutes les stratégies de backtest.

    Le moteur appelle :
      - on_window_start(slug, tick) : au début de chaque fenêtre
      - on_tick(slug, tick, has_position) : à chaque tick (0.25s)
      - on_window_end(slug, tick) : à la clôture de la fenêtre
    """

    def __init__(self, name: str = "strategy"):
        self.name = name

    def on_window_start(self, slug: str, tick: pd.Series) -> None:
        """Appelé au premier tick d'une nouvelle fenêtre."""
        pass

    @abstractmethod
    def on_tick(
        self,
        slug:         str,
        tick:         pd.Series,
        has_position: bool,
    ) -> Optional[Signal | LimitSignal | ExitSignal]:
        """
        Appelé à chaque tick.

        Retourner :
          - Signal      → ouvrir une position
          - ExitSignal  → fermer la position existante
          - None        → ne rien faire
        """

    def on_window_end(self, slug: str, tick: pd.Series) -> None:
        """Appelé au dernier tick d'une fenêtre."""
        pass

    def reset(self) -> None:
        """Reset l'état interne entre les runs (utile pour walk-forward)."""
        pass
