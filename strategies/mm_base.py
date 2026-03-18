"""
strategies/mm_base.py — Classe abstraite pour les stratégies de Market Making.

Différence vs Strategy :
  - on_tick retourne MMQuotes (les ordres à maintenir) pas un Signal
  - Accès à l'inventaire courant pour décider du skew
  - Pas de notion d'entrée/sortie — gestion continue de la liquidité
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class MMQuote:
    """Un ordre resting individuel."""
    side:      str    # "bid" ou "ask"
    direction: str    # "UP" ou "DOWN"
    price:     float
    size_usdc: float


@dataclass
class MMQuotes:
    """
    Les 4 quotes qu'un MM peut maintenir simultanément.
    None = ordre absent (retiré du book).
    
    Typiquement :
      bid_up   : on achète des tokens UP (paris UP monte)
      ask_up   : on vend des tokens UP
      bid_down : on achète des tokens DOWN
      ask_down : on vend des tokens DOWN
    """
    bid_up:   Optional[MMQuote] = None
    ask_up:   Optional[MMQuote] = None
    bid_down: Optional[MMQuote] = None
    ask_down: Optional[MMQuote] = None

    def active_count(self) -> int:
        return sum(1 for q in [self.bid_up, self.ask_up, self.bid_down, self.ask_down] if q)


@dataclass
class MMFill:
    """Information sur un fill reçu par la stratégie (pour adapter les quotes)."""
    side:      str
    direction: str
    price:     float
    size_usdc: float
    spread_pnl: float


class MMInventory:
    """État de l'inventaire (importé depuis mm_engine pour type hints)."""
    pass


class MMStrategy(ABC):
    """
    Classe abstraite pour les stratégies de Market Making.

    Le moteur appelle à chaque tick :
      on_tick(slug, tick_dict, inventory, bn_ret) → MMQuotes

    La stratégie reçoit l'inventaire courant et décide :
      - Quels ordres placer/maintenir/retirer
      - Quel prix (skew selon signal)
      - Quelle taille
    """

    def __init__(self, name: str = "mm_strategy"):
        self.name = name

    def on_window_start(self, slug: str, tick: dict) -> None:
        pass

    @abstractmethod
    def on_tick(
        self,
        slug:      str,
        tick:      dict,
        inventory, # MMInventory
        bn_ret:    float,
    ) -> Optional[MMQuotes]:
        """
        Retourner les quotes à maintenir ce tick.
        None = ne rien faire (garder les quotes précédentes).
        MMQuotes avec None = retirer cet ordre.
        """

    def on_window_end(self, slug: str, tick: dict) -> None:
        pass

    def reset(self) -> None:
        pass
