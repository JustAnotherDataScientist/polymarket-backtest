"""
strategies/validation.py — Stratégies de validation du moteur de backtest.

3 stratégies pour prouver que le moteur est correct :

1. OracleStrategy     → connaît l'outcome à l'avance → doit gagner ~97% des trades
2. RandomStrategy     → entre aléatoirement → doit perdre ~fees (win rate ~50%)
3. AlwaysWrongStrategy → entre toujours du mauvais côté → doit perdre ~97% des trades

Si ces 3 tests passent, le moteur est sain (pas de lookahead bias, PnL correct).
"""
from __future__ import annotations

import random
from typing import Optional
import pandas as pd

from strategies.base import Strategy, Signal, ExitSignal, Direction


class OracleStrategy(Strategy):
    """
    Connaît l'outcome à l'avance (lookahead autorisé ici, c'est voulu).
    Entre toujours du bon côté dès le premier tick.

    Résultat attendu :
      - Win rate : ~100% (perd seulement si pas de liquidité)
      - PnL/trade : (1/entry_price - 1) * size - fees
      - Prouve que le moteur calcule correctement le PnL
    """

    def __init__(self, size_usdc: float = 100.0):
        super().__init__(name="Oracle")
        self.size_usdc   = size_usdc
        self._outcomes:  dict[str, str] = {}   # slug → outcome pré-chargé
        self._entered:   set[str]       = set()

    def on_window_start(self, slug: str, tick: pd.Series) -> None:
        # Pré-charge l'outcome depuis la fenêtre (lookahead intentionnel)
        pass

    def set_outcome(self, slug: str, outcome: str) -> None:
        """Appelé par le moteur avant le premier tick (pour les tests)."""
        self._outcomes[slug] = outcome

    def on_tick(self, slug: str, tick: pd.Series,
                has_position: bool) -> Optional[Signal]:
        if has_position or slug in self._entered:
            return None

        outcome = self._outcomes.get(slug) or tick.get("outcome")
        if outcome not in ("0", "1", "0.0", "1.0"):
            return None

        # Vérifier qu'on a de la liquidité
        up_ask = tick.get("up_ask")
        if up_ask is None or pd.isna(up_ask):
            return None

        # N'entre QUE si UP gagne — c'est le comportement oracle parfait
        if str(outcome) not in ("1", "1.0"):
            return None

        self._entered.add(slug)
        return Signal(
            direction = Direction.UP,
            size_usdc = self.size_usdc,
            reason    = f"oracle outcome={outcome}",
        )

    def reset(self) -> None:
        # Ne pas effacer _outcomes — ils sont pré-chargés avant le run
        self._entered.clear()


class RandomStrategy(Strategy):
    """
    Entre aléatoirement sur ~50% des fenêtres.

    Résultat attendu :
      - Win rate : ~50% (±5% avec 200+ trades)
      - PnL moyen : négatif (~-2*fees par trade)
      - Prouve l'absence de lookahead bias
    """

    def __init__(self, size_usdc: float = 100.0, entry_prob: float = 0.3,
                 seed: int = 42):
        super().__init__(name="Random")
        self.size_usdc   = size_usdc
        self.entry_prob  = entry_prob
        self.seed        = seed
        self._rng        = random.Random(seed)
        self._entered:   set[str] = set()

    def on_tick(self, slug: str, tick: pd.Series,
                has_position: bool) -> Optional[Signal]:
        if has_position or slug in self._entered:
            return None

        # Entre avec probabilité entry_prob au premier tick éligible
        sec = tick.get("seconds_remaining")
        if sec is None or pd.isna(sec) or float(sec) < 30:
            return None

        if self._rng.random() < self.entry_prob:
            self._entered.add(slug)
            return Signal(
                direction = Direction.UP,
                size_usdc = self.size_usdc,
                reason    = "random",
            )
        return None

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._entered.clear()


class AlwaysWrongStrategy(Strategy):
    """
    Connaît l'outcome et entre toujours du MAUVAIS côté.

    Résultat attendu :
      - Win rate : ~0%
      - PnL : -(entry_price * size) - fees par trade
      - Prouve que les pertes sont correctement calculées
    """

    def __init__(self, size_usdc: float = 100.0):
        super().__init__(name="AlwaysWrong")
        self.size_usdc  = size_usdc
        self._entered:  set[str] = set()
        self._outcomes: dict[str, str] = {}

    def set_outcome(self, slug: str, outcome: str) -> None:
        self._outcomes[slug] = outcome

    def on_tick(self, slug: str, tick: pd.Series,
                has_position: bool) -> Optional[Signal]:
        if has_position or slug in self._entered:
            return None

        # Utilise l'outcome pré-chargé
        outcome = self._outcomes.get(slug) or tick.get("outcome")
        if outcome not in ("0", "1", "0.0", "1.0"):
            return None

        up_ask = tick.get("up_ask")
        if up_ask is None or pd.isna(up_ask):
            return None

        # Entre UP quand DOWN gagne → perd toujours
        if str(outcome) in ("0", "0.0"):
            self._entered.add(slug)
            return Signal(
                direction = Direction.UP,
                size_usdc = self.size_usdc,
                reason    = "always_wrong",
            )
        return None

    def reset(self) -> None:
        self._outcomes.clear()
        self._entered.clear()
