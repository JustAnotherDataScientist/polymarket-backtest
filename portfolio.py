"""
portfolio.py — Gestion des positions et PnL pour le backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from execution import Fill, compute_pnl


@dataclass
class Trade:
    """Un trade complet (entrée + sortie)."""
    slug:           str
    coin:           str
    interval_min:   str
    entry_ts:       pd.Timestamp
    entry_fill:     Fill
    exit_ts:        Optional[pd.Timestamp] = None
    exit_fill:      Optional[Fill]         = None
    outcome:        Optional[str]          = None
    exit_reason:    str                    = ""
    direction:      str                    = "UP"  # "UP" ou "DOWN"

    @property
    def pnl(self) -> float:
        if self.outcome is None and self.exit_fill is None:
            return 0.0
        return compute_pnl(self.entry_fill, self.exit_fill,
                           self.outcome or "0", direction=self.direction)

    @property
    def pnl_bps(self) -> float:
        cost = self.entry_fill.filled_size
        return (self.pnl / cost * 10000) if cost > 0 else 0.0

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.exit_ts is None:
            return None
        return (self.exit_ts - self.entry_ts).total_seconds()

    def is_closed(self) -> bool:
        return self.exit_fill is not None or self.outcome is not None


@dataclass
class Position:
    """Position ouverte sur un slug."""
    slug:        str
    coin:        str
    interval_min: str
    entry_ts:    pd.Timestamp
    entry_fill:  Fill
    tokens:      float   # tokens détenus
    direction:   str     # "UP" ou "DOWN"

    @property
    def cost(self) -> float:
        return self.entry_fill.net_cost

    def current_value(self, mid: float) -> float:
        return self.tokens * mid

    def unrealized_pnl(self, mid: float) -> float:
        return self.current_value(mid) - self.cost


class Portfolio:
    """
    Gère les positions ouvertes, le cash, et l'historique des trades.
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions:  dict[str, Position] = {}   # slug → Position
        self.trades:     list[Trade]         = []

    # ── Entrée ───────────────────────────────────────────────────────────────

    def open(self, slug: str, coin: str, interval_min: str,
             ts: pd.Timestamp, fill: Fill, direction: str = "UP") -> bool:
        """Ouvre une position. Retourne False si cash insuffisant."""
        if fill.net_cost > self.cash:
            return False
        if slug in self.positions:
            return False

        self.cash -= fill.net_cost
        tokens = fill.filled_size / fill.avg_price
        self.positions[slug] = Position(
            slug=slug, coin=coin, interval_min=interval_min,
            entry_ts=ts, entry_fill=fill, tokens=tokens,
            direction=direction
        )
        return True

    # ── Sortie ────────────────────────────────────────────────────────────────

    def close(self, slug: str, ts: pd.Timestamp,
              exit_fill: Optional[Fill], outcome: Optional[str],
              reason: str = "close") -> Optional[Trade]:
        """Ferme une position et enregistre le trade."""
        pos = self.positions.pop(slug, None)
        if pos is None:
            return None

        trade = Trade(
            slug=slug, coin=pos.coin, interval_min=pos.interval_min,
            entry_ts=pos.entry_ts, entry_fill=pos.entry_fill,
            exit_ts=ts, exit_fill=exit_fill,
            outcome=outcome, exit_reason=reason,
            direction=getattr(pos, "direction", "UP"),
        )

        # Récupère le cash
        if exit_fill is not None:
            self.cash += exit_fill.filled_size - exit_fill.fee
        elif outcome is not None:
            # Clôture à maturité — le token vaut 1.0 si la direction gagne
            direction = getattr(pos, "direction", "UP")
            if direction == "UP":
                outcome_val = 1.0 if str(outcome) in ("1", "1.0") else 0.0
            else:  # DOWN
                outcome_val = 1.0 if str(outcome) in ("0", "0.0") else 0.0
            self.cash += pos.tokens * outcome_val

        self.trades.append(trade)
        return trade

    # ── État ──────────────────────────────────────────────────────────────────

    def total_value(self, mid_prices: dict[str, float] = None) -> float:
        """Valeur totale (cash + positions marked-to-market)."""
        val = self.cash
        if mid_prices:
            for slug, pos in self.positions.items():
                mid = mid_prices.get(slug)
                if mid:
                    val += pos.current_value(mid)
                else:
                    val += pos.cost  # fallback : coût
        return val

    @property
    def n_open(self) -> int:
        return len(self.positions)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def has_position(self, slug: str) -> bool:
        return slug in self.positions

    # ── Résumé ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.is_closed()]
        if not closed:
            return {"n_trades": 0, "total_pnl": 0, "win_rate": 0}

        pnls    = [t.pnl for t in closed]
        winners = [p for p in pnls if p > 0]

        return {
            "n_trades":        len(closed),
            "total_pnl":       sum(pnls),
            "total_pnl_pct":   sum(pnls) / self.initial_capital * 100,
            "win_rate":        len(winners) / len(pnls) * 100,
            "avg_pnl":         sum(pnls) / len(pnls),
            "avg_pnl_bps":     sum(t.pnl_bps for t in closed) / len(closed),
            "best_trade":      max(pnls),
            "worst_trade":     min(pnls),
            "n_open":          self.n_open,
            "cash":            self.cash,
        }
