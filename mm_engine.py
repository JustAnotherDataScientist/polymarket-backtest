"""
mm_engine.py — Moteur de backtest Market Making pour Polymarket.

Différences vs engine.py classique :
  - Gère 2 ordres simultanés par slug (bid UP + ask DOWN ou bid UP + bid DOWN)
  - Inventaire continu (tokens accumulés des deux côtés)
  - PnL = spread capturé + PnL directionnel sur inventaire
  - Mode dynamique : la stratégie peut modifier/annuler les quotes chaque tick
  - Fill simulé via le book réel OU via volume observé (poly_trade_size_10s)

Architecture :
  MMQuote      — un ordre resting (bid ou ask, UP ou DOWN)
  MMInventory  — état de l'inventaire par slug
  MMResult     — résultat complet du backtest
  run_mm()     — moteur principal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from data import iter_windows_streaming
from execution import poly_taker_fee, Fill, get_book
from strategies.mm_base import MMStrategy, MMQuotes, MMFill

logger = logging.getLogger(__name__)


@dataclass
class MMQuote:
    """Un ordre resting dans le book."""
    side:        str    # "bid" ou "ask"
    direction:   str    # "UP" ou "DOWN" (quel token)
    price:       float  # prix limite
    size_usdc:   float  # taille en USDC
    placed_ts:   pd.Timestamp = None


@dataclass 
class MMTrade:
    """Un fill de market making."""
    ts:          pd.Timestamp
    slug:        str
    side:        str    # "bid" (on achète) ou "ask" (on vend)
    direction:   str    # "UP" ou "DOWN"
    price:       float
    size_usdc:   float
    fee:         float
    pnl_spread:  float  # PnL immédiat du spread capturé
    reason:      str    # pourquoi ce fill


@dataclass
class MMInventory:
    """Inventaire courant sur un slug."""
    slug:           str
    tokens_up:      float = 0.0   # tokens UP détenus (>0 = long, <0 = short)
    tokens_down:    float = 0.0   # tokens DOWN détenus
    cost_basis_up:  float = 0.0   # coût moyen des tokens UP
    cost_basis_down:float = 0.0   # coût moyen des tokens DOWN
    realized_pnl:   float = 0.0   # PnL réalisé sur ce slug
    n_fills:        int   = 0
    spread_earned:  float = 0.0   # gains de spread accumulés

    def net_inventory_usdc(self, up_mid: float, down_mid: float) -> float:
        """Valeur nette marked-to-market de l'inventaire."""
        return self.tokens_up * up_mid + self.tokens_down * down_mid


def _simulate_mm_fill(
    tick: dict,
    quote: MMQuote,
    prev_trade_count: float,
    curr_trade_count: float,
    vol_10s: float,
) -> bool:
    """
    Détermine si un ordre resting est fillé sur ce tick.

    Modèle réaliste :
      Un fill maker ne se produit QUE quand un taker vient trader contre notre ordre.
      On détecte ça via le changement de poly_trade_count_10s entre deux ticks.
      Si un nouveau trade vient de se produire ET que le volume est suffisant
      pour traverser notre position dans la queue → fill.

    Hypothèses conservatrices :
      - On n'est fillé que si un trade a eu lieu sur ce tick précis
      - Le volume du trade doit être suffisant (> notre taille)
      - 1 fill maximum par "burst" de trades (évite les doubles fills)
    """
    # Un trade vient-il de se produire sur ce tick ?
    new_trades = curr_trade_count - prev_trade_count
    if new_trades < 1: return False

    # Y a-t-il assez de volume pour nous filler ?
    # vol_10s est cumulé → approximation du volume par trade = vol_10s / trade_count
    if curr_trade_count > 0:
        vol_per_trade = vol_10s / curr_trade_count
    else:
        return False

    # Fill si le volume par trade > notre taille (on est en tête de queue approximativement)
    return vol_per_trade > quote.size_usdc * 0.5


def run_mm(
    live_dir:        str,
    strategy:        MMStrategy,
    initial_capital: float = 10_000.0,
    coin:            str   = None,
    interval:        int   = None,
    latency_ms:      float = 250.0,
    fill_model:      str   = "trade_based",  # modèle unique : fill sur nouveau trade
    max_inventory_usdc: float = 500.0,        # limite inventaire par slug
    verbose:         bool  = False,
) -> dict:
    """
    Moteur de backtest Market Making.

    À chaque tick, la stratégie retourne MMQuotes — les deux ordres à maintenir.
    Le moteur simule les fills, met à jour l'inventaire, calcule le PnL.
    """
    cash       = initial_capital
    inventory  = {}   # slug → MMInventory
    quotes        = {}   # slug → MMQuotes (ordres actifs)

    n_windows     = 0
    n_skipped     = 0
    n_trades      = 0
    n_spread_wins = 0
    total_spread_pnl    = 0.0
    total_inventory_pnl = 0.0

    MM_COLS = [
        "slug", "coin", "interval_min", "timestamp_utc", "outcome",
        "up_bid", "up_ask", "up_mid", "down_bid", "down_ask", "down_mid",
        "seconds_remaining", "coin_return_10s_bps", "coin_return_60s_bps",
        "poly_trade_count_10s", "poly_trade_size_10s",
        "up_bid_depth_top5",
    ]

    for window in iter_windows_streaming(live_dir, coin, interval):
        available_cols = [c for c in MM_COLS if c in window.columns]
        window = window[available_cols]

        slug         = window["slug"].iloc[0]
        coin_w       = window["coin"].iloc[0]
        iv           = str(window["interval_min"].iloc[0])
        oc_s         = window["outcome"].dropna() if "outcome" in window.columns else pd.Series(dtype=str)
        outcome_val  = str(oc_s.iloc[-1]) if len(oc_s) > 0 else None

        if outcome_val is None:
            n_skipped += 1
            continue

        n_windows += 1
        inv = inventory.setdefault(slug, MMInventory(slug=slug))

        strategy.on_window_start(slug, window.iloc[0])
        current_quotes: Optional[MMQuotes] = None
        prev_trade_count: float = 0.0

        tick_records = window.to_dict("records")
        last_tick    = window.iloc[-1].to_dict()
        del window

        for td in tick_records:
            ts = td["timestamp_utc"]

            # Colonnes utiles
            up_mid   = float(td.get("up_mid") or 0)
            down_mid = float(td.get("down_mid") or 0)
            vol_10s  = float(td.get("poly_trade_size_10s") or 0)
            bn_ret   = float(td.get("coin_return_10s_bps") or 0)
            curr_trade_count = float(td.get("poly_trade_count_10s") or 0)

            # ── 1. Simuler les fills sur les quotes actives ──────────────────
            if current_quotes is not None:
                for quote in [current_quotes.bid_up, current_quotes.bid_down,
                              current_quotes.ask_up, current_quotes.ask_down]:
                    if quote is None: continue

                    # Vérifier contrainte inventaire
                    inv_val = abs(inv.net_inventory_usdc(up_mid, down_mid))
                    if inv_val >= max_inventory_usdc and quote.side == "bid":
                        continue  # inventaire plein, pas de nouveaux achats

                    # Simuler le fill — seulement si un trade vient de se produire
                    filled = _simulate_mm_fill(td, quote, prev_trade_count,
                                               curr_trade_count, vol_10s)
                    if not filled: continue

                    # Appliquer le fill à l'inventaire
                    fee = 0.0  # maker = 0 frais
                    spread_pnl = 0.0

                    if quote.side == "bid":
                        # On achète des tokens → inventaire augmente
                        tokens = quote.size_usdc / quote.price if quote.price > 0 else 0
                        if quote.direction == "UP":
                            # Mise à jour coût moyen
                            total_tokens = inv.tokens_up + tokens
                            if total_tokens > 0:
                                inv.cost_basis_up = (
                                    inv.cost_basis_up * inv.tokens_up + quote.price * tokens
                                ) / total_tokens
                            inv.tokens_up += tokens
                        else:
                            total_tokens = inv.tokens_down + tokens
                            if total_tokens > 0:
                                inv.cost_basis_down = (
                                    inv.cost_basis_down * inv.tokens_down + quote.price * tokens
                                ) / total_tokens
                            inv.tokens_down += tokens
                        cash -= quote.size_usdc + fee

                    else:  # ask — on vend des tokens
                        tokens = quote.size_usdc / quote.price if quote.price > 0 else 0
                        if quote.direction == "UP" and inv.tokens_up >= tokens:
                            # Spread capturé = (ask - cost_basis) * tokens
                            spread_pnl = (quote.price - inv.cost_basis_up) * tokens
                            inv.tokens_up -= tokens
                            inv.realized_pnl += spread_pnl
                        elif quote.direction == "DOWN" and inv.tokens_down >= tokens:
                            spread_pnl = (quote.price - inv.cost_basis_down) * tokens
                            inv.tokens_down -= tokens
                            inv.realized_pnl += spread_pnl
                        else:
                            continue  # pas assez de tokens à vendre
                        cash += quote.size_usdc - fee

                    inv.n_fills += 1
                    inv.spread_earned += spread_pnl
                    total_spread_pnl += spread_pnl

                    n_trades += 1
                    if spread_pnl > 0:
                        n_spread_wins += 1

                    if verbose:
                        logger.info("[%s] MM FILL %s %s @%.4f size=%.0f spread_pnl=%+.3f",
                                    ts, quote.side, quote.direction, quote.price,
                                    quote.size_usdc, spread_pnl)

            # ── 2. Demander à la stratégie ses nouvelles quotes ──────────────
            current_quotes = strategy.on_tick(slug, td, inv, bn_ret)
            prev_trade_count = curr_trade_count

        # ── Fin de fenêtre : liquider l'inventaire au mid ───────────────────
        strategy.on_window_end(slug, last_tick)

        up_mid_final   = float(last_tick.get("up_mid") or 0)
        down_mid_final = float(last_tick.get("down_mid") or 0)

        # PnL inventaire : tokens × (mid_final - cost_basis)
        pnl_up   = inv.tokens_up   * (up_mid_final - inv.cost_basis_up)
        pnl_down = inv.tokens_down * (down_mid_final - inv.cost_basis_down)

        # Outcome : tokens qui gagnent valent 1.0
        if outcome_val in ("1", "1.0"):
            pnl_up   = inv.tokens_up   * (1.0 - inv.cost_basis_up)
            pnl_down = inv.tokens_down * (0.0 - inv.cost_basis_down)
        else:
            pnl_up   = inv.tokens_up   * (0.0 - inv.cost_basis_up)
            pnl_down = inv.tokens_down * (1.0 - inv.cost_basis_down)

        inv.realized_pnl += pnl_up + pnl_down
        total_inventory_pnl += pnl_up + pnl_down
        cash += inv.tokens_up * (1.0 if outcome_val in ("1","1.0") else 0.0)
        cash += inv.tokens_down * (0.0 if outcome_val in ("1","1.0") else 1.0)

        # Reset inventaire + libérer fenêtre
        del inventory[slug]
        del tick_records
        current_quotes = None

    total_pnl = cash - initial_capital

    return {
        "total_pnl":          total_pnl,
        "total_spread_pnl":   total_spread_pnl,
        "total_inventory_pnl":total_inventory_pnl,
        "n_trades":           n_trades,
        "n_windows":          n_windows,
        "n_skipped":          n_skipped,
        "n_spread_wins":      n_spread_wins,
        "cash":               cash,
    }
