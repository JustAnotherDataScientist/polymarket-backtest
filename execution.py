"""
execution.py — Simulation précise d'exécution Polymarket.

Nouveautés v3 :
  - _build_ask_levels / _build_bid_levels lisent le book JSON complet
    (colonne up_book_asks / up_book_bids désérialisée par data.py)
    avec fallback L1 + L2-L5 pour tick synthétiques des tests
  - simulate_limit_buy / simulate_limit_buy_down : limit orders maker (0 frais)
    queue_model "conservative" (best_ask < limit) ou "optimistic" (best_ask <= limit)
  - simulate_buy_down / simulate_sell_down : market orders DOWN token
  - compute_pnl supporte direction UP et DOWN
"""
from __future__ import annotations
from data import get_book

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Frais réels Polymarket — marchés crypto 5min et 15min
# Source : docs.polymarket.com/trading/fees
#
# Formule taker : fee = size_usdc × feeRate × (p × (1-p))^exponent
#   feeRate  = 0.25
#   exponent = 2
#   → taux effectif max = 1.5625% à p=0.50, ~0% aux extrêmes
#
# Maker : rebate = 20% des frais taker collectés (redistribué quotidiennement)
#   → coût net maker = 0 USDC (le rebate compense ou dépasse)
#   → dans le backtest on modélise maker fee = 0 (conservatif)
# ─────────────────────────────────────────────────────────────────────────────

FEE_RATE      = 0.25   # paramètre feeRate Polymarket crypto
FEE_EXPONENT  = 2      # paramètre exponent Polymarket crypto
MAKER_REBATE_PCT = 0.20  # 20% des frais taker redistribués aux makers

# Legacy — gardé pour compatibilité (ne plus utiliser)
TAKER_FEE_BPS = 2.0   # DEPRECATED — utiliser poly_taker_fee() à la place
MAKER_FEE_BPS = 0.0


def poly_taker_fee(size_usdc: float, avg_price: float) -> float:
    """
    Calcule les frais taker réels Polymarket pour les marchés crypto 5min/15min.

    fee = size_usdc × 0.25 × (avg_price × (1 - avg_price))^2

    Taux effectif : 0% aux extrêmes, max 1.5625% à p=0.50
    Conforme à docs.polymarket.com/trading/fees

    Args:
        size_usdc  : montant tradé en USDC
        avg_price  : prix moyen du fill ∈ (0, 1)
    Returns:
        fee en USDC (≥ 0, arrondi à 4 décimales)
    """
    if avg_price <= 0 or avg_price >= 1:
        return 0.0
    rate = FEE_RATE * (avg_price * (1 - avg_price)) ** FEE_EXPONENT
    fee  = size_usdc * rate
    return round(fee, 4)


def poly_maker_rebate(taker_fee: float) -> float:
    """Rebate maker = 20% des frais taker (redistribué quotidiennement)."""
    return round(taker_fee * MAKER_REBATE_PCT, 4)


@dataclass
class Fill:
    requested_size:    float
    filled_size:       float
    avg_price:         float
    fee:               float
    levels_used:       list = field(default_factory=list)
    partial:           bool  = False
    slippage_bps:      float = 0.0
    market_impact_bps: float = 0.0

    @property
    def net_cost(self) -> float:
        return self.filled_size + self.fee

    @property
    def net_revenue(self) -> float:
        return self.filled_size - self.fee

    @property
    def fill_rate(self) -> float:
        return self.filled_size / self.requested_size if self.requested_size > 0 else 0.0


def _valid(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers : construction du book à partir d'un tick
# ─────────────────────────────────────────────────────────────────────────────

def _build_ask_levels(tick: dict) -> list:
    """
    Niveaux ask UP.
    Priorité 1 : up_book_asks (liste Python [[price, size], ...] désérialisée).
    Fallback : up_ask/up_best_ask_size + ask_l2..l5 (vieux format / tests).
    """
    book = get_book(tick, "up_book_asks")
    if book:
        return [(float(p), float(s)) for p, s in book if float(s) > 0]

    levels = []
    p1, s1 = tick.get("up_ask"), tick.get("up_best_ask_size")
    if _valid(p1) and _valid(s1) and float(s1) > 0:
        levels.append((float(p1), float(s1)))
    for i in range(2, 6):
        p = tick.get(f"ask_l{i}_price")
        s = tick.get(f"ask_l{i}_size")
        if _valid(p) and _valid(s) and float(s) > 0:
            levels.append((float(p), float(s)))
    return levels


def _build_bid_levels(tick: dict) -> list:
    """Niveaux bid UP (même logique)."""
    book = get_book(tick, "up_book_bids")
    if book:
        return [(float(p), float(s)) for p, s in book if float(s) > 0]

    levels = []
    p1, s1 = tick.get("up_bid"), tick.get("up_best_bid_size")
    if _valid(p1) and _valid(s1) and float(s1) > 0:
        levels.append((float(p1), float(s1)))
    for i in range(2, 6):
        p = tick.get(f"bid_l{i}_price")
        s = tick.get(f"bid_l{i}_size")
        if _valid(p) and _valid(s) and float(s) > 0:
            levels.append((float(p), float(s)))
    return levels


def _build_down_ask_levels(tick: dict) -> list:
    """Niveaux ask DOWN."""
    book = get_book(tick, "down_book_asks")
    if book:
        return [(float(p), float(s)) for p, s in book if float(s) > 0]
    levels = []
    p1, s1 = tick.get("down_ask"), tick.get("down_best_ask_size")
    if _valid(p1) and _valid(s1) and float(s1) > 0:
        levels.append((float(p1), float(s1)))
    return levels


def _build_down_bid_levels(tick: dict) -> list:
    """Niveaux bid DOWN."""
    book = get_book(tick, "down_book_bids")
    if book:
        return [(float(p), float(s)) for p, s in book if float(s) > 0]
    levels = []
    p1, s1 = tick.get("down_bid"), tick.get("down_best_bid_size")
    if _valid(p1) and _valid(s1) and float(s1) > 0:
        levels.append((float(p1), float(s1)))
    return levels


def _walk_book(levels: list, order_usdc: float) -> tuple:
    """Walk the book. Returns (filled_usdc, total_tokens, levels_used, remaining)."""
    remaining    = order_usdc
    total_tokens = 0.0
    levels_used  = []
    for price, avail_usdc in levels:
        if remaining <= 0:
            break
        fill_usdc    = min(remaining, avail_usdc)
        fill_tokens  = fill_usdc / price if price > 0 else 0
        total_tokens += fill_tokens
        levels_used.append((price, fill_usdc))
        remaining -= fill_usdc
    filled_usdc = order_usdc - remaining
    return filled_usdc, total_tokens, levels_used, remaining


# ─────────────────────────────────────────────────────────────────────────────
# Market orders — UP
# ─────────────────────────────────────────────────────────────────────────────

def simulate_buy(tick: dict, order_size_usdc: float) -> Optional[Fill]:
    """Achat token UP — walk ask book (taker, fee = 2bps)."""
    levels = _build_ask_levels(tick)
    if not levels:
        return None
    best_price = levels[0][0]
    filled, tokens, used, remaining = _walk_book(levels, order_size_usdc)
    if filled <= 0:
        return None

    avg_price    = filled / tokens if tokens > 0 else best_price
    fee          = poly_taker_fee(filled, avg_price)
    slippage_bps = (avg_price - best_price) / best_price * 10000 if best_price > 0 else 0

    depth5     = tick.get("up_ask_depth_top5") or 0
    impact_bps = 0.0
    if depth5 > 0:
        pct = filled / float(depth5)
        if pct > 0.3:
            impact_bps = pct * 15

    return Fill(
        requested_size=order_size_usdc, filled_size=filled,
        avg_price=avg_price, fee=fee, levels_used=used,
        partial=(remaining > 1e-6),
        slippage_bps=slippage_bps, market_impact_bps=impact_bps,
    )


def simulate_sell(tick: dict, order_size_usdc: float) -> Optional[Fill]:
    """
    Vente token UP — walk bid book (taker, fee = 2bps).
    order_size_usdc : valeur nominale USDC des tokens à vendre.
    """
    levels = _build_bid_levels(tick)
    if not levels:
        return None
    best_price        = levels[0][0]
    remaining_nominal = order_size_usdc
    usdc_received     = 0.0
    tokens_sold       = 0.0
    levels_used       = []

    for price, avail_usdc in levels:
        if remaining_nominal <= 0:
            break
        fill_nominal  = min(remaining_nominal, avail_usdc)
        fill_tokens   = fill_nominal / price if price > 0 else 0
        fill_received = fill_tokens * price
        usdc_received += fill_received
        tokens_sold   += fill_tokens
        levels_used.append((price, fill_nominal))
        remaining_nominal -= fill_nominal

    nominal_filled = order_size_usdc - remaining_nominal
    if nominal_filled <= 0:
        return None

    avg_price    = usdc_received / tokens_sold if tokens_sold > 0 else best_price
    fee          = poly_taker_fee(usdc_received, avg_price)
    slippage_bps = (best_price - avg_price) / best_price * 10000 if best_price > 0 else 0

    depth5     = tick.get("up_bid_depth_top5") or 0
    impact_bps = 0.0
    if depth5 > 0:
        pct = nominal_filled / float(depth5)
        if pct > 0.3:
            impact_bps = pct * 15

    return Fill(
        requested_size=order_size_usdc, filled_size=usdc_received,
        avg_price=avg_price, fee=fee, levels_used=levels_used,
        partial=(remaining_nominal > 1e-6),
        slippage_bps=slippage_bps, market_impact_bps=impact_bps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Market orders — DOWN
# ─────────────────────────────────────────────────────────────────────────────

def simulate_buy_down(tick: dict, order_size_usdc: float) -> Optional[Fill]:
    """Achat token DOWN — walk down ask book (taker, fee = 2bps)."""
    levels = _build_down_ask_levels(tick)
    if not levels:
        return None
    best_price = levels[0][0]
    filled, tokens, used, remaining = _walk_book(levels, order_size_usdc)
    if filled <= 0:
        return None

    avg_price    = filled / tokens if tokens > 0 else best_price
    fee          = poly_taker_fee(filled, avg_price)
    slippage_bps = (avg_price - best_price) / best_price * 10000 if best_price > 0 else 0

    depth5     = tick.get("down_ask_depth_top5") or 0
    impact_bps = 0.0
    if depth5 > 0:
        pct = filled / float(depth5)
        if pct > 0.3:
            impact_bps = pct * 15

    return Fill(
        requested_size=order_size_usdc, filled_size=filled,
        avg_price=avg_price, fee=fee, levels_used=used,
        partial=(remaining > 1e-6),
        slippage_bps=slippage_bps, market_impact_bps=impact_bps,
    )


def simulate_sell_down(tick: dict, order_size_usdc: float) -> Optional[Fill]:
    """Vente token DOWN — walk down bid book (taker, fee = 2bps)."""
    levels = _build_down_bid_levels(tick)
    if not levels:
        return None
    best_price        = levels[0][0]
    remaining_nominal = order_size_usdc
    usdc_received     = 0.0
    tokens_sold       = 0.0
    levels_used       = []

    for price, avail_usdc in levels:
        if remaining_nominal <= 0:
            break
        fill_nominal  = min(remaining_nominal, avail_usdc)
        fill_tokens   = fill_nominal / price if price > 0 else 0
        fill_received = fill_tokens * price
        usdc_received += fill_received
        tokens_sold   += fill_tokens
        levels_used.append((price, fill_nominal))
        remaining_nominal -= fill_nominal

    nominal_filled = order_size_usdc - remaining_nominal
    if nominal_filled <= 0:
        return None

    avg_price    = usdc_received / tokens_sold if tokens_sold > 0 else best_price
    fee          = poly_taker_fee(usdc_received, avg_price)
    slippage_bps = (best_price - avg_price) / best_price * 10000 if best_price > 0 else 0

    return Fill(
        requested_size=order_size_usdc, filled_size=usdc_received,
        avg_price=avg_price, fee=fee, levels_used=levels_used,
        partial=(remaining_nominal > 1e-6),
        slippage_bps=slippage_bps, market_impact_bps=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Limit orders — UP
# ─────────────────────────────────────────────────────────────────────────────

def simulate_limit_buy(
    tick_sequence: list[dict],
    limit_price: float,
    size_usdc: float,
    queue_model: str = "conservative",
) -> Optional[Fill]:
    """
    Simule un limit buy sur le token UP (maker, fee = 0).

    Rejoue tick par tick depuis le signal jusqu'à fin de fenêtre.
    Retourne None si pas de fill (ordre annulé).

    queue_model :
      "conservative" : best_ask < limit_price
        → le prix doit passer SOUS notre limite (on est en queue)
      "optimistic"   : best_ask <= limit_price
        → fill dès que le prix touche notre limite (on est en tête)
    """
    for tick in tick_sequence:
        asks = get_book(tick, "up_book_asks")
        if not asks:
            # Fallback tick synthétique
            p = tick.get("up_ask")
            if p is None:
                continue
            asks = [[float(p), tick.get("up_best_ask_size") or 0]]

        if not asks:
            continue
        best_ask = float(asks[0][0])

        triggered = (best_ask < limit_price) if queue_model == "conservative" \
                    else (best_ask <= limit_price)

        if triggered:
            return Fill(
                requested_size=size_usdc,
                filled_size=size_usdc,
                avg_price=limit_price,
                fee=0.0,                          # maker = 0 frais
                levels_used=[(limit_price, size_usdc)],
                partial=False,
                slippage_bps=0.0,
                market_impact_bps=0.0,
            )

    return None  # pas de fill


# ─────────────────────────────────────────────────────────────────────────────
# Limit orders — DOWN
# ─────────────────────────────────────────────────────────────────────────────

def simulate_limit_buy_down(
    tick_sequence: list[dict],
    limit_price: float,
    size_usdc: float,
    queue_model: str = "conservative",
) -> Optional[Fill]:
    """
    Simule un limit buy sur le token DOWN (maker, fee = 0).
    Même logique que simulate_limit_buy mais utilise down_book_asks.
    """
    for tick in tick_sequence:
        asks = get_book(tick, "down_book_asks")
        if not asks:
            p = tick.get("down_ask")
            if p is None:
                continue
            asks = [[float(p), tick.get("down_best_ask_size") or 0]]

        if not asks:
            continue
        best_ask = float(asks[0][0])

        triggered = (best_ask < limit_price) if queue_model == "conservative" \
                    else (best_ask <= limit_price)

        if triggered:
            return Fill(
                requested_size=size_usdc,
                filled_size=size_usdc,
                avg_price=limit_price,
                fee=0.0,
                levels_used=[(limit_price, size_usdc)],
                partial=False,
                slippage_bps=0.0,
                market_impact_bps=0.0,
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# PnL
# ─────────────────────────────────────────────────────────────────────────────

def compute_pnl(
    entry: Fill,
    exit_fill: Optional[Fill],
    outcome: str,
    direction: str = "UP",
) -> float:
    """
    Calcule le PnL d'un trade.

    direction :
      "UP"   : gagne si outcome = "1" (token UP expire à 1.0)
      "DOWN" : gagne si outcome = "0" (token DOWN expire à 1.0)

    Si exit_fill fourni → exit anticipé (outcome ignoré).
    Sinon → hold à clôture (token vaut 1.0 ou 0.0).
    """
    tokens_bought = entry.filled_size / entry.avg_price if entry.avg_price > 0 else 0
    outcome_str   = str(outcome)

    if direction == "UP":
        outcome_val = 1.0 if outcome_str in ("1", "1.0") else 0.0
    else:  # DOWN
        outcome_val = 1.0 if outcome_str in ("0", "0.0") else 0.0

    if exit_fill is not None:
        revenue = exit_fill.filled_size - exit_fill.fee
        cost    = entry.filled_size + entry.fee
        return revenue - cost
    else:
        return tokens_bought * outcome_val - entry.net_cost
