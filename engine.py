"""
engine.py — Moteur de backtest event-driven avec latence configurable.

Paramètres clés :
  latency_ms      : délai entre signal et fill (0 = instantané, 250 = 1 tick, etc.)
  market_impact   : modéliser l'impact des gros ordres sur le book
  max_positions   : positions simultanées max
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional
import pandas as pd

from data import iter_windows, iter_windows_streaming
from execution import simulate_buy, simulate_sell, simulate_buy_down, simulate_sell_down, \
    simulate_limit_buy, simulate_limit_buy_down, Fill
from portfolio import Portfolio
from metrics import compute_metrics, print_report, equity_curve
from strategies.base import Strategy, Signal, LimitSignal, ExitSignal, Direction

logger = logging.getLogger(__name__)

TICK_MS = 250   # ms par tick (4 Hz)

# NOTE latence critique :
# latency_ms=0 et latency_ms=250 sont IDENTIQUES en pratique.
# L'ordre est créé en step 4 (signal), mais les fills sont traités en step 1b
# au tick suivant. Donc la latence minimale réelle est toujours 1 tick = 250ms.
# Valeurs significatives : 250ms, 500ms, 750ms, 1000ms


class PendingOrder:
    """Ordre market en attente d'exécution (latence)."""
    __slots__ = ("signal", "slug", "coin", "interval_min", "submit_ts", "fill_after_ts")

    def __init__(self, signal, slug, coin, interval_min, submit_ts, latency_ms):
        self.signal       = signal
        self.slug         = slug
        self.coin         = coin
        self.interval_min = interval_min
        self.submit_ts    = submit_ts
        self.fill_after_ts = submit_ts + pd.Timedelta(milliseconds=latency_ms)


class PendingLimitOrder:
    """
    Limit order en attente.
    Collecte les ticks depuis submit_ts et tente le fill à chaque tick.
    L'ordre est annulé à la fin de la fenêtre s'il n'est pas rempli.
    """
    __slots__ = ("signal", "slug", "coin", "interval_min", "submit_ts",
                 "fill_after_ts", "tick_buffer")

    def __init__(self, signal, slug, coin, interval_min, submit_ts, latency_ms):
        self.signal        = signal
        self.slug          = slug
        self.coin          = coin
        self.interval_min  = interval_min
        self.submit_ts     = submit_ts
        # On commence à tenter le fill après latency_ms (pose de l'ordre)
        self.fill_after_ts = submit_ts + pd.Timedelta(milliseconds=latency_ms)
        self.tick_buffer: list[dict] = []  # ticks accumulés depuis fill_after_ts


def run(
    df:              pd.DataFrame,
    strategy:        Strategy,
    initial_capital: float = 10_000.0,
    latency_ms:      float = 250.0,    # latence en ms (min réel = 250ms = 1 tick)
    market_impact:   bool  = True,     # ← NOUVEAU : modéliser impact
    verbose:         bool  = False,
    max_positions:   int   = 4,
    allow_multi:     bool  = False,
) -> dict:
    """
    Lance le backtest.

    latency_ms = 0     → fill instantané (optimiste)
    latency_ms = 250   → 1 tick de délai (= 250ms, latence réseau standard)
    latency_ms = 500   → 2 ticks (VM distante)
    latency_ms = 1000  → 4 ticks (connexion lente)
    """
    portfolio = Portfolio(initial_capital=initial_capital)
    strategy.reset()

    pending_orders: deque[PendingOrder | PendingLimitOrder] = deque()
    n_windows = 0
    n_skipped = 0

    for window in iter_windows(df):
        slug         = window["slug"].iloc[0]
        coin         = window["coin"].iloc[0]
        interval_min = str(window["interval_min"].iloc[0])
        outcome_s    = window["outcome"].dropna()
        outcome_val  = str(outcome_s.iloc[-1]) if len(outcome_s) > 0 else None

        if outcome_val is None:
            n_skipped += 1
            continue

        n_windows += 1
        strategy.on_window_start(slug, window.iloc[0])

        for _, tick in window.iterrows():
            ts           = tick["timestamp_utc"]
            tick_dict    = tick.to_dict()
            has_position = portfolio.has_position(slug)

            # ── 1a. Market orders : fill dès que la latence est écoulée ──────
            executed = []
            for order in list(pending_orders):
                if order.slug != slug:
                    continue
                if not isinstance(order, PendingOrder):
                    continue
                if ts >= order.fill_after_ts:
                    if not portfolio.has_position(order.slug):
                        sig = order.signal
                        if sig.direction == Direction.UP:
                            fill = simulate_buy(tick_dict, sig.size_usdc)
                        else:
                            fill = simulate_buy_down(tick_dict, sig.size_usdc)
                        if fill:
                            ok = portfolio.open(
                                order.slug, order.coin, order.interval_min,
                                ts, fill, direction=sig.direction.value
                            )
                            if ok and verbose:
                                logger.info(
                                    "[%s] MARKET FILL (after %.0fms) %s %.0f@%.4f slip=%.1fbps",
                                    ts, latency_ms, slug[:25],
                                    fill.filled_size, fill.avg_price,
                                    fill.slippage_bps
                                )
                    executed.append(order)

            for o in executed:
                try: pending_orders.remove(o)
                except ValueError: pass

            # ── 1b. Limit orders : accumuler les ticks + tenter fill ──────────
            limit_executed = []
            for order in list(pending_orders):
                if order.slug != slug:
                    continue
                if not isinstance(order, PendingLimitOrder):
                    continue
                # Accumuler les ticks à partir de fill_after_ts
                if ts >= order.fill_after_ts:
                    order.tick_buffer.append(tick_dict)
                    # Tenter le fill avec le buffer courant (1 tick à la fois)
                    if not portfolio.has_position(order.slug):
                        sig = order.signal
                        if sig.direction == Direction.UP:
                            fill = simulate_limit_buy(
                                [tick_dict], sig.limit_price, sig.size_usdc,
                                sig.queue_model
                            )
                        else:
                            fill = simulate_limit_buy_down(
                                [tick_dict], sig.limit_price, sig.size_usdc,
                                sig.queue_model
                            )
                        if fill:
                            ok = portfolio.open(
                                order.slug, order.coin, order.interval_min,
                                ts, fill, direction=sig.direction.value
                            )
                            if ok:
                                limit_executed.append(order)
                                if verbose:
                                    logger.info(
                                        "[%s] LIMIT FILL (%s) %s %.0f@%.4f fee=0",
                                        ts, sig.queue_model, slug[:25],
                                        fill.filled_size, fill.avg_price
                                    )

            for o in limit_executed:
                try: pending_orders.remove(o)
                except ValueError: pass

            # ── 2. Appelle la stratégie ───────────────────────────────────────
            has_position = portfolio.has_position(slug)
            # Injecter les données du fill dans tick_dict pour la stratégie
            # → permet take_profit/stop_loss calculés depuis le prix réel du fill
            if has_position and slug in portfolio.positions:
                pos = portfolio.positions[slug]
                tick_dict["_entry_fill_price"] = pos.entry_fill.avg_price
                tick_dict["_entry_ts"]          = pos.entry_ts
                tick_dict["_entry_direction"]   = getattr(pos, "direction", "UP")
            signal = strategy.on_tick(slug, tick_dict, has_position)

            if signal is None:
                pass

            # ── 3. Exit signal (pas de latence sur les exits) ─────────────────
            elif isinstance(signal, ExitSignal) and has_position:
                pos       = portfolio.positions[slug]
                direction = getattr(pos, "direction", "UP")
                exit_fill = None
                if direction == "UP":
                    up_mid = tick.get("up_mid")
                    if up_mid is not None and not pd.isna(up_mid):
                        sell_size = pos.tokens * float(up_mid)
                        exit_fill = simulate_sell(tick_dict, sell_size)
                else:  # DOWN
                    down_mid = tick.get("down_mid")
                    if down_mid is not None and not pd.isna(down_mid):
                        sell_size = pos.tokens * float(down_mid)
                        exit_fill = simulate_sell_down(tick_dict, sell_size)
                trade = portfolio.close(
                    slug=slug, ts=ts,
                    exit_fill=exit_fill, outcome=outcome_val,
                    reason=signal.reason
                )
                if verbose and trade:
                    logger.info("[%s] EXIT %s PnL=%+.2f (%s)",
                                ts, slug[:25], trade.pnl, signal.reason)

            # ── 4. Entry signal (market ou limit) → créer ordre en attente ───
            elif isinstance(signal, (Signal, LimitSignal)) and not has_position:
                if portfolio.n_open >= max_positions:
                    pass
                elif not allow_multi and any(
                    portfolio.positions[s].coin == coin
                    for s in portfolio.positions
                ):
                    pass
                else:
                    already_pending = any(o.slug == slug for o in pending_orders)
                    if not already_pending:
                        if isinstance(signal, LimitSignal):
                            order = PendingLimitOrder(
                                signal=signal, slug=slug, coin=coin,
                                interval_min=interval_min,
                                submit_ts=ts, latency_ms=latency_ms
                            )
                        else:
                            order = PendingOrder(
                                signal=signal, slug=slug, coin=coin,
                                interval_min=interval_min,
                                submit_ts=ts, latency_ms=latency_ms
                            )
                        pending_orders.append(order)
                        if verbose:
                            kind = "LIMIT" if isinstance(signal, LimitSignal) else "MARKET"
                            logger.info("[%s] %s SIGNAL %s → pending (%.0fms)",
                                        ts, kind, slug[:25], latency_ms)

        # ── Fin de fenêtre ────────────────────────────────────────────────────
        strategy.on_window_end(slug, window.iloc[-1])

        # Annuler les ordres pending non exécutés (fenêtre fermée)
        cancelled = [o for o in pending_orders if o.slug == slug]
        for o in cancelled:
            try: pending_orders.remove(o)
            except ValueError: pass
        if cancelled and verbose:
            logger.info("[window end] %d pending order(s) cancelled for %s",
                        len(cancelled), slug[:25])

        # Fermer position à maturité
        if portfolio.has_position(slug):
            trade = portfolio.close(
                slug=slug, ts=window.iloc[-1]["timestamp_utc"],
                exit_fill=None, outcome=outcome_val, reason="close"
            )
            if verbose and trade:
                logger.info("[%s] CLOSE %s PnL=%+.2f outcome=%s",
                            window.iloc[-1]["timestamp_utc"],
                            slug[:25], trade.pnl, outcome_val)

    # Fermer les positions restantes
    for slug in list(portfolio.positions.keys()):
        portfolio.close(slug=slug, ts=pd.Timestamp.now(tz="UTC"),
                       exit_fill=None, outcome=None, reason="end_of_data")

    metrics  = compute_metrics(portfolio.trades, initial_capital)
    eq_curve = equity_curve(portfolio.trades, initial_capital)

    if verbose:
        print(f"\nWindows: {n_windows}  |  Skipped: {n_skipped}"
              f"  |  Latency: {latency_ms:.0f}ms"
              f"  |  Market impact: {market_impact}")
        print_report(metrics)

    return {
        "metrics":      metrics,
        "trades":       portfolio.trades,
        "equity_curve": eq_curve,
        "portfolio":    portfolio,
        "n_windows":    n_windows,
        "n_skipped":    n_skipped,
        "config": {
            "latency_ms":    latency_ms,
            "market_impact": market_impact,
            "max_positions": max_positions,
            "initial_capital": initial_capital,
        }
    }


def run_streaming(
    live_dir:        str,
    strategy,
    initial_capital: float = 10_000.0,
    latency_ms:      float = 250.0,
    verbose:         bool  = False,
    max_positions:   int   = 4,
    coin:            str   = None,
    interval:        int   = None,
    start:           str   = None,
    end:             str   = None,
) -> dict:
    """
    Version streaming de run() — RAM constante quelle que soit la durée.
    Lit les parquets fenetre par fenetre sans tout charger en memoire.
    Utiliser pour les backtests sur > 1 jour de donnees.
    """
    portfolio = Portfolio(initial_capital=initial_capital)
    strategy.reset()
    pending: deque = deque()
    n_windows = 0
    n_skipped = 0

    for window in iter_windows_streaming(live_dir, coin, interval, start, end):
        slug         = window["slug"].iloc[0]
        coin_w       = window["coin"].iloc[0]
        iv           = str(window["interval_min"].iloc[0])
        oc_s         = window["outcome"].dropna() if "outcome" in window.columns else pd.Series(dtype=str)
        outcome_val  = str(oc_s.iloc[-1]) if len(oc_s) > 0 else None

        if outcome_val is None:
            n_skipped += 1
            continue

        n_windows += 1
        strategy.on_window_start(slug, window.iloc[0])

        for _, tick in window.iterrows():
            ts        = tick["timestamp_utc"]
            td        = tick.to_dict()
            has_pos   = portfolio.has_position(slug)

            # Fill market orders
            done = []
            for o in list(pending):
                if o.slug != slug or not isinstance(o, PendingOrder): continue
                if ts >= o.fill_after_ts and not portfolio.has_position(o.slug):
                    sig  = o.signal
                    fill = simulate_buy(td, sig.size_usdc) if sig.direction == Direction.UP else simulate_buy_down(td, sig.size_usdc)
                    if fill:
                        portfolio.open(o.slug, o.coin, o.interval_min, ts, fill, direction=sig.direction.value)
                    done.append(o)
            for o in done:
                try: pending.remove(o)
                except ValueError: pass

            # Fill limit orders
            done2 = []
            for o in list(pending):
                if o.slug != slug or not isinstance(o, PendingLimitOrder): continue
                if ts >= o.fill_after_ts and not portfolio.has_position(o.slug):
                    sig  = o.signal
                    fill = simulate_limit_buy([td], sig.limit_price, sig.size_usdc, sig.queue_model) if sig.direction == Direction.UP else simulate_limit_buy_down([td], sig.limit_price, sig.size_usdc, sig.queue_model)
                    if fill:
                        portfolio.open(o.slug, o.coin, o.interval_min, ts, fill, direction=sig.direction.value)
                        done2.append(o)
            for o in done2:
                try: pending.remove(o)
                except ValueError: pass

            has_pos = portfolio.has_position(slug)
            # Injecter données du fill pour take_profit/stop_loss précis
            if has_pos and slug in portfolio.positions:
                pos = portfolio.positions[slug]
                td["_entry_fill_price"] = pos.entry_fill.avg_price
                td["_entry_ts"]          = pos.entry_ts
                td["_entry_direction"]   = getattr(pos, "direction", "UP")
            signal  = strategy.on_tick(slug, td, has_pos)

            if signal is None:
                pass
            elif isinstance(signal, ExitSignal) and has_pos:
                pos       = portfolio.positions[slug]
                direction = getattr(pos, "direction", "UP")
                ef = None
                if direction == "UP":
                    um = tick.get("up_mid")
                    if um is not None and not pd.isna(um):
                        ef = simulate_sell(td, pos.tokens * float(um))
                else:
                    dm = tick.get("down_mid")
                    if dm is not None and not pd.isna(dm):
                        ef = simulate_sell_down(td, pos.tokens * float(dm))
                portfolio.close(slug=slug, ts=ts, exit_fill=ef, outcome=outcome_val, reason=signal.reason)

            elif isinstance(signal, (Signal, LimitSignal)) and not has_pos:
                if portfolio.n_open < max_positions and not any(o.slug == slug for o in pending):
                    if isinstance(signal, LimitSignal):
                        pending.append(PendingLimitOrder(signal=signal, slug=slug, coin=coin_w, interval_min=iv, submit_ts=ts, latency_ms=latency_ms))
                    else:
                        pending.append(PendingOrder(signal=signal, slug=slug, coin=coin_w, interval_min=iv, submit_ts=ts, latency_ms=latency_ms))

        strategy.on_window_end(slug, window.iloc[-1])
        for o in [o for o in pending if o.slug == slug]:
            try: pending.remove(o)
            except ValueError: pass
        if portfolio.has_position(slug):
            portfolio.close(slug=slug, ts=window.iloc[-1]["timestamp_utc"], exit_fill=None, outcome=outcome_val, reason="close")

    metrics  = compute_metrics(portfolio.trades, initial_capital)
    eq_curve = equity_curve(portfolio.trades, initial_capital)
    return {"metrics": metrics, "trades": portfolio.trades, "equity_curve": eq_curve,
            "portfolio": portfolio, "n_windows": n_windows, "n_skipped": n_skipped,
            "config": {"latency_ms": latency_ms, "max_positions": max_positions, "initial_capital": initial_capital}}
