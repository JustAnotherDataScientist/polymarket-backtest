"""
metrics.py — Métriques de performance pour le backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
from portfolio import Trade


def compute_metrics(trades: List[Trade], initial_capital: float = 10_000.0) -> dict:
    """Calcule toutes les métriques de performance."""
    closed = [t for t in trades if t.is_closed()]
    if not closed:
        return {"error": "no closed trades"}

    pnls     = np.array([t.pnl for t in closed])
    pnls_bps = np.array([t.pnl_bps for t in closed])
    costs    = np.array([t.entry_fill.filled_size for t in closed])
    fees     = np.array([t.entry_fill.fee + (t.exit_fill.fee if t.exit_fill else 0) for t in closed])

    # Returns
    returns = pnls / costs
    wins    = pnls > 0
    losses  = pnls < 0

    # Sharpe (annualisé, base journalière)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12)  # 5min windows
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = cumulative - running_max
    max_drawdown = drawdowns.min()

    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if wins.any() else 0
    gross_loss   = abs(pnls[pnls < 0].sum()) if losses.any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Fill quality
    fill_rates = [t.entry_fill.fill_rate for t in closed]
    partial_fills = sum(1 for t in closed if t.entry_fill.partial)

    # By coin/interval
    by_market: dict[str, list] = {}
    for t in closed:
        key = f"{t.coin}-{t.interval_min}m"
        by_market.setdefault(key, []).append(t.pnl)

    market_pnl = {k: {"n": len(v), "pnl": sum(v), "win_rate": sum(1 for x in v if x > 0)/len(v)*100}
                  for k, v in by_market.items()}

    return {
        # Core
        "n_trades":          len(closed),
        "total_pnl":         float(pnls.sum()),
        "total_pnl_pct":     float(pnls.sum() / initial_capital * 100),
        "total_fees":        float(fees.sum()),
        "net_pnl_after_fees": float(pnls.sum()),  # fees already included in pnl

        # Per trade
        "avg_pnl":           float(pnls.mean()),
        "avg_pnl_bps":       float(pnls_bps.mean()),
        "median_pnl_bps":    float(np.median(pnls_bps)),
        "std_pnl_bps":       float(pnls_bps.std()),

        # Win/loss
        "win_rate":          float(wins.mean() * 100),
        "n_wins":            int(wins.sum()),
        "n_losses":          int(losses.sum()),
        "avg_win":           float(pnls[wins].mean()) if wins.any() else 0,
        "avg_loss":          float(pnls[losses].mean()) if losses.any() else 0,
        "best_trade":        float(pnls.max()),
        "worst_trade":       float(pnls.min()),

        # Risk
        "sharpe":            float(sharpe),
        "max_drawdown":      float(max_drawdown),
        "max_drawdown_pct":  float(max_drawdown / initial_capital * 100),
        "profit_factor":     float(profit_factor),

        # Execution quality
        "avg_fill_rate":     float(np.mean(fill_rates) * 100),
        "n_partial_fills":   partial_fills,
        "pct_partial_fills": float(partial_fills / len(closed) * 100),

        # By market
        "by_market":         market_pnl,
    }


def print_report(metrics: dict) -> None:
    """Affiche un rapport formaté."""
    if "error" in metrics:
        print(f"No results: {metrics['error']}")
        return

    print("\n" + "="*55)
    print("BACKTEST RESULTS")
    print("="*55)
    print(f"  Trades:          {metrics['n_trades']}")
    print(f"  Total PnL:       {metrics['total_pnl']:+.2f} USDC  ({metrics['total_pnl_pct']:+.2f}%)")
    print(f"  Total fees:      {metrics['total_fees']:.2f} USDC")
    print()
    print(f"  Win rate:        {metrics['win_rate']:.1f}%  ({metrics['n_wins']}W / {metrics['n_losses']}L)")
    print(f"  Avg PnL:         {metrics['avg_pnl_bps']:+.1f} bps/trade")
    print(f"  Best trade:      {metrics['best_trade']:+.2f} USDC")
    print(f"  Worst trade:     {metrics['worst_trade']:+.2f} USDC")
    print()
    print(f"  Sharpe:          {metrics['sharpe']:.2f}")
    print(f"  Max drawdown:    {metrics['max_drawdown']:+.2f} USDC  ({metrics['max_drawdown_pct']:+.2f}%)")
    print(f"  Profit factor:   {metrics['profit_factor']:.2f}")
    print()
    print(f"  Fill rate:       {metrics['avg_fill_rate']:.1f}%")
    print(f"  Partial fills:   {metrics['pct_partial_fills']:.1f}%")
    print()
    print("  By market:")
    for market, stats in sorted(metrics["by_market"].items()):
        print(f"    {market:<15} {stats['n']:3d} trades  "
              f"PnL={stats['pnl']:+.2f}  WR={stats['win_rate']:.0f}%")
    print("="*55)


def equity_curve(trades: List[Trade], initial_capital: float = 10_000.0) -> pd.Series:
    """Retourne la courbe d'équité."""
    closed = sorted([t for t in trades if t.is_closed()], key=lambda t: t.exit_ts)
    if not closed:
        return pd.Series(dtype=float)
    pnls = [t.pnl for t in closed]
    times = [t.exit_ts for t in closed]
    equity = initial_capital + pd.Series(pnls, index=times).cumsum()
    return equity
