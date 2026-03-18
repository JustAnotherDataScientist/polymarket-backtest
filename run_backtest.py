"""
run_backtest.py v2 — CLI avec latence configurable.

Usage:
    # Sans latence (optimiste)
    python run_backtest.py --strategy edge_zscore --latency 0

    # Avec latence réaliste (AWS Dublin → Polymarket London)
    python run_backtest.py --strategy edge_zscore --latency 50

    # Latence locale (VM distante)
    python run_backtest.py --strategy edge_zscore --latency 250

    # Comparer différentes latences
    python run_backtest.py --strategy edge_zscore --latency-sweep

    # Comparer toutes les stratégies
    python run_backtest.py --compare --latency 50
"""
from __future__ import annotations

import argparse, sys, logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "strategies"))

import data as D
import engine as E
from metrics import print_report
from strategies.rule_based import EdgeZScoreStrategy, FlowMomentumStrategy
from strategies.ml_based   import MLStrategy, train_model

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")


def build_strategy(args):
    if args.strategy == "edge_zscore":
        return EdgeZScoreStrategy(
            min_edge_bps=args.min_edge, min_z_score=args.min_z,
            max_distance_bps=args.max_dist, min_seconds=args.min_sec,
            size_usdc=args.size, stop_loss_mid=args.stop_loss,
        )
    elif args.strategy == "flow_momentum":
        return FlowMomentumStrategy(size_usdc=args.size)
    elif args.strategy == "ml":
        print("Training ML model...")
        df_train = D.load(args.dir, args.coin, args.interval,
                          args.start_date, args.end_date)
        model = train_model(df_train, model_type=args.model_type)
        return MLStrategy(model=model, threshold=args.threshold,
                          min_edge_bps=args.min_edge, min_seconds=args.min_sec,
                          size_usdc=args.size)
    raise ValueError(f"Unknown strategy: {args.strategy}")


def load_data(args):
    print(f"Loading data from {args.dir}...")
    df = D.load(args.dir, args.coin, args.interval,
                args.start_date, args.end_date)
    s = D.window_summary(df)
    print(f"  {s['total_rows']:,} rows  |  "
          f"{s['labeled_windows']}/{s['total_windows']} labeled windows  |  "
          f"{s['date_range'][0].date()} → {s['date_range'][1].date()}")
    if s["labeled_windows"] == 0:
        print("\nERROR: No labeled windows — run backfill_outcomes.py first")
        sys.exit(1)
    return df


def run_latency_sweep(args):
    """Compare les performances selon différentes latences."""
    df       = load_data(args)
    latencies = [0, 50, 100, 250, 500, 1000]   # ms

    print(f"\nLatency sweep — strategy: {args.strategy}")
    print(f"{'Latency':>10} {'Trades':>8} {'WinRate':>8} "
          f"{'PnL':>10} {'Sharpe':>8} {'MaxDD':>10} {'AvgSlip':>10}")
    print("-" * 68)

    for lat in latencies:
        strat  = build_strategy(args)
        result = E.run(df=df, strategy=strat,
                       initial_capital=args.capital,
                       latency_ms=lat,
                       market_impact=args.market_impact,
                       max_positions=args.max_pos)
        m = result["metrics"]
        if "error" in m:
            print(f"{lat:>8}ms  {'no trades':>8}")
            continue

        # Avg slippage from trades
        avg_slip = 0.0
        fills = [t.entry_fill for t in result["trades"] if t.entry_fill]
        if fills:
            avg_slip = sum(f.slippage_bps for f in fills) / len(fills)

        print(f"{lat:>8}ms"
              f"  {m['n_trades']:>8}"
              f"  {m['win_rate']:>7.1f}%"
              f"  {m['total_pnl']:>+9.2f}"
              f"  {m['sharpe']:>8.2f}"
              f"  {m['max_drawdown']:>+9.2f}"
              f"  {avg_slip:>+9.2f}bp")


def run_compare(args):
    """Compare toutes les stratégies."""
    df = load_data(args)
    strategies = [
        ("EdgeZScore_300",  EdgeZScoreStrategy(min_edge_bps=300, min_z_score=1.5, size_usdc=args.size)),
        ("EdgeZScore_500",  EdgeZScoreStrategy(min_edge_bps=500, min_z_score=2.0, size_usdc=args.size)),
        ("FlowMomentum",    FlowMomentumStrategy(size_usdc=args.size)),
    ]
    lat = args.latency

    print(f"\nStrategy comparison  (latency={lat}ms  market_impact={args.market_impact})")
    print(f"{'Strategy':<20} {'Trades':>8} {'WinRate':>8} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print("-" * 66)

    for name, strat in strategies:
        result = E.run(df=df, strategy=strat, initial_capital=args.capital,
                       latency_ms=lat, market_impact=args.market_impact,
                       max_positions=args.max_pos)
        m = result["metrics"]
        if "error" in m:
            print(f"{name:<20}  no trades")
            continue
        print(f"{name:<20}"
              f"  {m['n_trades']:>8}"
              f"  {m['win_rate']:>7.1f}%"
              f"  {m['total_pnl']:>+9.2f}"
              f"  {m['sharpe']:>8.2f}"
              f"  {m['max_drawdown']:>+9.2f}")


def run_single(args):
    df    = load_data(args)
    strat = build_strategy(args)

    print(f"\nRunning: {strat.name}"
          f"  latency={args.latency}ms"
          f"  market_impact={args.market_impact}"
          f"  capital={args.capital:.0f} USDC")

    result = E.run(
        df=df, strategy=strat,
        initial_capital  = args.capital,
        latency_ms       = args.latency,
        market_impact    = args.market_impact,
        verbose          = args.verbose,
        max_positions    = args.max_pos,
    )

    print_report(result["metrics"])
    print(f"\n  Windows: {result['n_windows']}  |  Skipped: {result['n_skipped']}")

    eq = result["equity_curve"]
    if len(eq) > 0:
        out = Path(f"results_{strat.name}_lat{args.latency:.0f}ms.csv")
        eq.to_csv(out, header=["equity"])
        print(f"  Equity curve → {out}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",   default="edge_zscore",
                   choices=["edge_zscore","flow_momentum","ml"])
    p.add_argument("--compare",        action="store_true")
    p.add_argument("--latency-sweep",  action="store_true",
                   help="Compare latency 0/50/100/250/500/1000ms")
    p.add_argument("--dir",        default="historical_data/live")
    p.add_argument("--coin",       default=None)
    p.add_argument("--interval",   type=int, default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    p.add_argument("--capital",    type=float, default=10_000.0)
    p.add_argument("--max-pos",    type=int,   default=4)
    p.add_argument("--verbose",    action="store_true")
    # Execution
    p.add_argument("--latency",        type=float, default=0.0,
                   help="Latency in ms (0=instantaneous, 50=AWS Dublin, 250=1 tick)")
    p.add_argument("--market-impact",  action="store_true", default=True)
    p.add_argument("--no-market-impact", dest="market_impact", action="store_false")
    # Strategy params
    p.add_argument("--min-edge",   type=float, default=300.0)
    p.add_argument("--min-z",      type=float, default=1.5)
    p.add_argument("--max-dist",   type=float, default=100.0)
    p.add_argument("--min-sec",    type=float, default=30.0)
    p.add_argument("--size",       type=float, default=100.0)
    p.add_argument("--stop-loss",  type=float, default=None)
    # ML
    p.add_argument("--model-type", default="xgboost",
                   choices=["xgboost","random_forest","logistic"])
    p.add_argument("--threshold",  type=float, default=0.65)

    args = p.parse_args()

    if args.latency_sweep:
        run_latency_sweep(args)
    elif args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
