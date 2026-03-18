"""
validate_engine.py — Valide que le moteur de backtest est correct.

3 tests :
  1. Oracle      → win rate ~100%, PnL positif
  2. Random      → win rate ~50%, pas de lookahead bias
  3. AlwaysWrong → win rate ~0%, PnL très négatif

Usage:
    python validate_engine.py --dir historical_data/live
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "strategies"))

import data as D
import engine as E
from strategies.validation import OracleStrategy, RandomStrategy, AlwaysWrongStrategy

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def check(condition: bool, msg: str) -> bool:
    print(f"  {PASS if condition else FAIL} {msg}")
    return condition


def preload_outcomes(df, strat) -> None:
    """Pré-charge les outcomes dans la stratégie avant le run."""
    for slug, group in df.groupby("slug"):
        outcome = group["outcome"].replace("", None).dropna()
        if len(outcome) > 0:
            strat.set_outcome(slug, str(outcome.iloc[-1]))


def run_validation(df, strategy, capital=10_000.0, latency_ms=0.0):
    return E.run(df=df, strategy=strategy, initial_capital=capital,
                 latency_ms=latency_ms, market_impact=False,
                 verbose=False, max_positions=8, allow_multi=True)


def test_oracle(df, capital):
    print("\n── TEST 1: ORACLE (doit gagner ~100%) ──────────────────────")
    strat = OracleStrategy(size_usdc=100.0)
    preload_outcomes(df, strat)
    result = run_validation(df, strat, capital)
    m = result["metrics"]

    if "error" in m:
        print(f"  {FAIL} No trades — need more labeled windows")
        return False

    n, wr, pnl = m["n_trades"], m["win_rate"], m["total_pnl"]
    avg_price = sum(t.entry_fill.avg_price for t in result["trades"]) / n
    print(f"  Trades: {n}  |  Win rate: {wr:.1f}%  |  PnL: {pnl:+.2f} USDC  |  Avg entry: {avg_price:.3f}")

    ok = True
    ok &= check(n >= 5,   f"At least 5 trades ({n})")
    ok &= check(wr >= 90, f"Win rate >= 90% ({wr:.1f}%)")
    ok &= check(pnl > 0,  f"PnL positive ({pnl:+.2f})")
    return ok


def test_random(df, capital):
    print("\n── TEST 2: RANDOM (win rate ~50%, pas de lookahead bias) ───")
    strat = RandomStrategy(size_usdc=100.0, entry_prob=0.5, seed=42)
    result = run_validation(df, strat, capital)
    m = result["metrics"]

    if "error" in m:
        print(f"  {WARN} No trades")
        return True

    n, wr, pnl = m["n_trades"], m["win_rate"], m["total_pnl"]
    print(f"  Trades: {n}  |  Win rate: {wr:.1f}%  |  PnL: {pnl:+.2f} USDC")
    print(f"  {WARN} Note: PnL peut être positif en période de fort trend — ce n'est pas un bug")
    print(f"         Ce qui compte : win rate proche de 50% (pas de lookahead bias)")

    ok = True
    if n < 20:
        print(f"  {WARN} Seulement {n} trades — besoin de 20+ pour être fiable")
        ok &= check(n >= 5, f"At least 5 trades")
    else:
        ok &= check(30 <= wr <= 70, f"Win rate 30-70% ({wr:.1f}%) — no lookahead bias")
    return ok


def test_always_wrong(df, capital):
    print("\n── TEST 3: ALWAYS WRONG (doit perdre ~100%) ────────────────")
    strat = AlwaysWrongStrategy(size_usdc=100.0)
    preload_outcomes(df, strat)
    result = run_validation(df, strat, capital)
    m = result["metrics"]

    if "error" in m:
        print(f"  {WARN} No trades (need windows where DOWN wins)")
        return True

    n, wr, pnl = m["n_trades"], m["win_rate"], m["total_pnl"]
    print(f"  Trades: {n}  |  Win rate: {wr:.1f}%  |  PnL: {pnl:+.2f} USDC")

    ok = True
    ok &= check(n >= 1,   f"At least 1 trade ({n})")
    ok &= check(wr <= 10, f"Win rate <= 10% ({wr:.1f}%)")
    ok &= check(pnl < 0,  f"PnL negative ({pnl:+.2f})")
    return ok


def test_latency(df, capital):
    print("\n── TEST 4: LATENCE (moins de trades ou PnL différent avec latence) ──")
    strat_0   = OracleStrategy(size_usdc=100.0)
    strat_250 = OracleStrategy(size_usdc=100.0)
    preload_outcomes(df, strat_0)
    preload_outcomes(df, strat_250)
    r0   = run_validation(df, strat_0,   capital, latency_ms=0)
    r250 = run_validation(df, strat_250, capital, latency_ms=250)
    m0, m250 = r0["metrics"], r250["metrics"]

    if "error" in m0 or "error" in m250:
        print(f"  {WARN} Not enough data")
        return True

    n0, pnl0   = m0["n_trades"], m0["total_pnl"]
    n250, pnl250 = m250["n_trades"], m250["total_pnl"]
    print(f"  Latency=0ms:   {n0} trades  PnL={pnl0:+.2f}")
    print(f"  Latency=250ms: {n250} trades  PnL={pnl250:+.2f}")

    ok = check(n250 <= n0, f"Trades avec latence <= sans latence ({n250} <= {n0})")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     default="historical_data/live")
    p.add_argument("--capital", type=float, default=10_000.0)
    args = p.parse_args()

    print("=" * 55)
    print("BACKTEST ENGINE VALIDATION")
    print("=" * 55)

    df = D.load(args.dir)
    s  = D.window_summary(df)
    print(f"Data: {s['total_rows']:,} rows  |  "
          f"{s['labeled_windows']}/{s['total_windows']} labeled windows")

    if s["labeled_windows"] < 5:
        print(f"\n{FAIL} Need at least 5 labeled windows.")
        sys.exit(1)

    results = {
        "oracle":       test_oracle(df, args.capital),
        "random":       test_random(df, args.capital),
        "always_wrong": test_always_wrong(df, args.capital),
        "latency":      test_latency(df, args.capital),
    }

    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")

    if all(results.values()):
        print(f"\n{PASS} Engine validated — ready for strategy backtesting")
    else:
        print(f"\n{FAIL} Engine has issues — review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
