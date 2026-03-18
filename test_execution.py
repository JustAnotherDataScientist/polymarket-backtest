"""
test_execution.py — Tests unitaires du moteur d'exécution.

Tests existants (8) — inchangés :
  1. buy_simple, 2. buy_walk_book, 3. buy_partial,
  4. pnl_hold_win, 5. pnl_hold_lose,
  6. pnl_early_exit_win, 7. pnl_early_exit_lose, 8. exit_vs_hold

Nouveaux tests (6) :
   9. limit_buy_fill_conservative  — fill quand best_ask < limit_price
  10. limit_buy_no_fill            — pas de fill si prix ne descend jamais
  11. limit_buy_fill_optimistic    — fill dès que best_ask <= limit_price
  12. limit_buy_down_fill          — fill sur token DOWN
  13. limit_fee_zero               — frais maker = 0
  14. pnl_down_win / pnl_down_lose — compute_pnl avec direction DOWN

Usage:
    python test_execution.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "strategies"))

from execution import (
    simulate_buy, simulate_sell, simulate_buy_down,
    simulate_limit_buy, simulate_limit_buy_down,
    compute_pnl, poly_taker_fee, MAKER_FEE_BPS,
)

PASS = "✅"
FAIL = "❌"
TOL  = 0.001


def check(condition: bool, msg: str, got=None, expected=None) -> bool:
    detail = f"  got={got:.4f}  expected={expected:.4f}" if got is not None else ""
    print(f"  {PASS if condition else FAIL} {msg}{detail}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Helpers tick synthétiques
# ─────────────────────────────────────────────────────────────────────────────

def make_tick(ask=0.40, ask_size=1000, bid=0.38, bid_size=1000,
              ask2=0.42, ask2_size=500, bid2=0.36, bid2_size=500,
              ask3=0.44, ask3_size=300, bid3=0.34, bid3_size=300) -> dict:
    """Tick synthétique format ANCIEN (colonnes L2-L5) — pour tester le fallback."""
    return {
        "up_ask":           ask,
        "up_best_ask_size": ask_size,
        "up_bid":           bid,
        "up_best_bid_size": bid_size,
        "ask_l2_price": ask2, "ask_l2_size": ask2_size,
        "ask_l3_price": ask3, "ask_l3_size": ask3_size,
        "ask_l4_price": None, "ask_l4_size": None,
        "ask_l5_price": None, "ask_l5_size": None,
        "bid_l2_price": bid2, "bid_l2_size": bid2_size,
        "bid_l3_price": bid3, "bid_l3_size": bid3_size,
        "bid_l4_price": None, "bid_l4_size": None,
        "bid_l5_price": None, "bid_l5_size": None,
        "up_ask_depth_top5": sum(x for x in [ask_size, ask2_size, ask3_size] if x),
        "up_bid_depth_top5": sum(x for x in [bid_size, bid2_size, bid3_size] if x),
    }


def make_tick_json(ask=0.40, ask_size=1000, bid=0.38, bid_size=1000,
                   extra_asks=None, extra_bids=None,
                   down_ask=None, down_ask_size=None) -> dict:
    """
    Tick synthétique format NOUVEAU (colonnes JSON book complet).
    extra_asks / extra_bids : liste de [price, size] additionnels
    """
    asks = [[ask, ask_size]] + (extra_asks or [])
    bids = [[bid, bid_size]] + (extra_bids or [])
    t = {
        "up_ask":            ask,
        "up_best_ask_size":  ask_size,
        "up_bid":            bid,
        "up_best_bid_size":  bid_size,
        "up_book_asks":      asks,
        "up_book_bids":      bids,
        "up_ask_depth_top5": sum(s for _, s in asks[:5]),
        "up_bid_depth_top5": sum(s for _, s in bids[:5]),
    }
    if down_ask is not None:
        t["down_ask"] = down_ask
        t["down_best_ask_size"] = down_ask_size or 0
        t["down_book_asks"] = [[down_ask, down_ask_size or 0]]
        t["down_book_bids"] = []
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Tests market orders (inchangés, fonctionnent avec fallback L2-L5)
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_simple():
    print("\n── TEST 1 : Achat simple (100 USDC @ ask=0.40) ────────────")
    tick = make_tick(ask=0.40, ask_size=1000)
    fill = simulate_buy(tick, 100.0)

    tokens_expected   = 100.0 / 0.40
    fee_expected      = poly_taker_fee(100.0, 0.40)   # @ ask=0.40
    net_cost_expected = 100.0 + fee_expected

    ok = True
    ok &= check(fill is not None, "Fill exists")
    if fill is None: return False
    ok &= check(abs(fill.filled_size - 100.0) < TOL,
                "filled_size = 100 USDC", fill.filled_size, 100.0)
    ok &= check(abs(fill.avg_price - 0.40) < TOL,
                "avg_price = 0.40", fill.avg_price, 0.40)
    ok &= check(abs(fill.fee - fee_expected) < TOL,
                f"fee = {fee_expected:.4f}", fill.fee, fee_expected)
    ok &= check(abs(fill.net_cost - net_cost_expected) < TOL,
                f"net_cost = {net_cost_expected:.4f}", fill.net_cost, net_cost_expected)
    ok &= check(not fill.partial, "No partial fill")
    ok &= check(abs(fill.slippage_bps) < TOL, "No slippage on L1")
    return ok


def test_buy_walk_book():
    print("\n── TEST 2 : Achat walk book (200 USDC, L1=100, L2=500) ────")
    tick = make_tick(ask=0.40, ask_size=100, ask2=0.42, ask2_size=500)
    fill = simulate_buy(tick, 200.0)

    tokens_l1    = 100.0 / 0.40
    tokens_l2    = 100.0 / 0.42
    total_tokens = tokens_l1 + tokens_l2
    avg_expected = 200.0 / total_tokens
    fee_expected = poly_taker_fee(200.0, avg_expected)

    ok = True
    ok &= check(fill is not None, "Fill exists")
    if fill is None: return False
    ok &= check(abs(fill.filled_size - 200.0) < TOL,
                "filled_size = 200 USDC", fill.filled_size, 200.0)
    ok &= check(abs(fill.avg_price - avg_expected) < TOL,
                f"avg_price = {avg_expected:.4f}", fill.avg_price, avg_expected)
    ok &= check(abs(fill.fee - fee_expected) < TOL,
                f"fee = {fee_expected:.4f}", fill.fee, fee_expected)
    ok &= check(not fill.partial, "No partial fill (200 <= 600 available)")
    ok &= check(fill.slippage_bps > 0, "Slippage > 0 (walked to L2)")
    return ok


def test_buy_partial():
    print("\n── TEST 3 : Achat partiel (500 USDC, seulement 300 dispo) ─")
    tick = make_tick(ask=0.40, ask_size=200, ask2=0.42, ask2_size=100,
                     ask3=None, ask3_size=None)
    tick["ask_l3_price"] = None
    tick["ask_l3_size"]  = None
    fill = simulate_buy(tick, 500.0)

    ok = True
    ok &= check(fill is not None, "Fill exists (partial)")
    if fill is None: return False
    ok &= check(fill.partial, "Fill is partial")
    ok &= check(fill.filled_size < 500.0,
                f"filled < 500 ({fill.filled_size:.2f})", fill.filled_size, 300.0)
    ok &= check(abs(fill.fill_rate - 300.0/500.0) < 0.01,
                f"fill_rate = {300/500:.2f}", fill.fill_rate, 300.0/500.0)
    return ok


def test_pnl_hold_win():
    print("\n── TEST 4 : PnL hold to close, UP gagne ───────────────────")
    tick  = make_tick(ask=0.40, ask_size=1000)
    entry = simulate_buy(tick, 100.0)
    tokens       = 100.0 / 0.40
    pnl_expected = tokens * 1.0 - entry.net_cost
    pnl = compute_pnl(entry, None, "1")
    ok = True
    ok &= check(abs(pnl - pnl_expected) < TOL,
                f"PnL hold win = {pnl_expected:.4f}", pnl, pnl_expected)
    ok &= check(pnl > 0, f"PnL positive ({pnl:.4f})")
    return ok


def test_pnl_hold_lose():
    print("\n── TEST 5 : PnL hold to close, UP perd ────────────────────")
    tick  = make_tick(ask=0.40, ask_size=1000)
    entry = simulate_buy(tick, 100.0)
    pnl_expected = -entry.net_cost
    pnl = compute_pnl(entry, None, "0")
    ok = True
    ok &= check(abs(pnl - pnl_expected) < TOL,
                f"PnL hold lose = {pnl_expected:.4f}", pnl, pnl_expected)
    ok &= check(pnl < 0, f"PnL negative ({pnl:.4f})")
    return ok


def test_pnl_early_exit_win():
    print("\n── TEST 6 : Exit anticipé gagnant (buy @0.40, sell @0.50) ─")
    tick_entry = make_tick(ask=0.40, ask_size=1000)
    tick_exit  = make_tick(bid=0.50, bid_size=1000)
    entry  = simulate_buy(tick_entry, 100.0)
    tokens = entry.filled_size / entry.avg_price
    exit_f = simulate_sell(tick_exit, tokens * 0.50)

    usdc_received = tokens * 0.50
    fee_exit      = poly_taker_fee(usdc_received, 0.50)  # sell @ bid=0.50
    pnl_expected  = (usdc_received - fee_exit) - entry.net_cost
    pnl = compute_pnl(entry, exit_f, "1")

    ok = True
    ok &= check(exit_f is not None, "Exit fill exists")
    ok &= check(abs(exit_f.filled_size - usdc_received) < TOL,
                f"exit filled_size = {usdc_received:.4f}", exit_f.filled_size, usdc_received)
    ok &= check(abs(pnl - pnl_expected) < TOL,
                f"PnL early exit win = {pnl_expected:.4f}", pnl, pnl_expected)
    ok &= check(pnl > 0, f"PnL positive ({pnl:.4f})")
    return ok


def test_pnl_early_exit_lose():
    print("\n── TEST 7 : Exit anticipé perdant (buy @0.40, sell @0.25) ─")
    tick_entry = make_tick(ask=0.40, ask_size=1000)
    tick_exit  = make_tick(bid=0.25, bid_size=1000)
    entry  = simulate_buy(tick_entry, 100.0)
    tokens = entry.filled_size / entry.avg_price
    exit_f = simulate_sell(tick_exit, tokens * 0.25)

    usdc_received = tokens * 0.25
    fee_exit      = poly_taker_fee(usdc_received, 0.25)  # sell @ bid=0.25
    pnl_expected  = (usdc_received - fee_exit) - entry.net_cost
    pnl = compute_pnl(entry, exit_f, "1")

    ok = True
    ok &= check(abs(pnl - pnl_expected) < TOL,
                f"PnL early exit lose = {pnl_expected:.4f}", pnl, pnl_expected)
    ok &= check(pnl < 0, f"PnL negative même si outcome=1 ({pnl:.4f})")
    return ok


def test_exit_vs_hold():
    print("\n── TEST 8 : Exit @0.70 vs hold (outcome=1, entry=0.40) ────")
    tick_e = make_tick(ask=0.40, ask_size=1000)
    tick_x = make_tick(bid=0.70, bid_size=1000)
    entry  = simulate_buy(tick_e, 100.0)
    tokens = entry.filled_size / entry.avg_price
    exit_f = simulate_sell(tick_x, tokens * 0.70)
    pnl_hold = compute_pnl(entry, None,   "1")
    pnl_exit = compute_pnl(entry, exit_f, "1")
    ok = True
    ok &= check(pnl_hold > pnl_exit,
                f"Hold ({pnl_hold:.2f}) > Exit@0.70 ({pnl_exit:.2f}) si outcome=1")
    print(f"    → Gain supplémentaire à tenir : {pnl_hold - pnl_exit:.2f} USDC")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Nouveaux tests — limit orders
# ─────────────────────────────────────────────────────────────────────────────

def test_limit_buy_fill_conservative():
    print("\n── TEST 9 : Limit buy fill conservative ───────────────────")
    # Prix descend : tick1 ask=0.41, tick2 ask=0.39 → fill car 0.39 < 0.40
    limit_price = 0.40
    tick1 = make_tick_json(ask=0.41, ask_size=500)
    tick2 = make_tick_json(ask=0.39, ask_size=500)  # passe sous la limite
    fill = simulate_limit_buy([tick1, tick2], limit_price, 100.0, "conservative")

    ok = True
    ok &= check(fill is not None, "Fill exists (prix est passé sous la limite)")
    if fill is None: return False
    ok &= check(abs(fill.avg_price - limit_price) < TOL,
                f"avg_price = limit_price = {limit_price}", fill.avg_price, limit_price)
    ok &= check(abs(fill.filled_size - 100.0) < TOL,
                "filled_size = 100 USDC", fill.filled_size, 100.0)
    ok &= check(abs(fill.fee) < TOL, "Fee = 0 (maker)", fill.fee, 0.0)
    ok &= check(not fill.partial, "Pas de fill partiel (limit order = tout ou rien)")
    return ok


def test_limit_buy_no_fill():
    print("\n── TEST 10 : Limit buy — pas de fill (prix reste haut) ────")
    limit_price = 0.40
    # Prix reste toujours AU-DESSUS de la limite
    ticks = [make_tick_json(ask=0.41 + i * 0.01, ask_size=500) for i in range(5)]
    fill = simulate_limit_buy(ticks, limit_price, 100.0, "conservative")

    ok = True
    ok &= check(fill is None, "Fill = None (ordre non rempli, annulé)")
    return ok


def test_limit_buy_fill_optimistic():
    print("\n── TEST 11 : Limit buy fill optimistic (ask == limit) ──────")
    limit_price = 0.40
    # En optimistic, fill dès que best_ask <= limit_price
    tick1 = make_tick_json(ask=0.41, ask_size=500)
    tick2 = make_tick_json(ask=0.40, ask_size=500)  # exactement égal
    fill_conservative = simulate_limit_buy([tick1, tick2], limit_price, 100.0, "conservative")
    fill_optimistic   = simulate_limit_buy([tick1, tick2], limit_price, 100.0, "optimistic")

    ok = True
    ok &= check(fill_conservative is None,
                "Conservative : pas de fill si ask == limit (besoin ask < limit)")
    ok &= check(fill_optimistic is not None,
                "Optimistic : fill si ask == limit")
    if fill_optimistic:
        ok &= check(abs(fill_optimistic.fee) < TOL, "Fee = 0 maker", fill_optimistic.fee, 0.0)
    return ok


def test_limit_buy_down_fill():
    print("\n── TEST 12 : Limit buy DOWN fill ───────────────────────────")
    limit_price = 0.60  # on veut acheter DOWN à max 0.60
    # tick1 : down_ask = 0.62 (pas encore)
    # tick2 : down_ask = 0.58 (passe sous 0.60) → fill conservative
    tick1 = {"down_ask": 0.62, "down_best_ask_size": 500,
              "down_book_asks": [[0.62, 500]], "down_book_bids": []}
    tick2 = {"down_ask": 0.58, "down_best_ask_size": 500,
              "down_book_asks": [[0.58, 500]], "down_book_bids": []}

    fill = simulate_limit_buy_down([tick1, tick2], limit_price, 100.0, "conservative")

    ok = True
    ok &= check(fill is not None, "Fill DOWN exists")
    if fill is None: return False
    ok &= check(abs(fill.avg_price - limit_price) < TOL,
                f"avg_price = {limit_price}", fill.avg_price, limit_price)
    ok &= check(abs(fill.fee) < TOL, "Fee = 0 maker DOWN", fill.fee, 0.0)
    return ok


def test_limit_fee_zero():
    print("\n── TEST 13 : Limit order fee = 0, market order fee = 2bps ─")
    tick = make_tick_json(ask=0.40, ask_size=1000)

    # Market order : fee = 2bps
    market_fill = simulate_buy(tick, 100.0)
    # Limit order fill immédiat (optimistic, ask == 0.40 == limit)
    limit_fill = simulate_limit_buy([tick], 0.40, 100.0, "optimistic")

    ok = True
    ok &= check(market_fill is not None, "Market fill exists")
    if market_fill:
        expected_taker_fee = poly_taker_fee(100.0, 0.40)  # @ ask=0.40
        ok &= check(abs(market_fill.fee - expected_taker_fee) < TOL,
                    f"Market fee = {expected_taker_fee:.4f} (poly formula @ p=0.40)",
                    market_fill.fee, expected_taker_fee)

    ok &= check(limit_fill is not None, "Limit fill exists")
    if limit_fill:
        ok &= check(abs(limit_fill.fee) < TOL,
                    f"Limit fee = 0 (maker = {MAKER_FEE_BPS} bps)",
                    limit_fill.fee, 0.0)

    if market_fill and limit_fill:
        net_cost_market = market_fill.net_cost
        net_cost_limit  = limit_fill.net_cost
        ok &= check(net_cost_limit < net_cost_market,
                    f"Limit moins cher : {net_cost_limit:.4f} < {net_cost_market:.4f}")
        fee_pct = (net_cost_market - net_cost_limit) / 100 * 100
        print(f"    → Économie maker vs taker : {net_cost_market - net_cost_limit:.4f} USDC "
              f"sur 100 USDC ({fee_pct:.2f}% effectif @ p=0.40)")
    return ok


def test_pnl_down_direction():
    print("\n── TEST 14 : PnL direction DOWN ────────────────────────────")
    # Achat DOWN @ 0.40
    tick = {"down_ask": 0.40, "down_best_ask_size": 1000,
            "down_book_asks": [[0.40, 1000]], "down_book_bids": []}
    from execution import simulate_buy_down
    entry = simulate_buy_down(tick, 100.0)

    ok = True
    ok &= check(entry is not None, "Entry DOWN fill exists")
    if entry is None: return False

    # outcome="0" → DOWN gagne → token vaut 1.0
    pnl_win  = compute_pnl(entry, None, "0", direction="DOWN")
    # outcome="1" → DOWN perd → token vaut 0.0
    pnl_lose = compute_pnl(entry, None, "1", direction="DOWN")

    tokens = entry.filled_size / entry.avg_price  # 100/0.40 = 250
    pnl_win_expected  = tokens * 1.0 - entry.net_cost
    pnl_lose_expected = 0.0 - entry.net_cost

    ok &= check(abs(pnl_win - pnl_win_expected) < TOL,
                f"PnL DOWN win (outcome=0) = {pnl_win_expected:.4f}",
                pnl_win, pnl_win_expected)
    ok &= check(pnl_win > 0, f"PnL DOWN win positive ({pnl_win:.4f})")
    ok &= check(abs(pnl_lose - pnl_lose_expected) < TOL,
                f"PnL DOWN lose (outcome=1) = {pnl_lose_expected:.4f}",
                pnl_lose, pnl_lose_expected)
    ok &= check(pnl_lose < 0, f"PnL DOWN lose negative ({pnl_lose:.4f})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("EXECUTION ENGINE v3 — UNIT TESTS")
    print("=" * 58)
    print(f"Taker fee: variable (max 1.56% à p=0.50)  |  Maker fee: {MAKER_FEE_BPS} USDC")

    tests = [
        # Market orders (inchangés)
        ("buy_simple",               test_buy_simple),
        ("buy_walk_book",            test_buy_walk_book),
        ("buy_partial",              test_buy_partial),
        ("pnl_hold_win",             test_pnl_hold_win),
        ("pnl_hold_lose",            test_pnl_hold_lose),
        ("pnl_early_exit_win",       test_pnl_early_exit_win),
        ("pnl_early_exit_lose",      test_pnl_early_exit_lose),
        ("exit_vs_hold",             test_exit_vs_hold),
        # Limit orders (nouveaux)
        ("limit_buy_fill_conservative", test_limit_buy_fill_conservative),
        ("limit_buy_no_fill",           test_limit_buy_no_fill),
        ("limit_buy_fill_optimistic",   test_limit_buy_fill_optimistic),
        ("limit_buy_down_fill",         test_limit_buy_down_fill),
        ("limit_fee_zero",              test_limit_fee_zero),
        ("pnl_down_direction",          test_pnl_down_direction),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"  ❌ EXCEPTION: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 58)
    print("SUMMARY")
    print("=" * 58)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    n_pass = sum(1 for ok in results.values() if ok)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print(f"\n✅ All {n_pass} unit tests passed — execution engine v3 correct")
    else:
        print(f"\n❌ {n_fail} test(s) failed  ({n_pass} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
