# CLAUDE.md — polymarket-backtest

Contexte et règles critiques pour toute session de développement sur ce projet.

## Architecture

```
polymarket-backtest/
├── data.py          # chargement parquets → DataFrame (désérialise JSON books)
├── engine.py        # moteur event-driven tick-par-tick
├── execution.py     # simulation fills (market + limit orders, UP + DOWN)
├── portfolio.py     # gestion positions, cash, PnL
├── metrics.py       # calcul métriques (Sharpe, drawdown, win rate...)
├── run_backtest.py  # entry point CLI
├── test_execution.py # 14 tests unitaires déterministes
├── validate_engine.py # validation Oracle (100%) + Random (~50%)
└── strategies/
    ├── base.py       # Signal, LimitSignal, ExitSignal, Direction
    ├── rule_based.py # stratégies à base de règles
    ├── ml_based.py   # stratégies ML
    └── validation.py # Oracle + Random (référence)
```

## Données — où elles sont

Les données sont sur la VM AWS, pas en local. **Ne jamais télécharger les parquets**
sur le poste local — ça explose la mémoire. Toujours travailler via SSH :

```bash
ssh -i ~/.ssh/mykeys.pem ubuntu@54.195.241.212
cd ~/polymarket-backtest
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live
```

## Lecture des données — data.py

```python
# Charge parquets finaux (YYYY-MM-DD.parquet) + chunks en cours (*_NNNN.parquet)
df = load(live_dir="~/pm_watcher/historical_data/live", coin="btc", interval=5)
```

**Désérialisation JSON books** — automatique dans `load()` :
```python
df["up_book_asks"]   # → liste Python [[price, size], ...]  après load()
df["down_book_bids"] # → liste Python [[price, size], ...]
```
**Ne jamais appeler `json.loads()` manuellement** — déjà fait dans `load()`.

## Execution — règles métier Polymarket

```
Token UP + DOWN = 1 USDC à expiration
Prix token      = probabilité implicite ∈ (0, 1)
Taker fee       = 2 bps  (simulate_buy / simulate_sell)
Maker fee       = 0 bps  (simulate_limit_buy / simulate_limit_buy_down)
outcome = "1"   → UP gagne (token UP expire à 1.0)
outcome = "0"   → DOWN gagne (token DOWN expire à 1.0)
```

## Simulate functions

```python
# Market orders (taker, 2bps)
simulate_buy(tick, size_usdc)           # achète UP
simulate_sell(tick, size_usdc)          # vend UP
simulate_buy_down(tick, size_usdc)      # achète DOWN
simulate_sell_down(tick, size_usdc)     # vend DOWN

# Limit orders (maker, 0bps)
simulate_limit_buy(tick_sequence, limit_price, size_usdc,
                   queue_model="conservative")  # UP
simulate_limit_buy_down(tick_sequence, limit_price, size_usdc,
                        queue_model="conservative")  # DOWN

# PnL
compute_pnl(entry_fill, exit_fill, outcome, direction="UP")
```

**queue_model** :
- `"conservative"` : fill si `best_ask < limit_price` (on est en queue)
- `"optimistic"` : fill si `best_ask <= limit_price` (on est en tête)

## Stratégies — comment en créer une

```python
from strategies.base import Strategy, Signal, LimitSignal, ExitSignal, Direction

class MaStrategie(Strategy):
    def on_tick(self, slug, tick, has_position):
        # Signal market order UP
        if condition_entree:
            return Signal(direction=Direction.UP, size_usdc=100.0,
                         reason="ma_raison")
        # Signal limit order UP (0 frais)
        if condition_limit:
            return LimitSignal(direction=Direction.UP, size_usdc=100.0,
                              limit_price=0.45, queue_model="conservative")
        # Exit anticipé
        if has_position and condition_sortie:
            return ExitSignal(slug=slug, reason="take_profit")
        return None
```

## Engine — comportement important

- **Latence configurable** : `run(df, strategy, latency_ms=250)` — 1 tick = 250ms
- **Market orders** : fill sur le tick actuel après `latency_ms`
- **Limit orders** : check tick par tick depuis `fill_after_ts`, annulé en fin de fenêtre
- **Direction DOWN** : `portfolio.close()` calcule le PnL correctement
  (outcome="0" → DOWN gagne → token vaut 1.0)
- **Pas d'exit latency** : les ExitSignal sont exécutés immédiatement

## Tests — toujours les lancer avant de modifier execution.py

```bash
python3 test_execution.py    # 14 tests unitaires (market + limit + DOWN + PnL)
python3 validate_engine.py   # Oracle 100%, Random ~50%
```

## Findings importants

- **Lead signal Binance → Polymarket ~600ms** : les limit orders sont essentiels
  pour exploiter ce lead sans payer 2bps à chaque trade.
- **Taker fees tuent les stratégies** : à 2bps par leg (entrée + sortie),
  il faut un edge > 4bps net pour être profitable en market orders.
- **Maker = 0 frais** : les limit orders sont la clé de la profitabilité.
- **Book Polymarket** : les colonnes `up_book_asks/bids` sont des listes Python
  après `load()`. Le backtest peut rejouer fidèlement le book niveau par niveau.
- **Chunks parquet** : `data.py` charge automatiquement les fichiers
  `*_NNNN.parquet` (chunks intra-journaliers) ET les `YYYY-MM-DD.parquet`
  (fichiers finaux fusionnés). Pas besoin de préciser.

## Lancer le backtest sur la VM

```bash
ssh -i C:\Users\Raphael\PythonProjects\.ssh\mykeys.pem ubuntu@54.195.241.212
cd ~/polymarket-backtest

# Lancer sur toutes les données disponibles
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live

# Filtres disponibles
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live \
    --coin btc --interval 5 --start 2026-03-18 --latency 250
```

## Repo lié

- **polymarket-logger** : source des données (`~/pm_watcher/historical_data/live/`)
- **polymarket-strategies** (à venir) : stratégies live basées sur les signaux du logger
