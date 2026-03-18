# CLAUDE.md — polymarket-backtest

Contexte et règles critiques pour toute session de développement sur ce projet.

## Architecture

```
polymarket-backtest/
├── data.py          # chargement parquets + get_book() lazy + iter_windows()
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

### CRITIQUE — Lazy loading des JSON books

**Ne jamais charger et désérialiser les JSON books en bulk au démarrage.**
120k `json.loads()` sur 30k lignes × 4 colonnes = process bloqué pendant des minutes,
OOM possible, VM inaccessible.

```python
# CORRECT — load() charge les strings brutes, get_book() désérialise à la demande
df = load(live_dir="~/pm_watcher/historical_data/live", coin="btc", interval=5)
# df["up_book_asks"] est encore une string ici

# Dans l'engine, au moment du fill :
book = get_book(tick, "up_book_asks")  # désérialise seulement ce tick
```

```python
# INTERDIT — ne jamais faire ça
df["up_book_asks"] = df["up_book_asks"].apply(json.loads)  # bloque tout
```

### API data.py

```python
# Chargement
df = load(live_dir, coin=None, interval=None, start=None, end=None, max_days=30)

# Lazy book access (dans l'engine/execution)
book = get_book(tick_dict, "up_book_asks")   # → [[price, size], ...]
book = get_book(tick_dict, "up_book_bids")
book = get_book(tick_dict, "down_book_asks")
book = get_book(tick_dict, "down_book_bids")

# Itération par fenêtre (utilisé par engine.py)
for window in iter_windows(df):
    # window = DataFrame de tous les ticks d'un slug
```

## Frais réels Polymarket — CRITIQUE

Source : docs.polymarket.com/trading/fees

**Les frais s'appliquent aux marchés 5min ET 15min crypto.**

```
Formule taker : fee = size_usdc × 0.25 × (p × (1-p))^2
  → max 1.5625% à p=0.50
  → ~0% aux extrêmes (p→0 ou p→1)

Maker : rebate = 20% des frais taker (redistribué quotidiennement en USDC)
  → coût net maker = 0 dans le backtest (conservatif)
```

```python
# Utiliser toujours cette fonction — jamais de constante fixe
from execution import poly_taker_fee
fee = poly_taker_fee(size_usdc=100, avg_price=0.50)  # → 1.5625 USDC
```

**Tableau de référence :**

| p    | Taux effectif | Fee sur 100 USDC |
|------|---------------|-----------------|
| 0.10 | 0.20%         | 0.20 USDC        |
| 0.30 | 1.10%         | 1.10 USDC        |
| 0.50 | **1.56%**     | **1.56 USDC**    |
| 0.70 | 1.10%         | 1.10 USDC        |
| 0.90 | 0.20%         | 0.20 USDC        |

**Implication stratégique** : un aller-retour en market order à p=0.50 coûte ~3.12%.
Il faut un edge > 3.12% pour être profitable → les limit orders (fee=0) sont essentiels.

## Execution — règles métier Polymarket

```
Token UP + DOWN = 1 USDC à expiration
Prix token      = probabilité implicite ∈ (0, 1)
Taker fee       = poly_taker_fee(size, avg_price) — variable
Maker fee       = 0 USDC (rebate positif de 20% du taker)
outcome = "1"   → UP gagne (token UP expire à 1.0)
outcome = "0"   → DOWN gagne (token DOWN expire à 1.0)
```

## Simulate functions

```python
# Market orders (taker, fee variable)
simulate_buy(tick, size_usdc)           # achète UP
simulate_sell(tick, size_usdc)          # vend UP
simulate_buy_down(tick, size_usdc)      # achète DOWN
simulate_sell_down(tick, size_usdc)     # vend DOWN

# Limit orders (maker, fee=0)
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
        if condition_entree:
            return Signal(direction=Direction.UP, size_usdc=100.0, reason="ma_raison")
        if condition_limit:
            return LimitSignal(direction=Direction.UP, size_usdc=100.0,
                              limit_price=0.45, queue_model="conservative")
        if has_position and condition_sortie:
            return ExitSignal(slug=slug, reason="take_profit")
        return None
```

## Tests — toujours lancer avant de modifier execution.py ou data.py

```bash
python3 test_execution.py    # 14 tests unitaires
python3 validate_engine.py   # Oracle 100%, Random ~50%
```

## Lancer le backtest sur la VM

```bash
ssh -i C:\Users\Raphael\PythonProjects\.ssh\mykeys.pem ubuntu@54.195.241.212
cd ~/polymarket-backtest

# Standard
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live

# Filtres
python3 run_backtest.py --dir ~/pm_watcher/historical_data/live \
    --coin btc --interval 5 --start 2026-03-18 --latency 250
```

## Findings importants

- **Lead signal Binance → Polymarket ~600ms** — les limit orders permettent
  d'exploiter ce lead sans payer de frais taker.
- **book JSON lazy** : get_book() désérialise un seul tick à la fois.
  Ne JAMAIS bulk-désérialiser au chargement (VM OOM + inaccessible).
- **iter_windows()** : regroupe les ticks par slug pour le replay.
  Chaque fenêtre = un slug = une instance de marché (coin + interval + window_start).
- **Chunks parquet** : data.py charge finaux + chunks automatiquement.
  Pour chaque date, préfère le final s'il existe (évite les doublons).
- **VM freeze** : si `python3 run_backtest.py` bloque la VM (OOM),
  rebooter depuis AWS Console. Le logger redémarre automatiquement (systemd).

## Repo lié

- **polymarket-logger** : source des données (`~/pm_watcher/historical_data/live/`)
- **polymarket-strategies** (à venir) : stratégies live
