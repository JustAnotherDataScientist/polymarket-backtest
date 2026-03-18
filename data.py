"""
data.py — Chargement des données pour le backtest.

ARCHITECTURE :
- Les colonnes JSON book (up_book_bids/asks, down_book_bids/asks) sont
  désérialisées LAZILY tick par tick dans l'engine — pas en bulk au chargement.
- Le DataFrame chargé contient les colonnes book en string (comme dans le parquet).
- L'engine appelle get_book(tick, col) pour obtenir la liste Python au moment voulu.

Cela évite de faire 120k json.loads() au démarrage (~5min sur 30k lignes).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import pandas as pd

BOOK_COLS = ("up_book_bids", "up_book_asks", "down_book_bids", "down_book_asks")


def get_book(tick: dict, col: str) -> list:
    """
    Désérialise une colonne book à la demande (lazy).
    Retourne [] si absent ou invalide.
    """
    val = tick.get(col)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip() not in ("", "[]", "None"):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def load(
    live_dir: str = "historical_data/live",
    coin: Optional[str] = None,
    interval: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_days: int = 30,
) -> pd.DataFrame:
    """
    Charge les parquets sans désérialiser les JSON books.
    Les colonnes book restent en string — utilisez get_book(tick, col) dans l'engine.

    Args:
        live_dir  : dossier contenant les parquets
        coin      : filtre sur la colonne 'coin' (ex: 'btc')
        interval  : filtre sur 'interval_min' (ex: 5)
        start     : date de début 'YYYY-MM-DD'
        end       : date de fin 'YYYY-MM-DD'
        max_days  : limite de jours à charger (protection mémoire)
    Returns:
        DataFrame trié par timestamp_utc, sans désérialisation JSON
    """
    p = Path(live_dir)
    if not p.exists():
        raise FileNotFoundError(f"Dossier introuvable : {live_dir}")

    # Lister les fichiers : finaux (YYYY-MM-DD.parquet) + chunks (*_NNNN.parquet)
    finals = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
    chunks = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.parquet"))

    # Filtrer par date si précisé
    if start or end:
        def in_range(f: Path) -> bool:
            date_str = f.name[:10]
            if start and date_str < start:
                return False
            if end and date_str > end:
                return False
            return True
        finals = [f for f in finals if in_range(f)]
        chunks = [f for f in chunks if in_range(f)]

    # Limiter à max_days
    all_dates = sorted(set(f.name[:10] for f in finals + chunks))
    if len(all_dates) > max_days:
        all_dates = all_dates[-max_days:]
        finals = [f for f in finals if f.name[:10] in all_dates]
        chunks = [f for f in chunks if f.name[:10] in all_dates]

    # Pour chaque date, préférer le final s'il existe, sinon les chunks
    final_dates = {f.name[:10] for f in finals}
    files = finals + [c for c in chunks if c.name[:10] not in final_dates]
    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"Aucun parquet trouvé dans {live_dir}")

    # Charger sans désérialiser les books
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ {f.name} illisible : {e}")

    if not dfs:
        raise ValueError("Aucun parquet lisible")

    df = pd.concat(dfs, ignore_index=True)

    # Filtres
    if coin:
        df = df[df["coin"] == coin]
    if interval:
        df = df[df["interval_min"] == str(interval)]

    # Tri + reset
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Colonnes numériques critiques
    num_cols = [
        "up_bid", "up_ask", "up_mid", "up_best_bid_size", "up_best_ask_size",
        "down_bid", "down_ask", "down_mid", "down_best_bid_size", "down_best_ask_size",
        "coin_price", "z_score", "analytical_prob_up", "edge_bps",
        "bn_best_bid", "bn_best_ask", "seconds_remaining",
        "coin_vol_60s_bps", "coin_return_60s_bps", "coin_flow_imbalance_60s",
        "up_depth_imbalance", "down_depth_imbalance",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_files = len(files)
    n_rows  = len(df)
    labeled = df["outcome"].notna().sum() if "outcome" in df.columns else 0
    dates   = sorted(df["timestamp_utc"].dt.date.unique())
    date_range = f"{dates[0]} → {dates[-1]}" if dates else "?"

    print(f"  {n_rows:,} rows  |  {labeled}/{n_rows} labeled windows  "
          f"|  {date_range}  |  {n_files} file(s)")

    return df
