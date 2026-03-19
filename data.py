"""
data.py — Chargement et préprocessing des données parquet pm_watcher v4.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "coin_price", "target_price", "distance_bps",
    "coin_vol_10s_bps", "coin_vol_60s_bps", "coin_vol_15m_bps",
    "z_score", "analytical_prob_up", "edge_bps",
    "up_bid", "up_ask", "up_mid", "up_spread_bps",
    "up_best_bid_size", "up_best_ask_size",
    # up_book_bids / up_book_asks sont des colonnes JSON string — pas dans NUMERIC_COLS
    "up_bid_depth_top5", "up_ask_depth_top5", "up_depth_imbalance",
    "poly_trade_count_10s", "poly_trade_count_60s",
    "poly_trade_size_10s", "poly_trade_size_60s",
    "up_mid_change_5s", "up_mid_change_10s",
    "poly_mid_velocity", "poly_mid_acceleration",
    "coin_return_10s_bps", "coin_return_30s_bps", "coin_return_60s_bps",
    "coin_flow_imbalance_10s", "coin_flow_imbalance_60s",
    "coin_buy_ratio_60s", "coin_volume_usdt_60s",
    "btc_return_10s_bps", "btc_return_60s_bps",
    "btc_flow_imbalance_60s", "btc_vol_60s_bps",
    "eth_return_10s_bps", "eth_return_60s_bps",
    "eth_flow_imbalance_60s", "eth_vol_60s_bps",
    "sol_return_10s_bps", "sol_return_60s_bps",
    "sol_flow_imbalance_60s", "sol_vol_60s_bps",
    "xrp_return_10s_bps", "xrp_return_60s_bps",
    "xrp_flow_imbalance_60s", "xrp_vol_60s_bps",
    "cross_asset_agreement_10s", "cross_asset_agreement_60s",
    "cross_asset_momentum_10s", "cross_asset_momentum_60s",
    "bn_best_bid", "bn_best_ask", "bn_mid", "bn_spread_bps",
    "bn_bid_depth5", "bn_ask_depth5", "bn_depth_imbalance",
    "bn_bid_l2_price", "bn_bid_l2_qty", "bn_bid_l3_price", "bn_bid_l3_qty",
    "bn_bid_l4_price", "bn_bid_l4_qty", "bn_bid_l5_price", "bn_bid_l5_qty",
    "bn_ask_l2_price", "bn_ask_l2_qty", "bn_ask_l3_price", "bn_ask_l3_qty",
    "bn_ask_l4_price", "bn_ask_l4_qty", "bn_ask_l5_price", "bn_ask_l5_qty",
    # DOWN top of book
    "down_bid", "down_ask", "down_mid", "down_spread_bps",
    "down_best_bid_size", "down_best_ask_size",
    "down_bid_depth_top5", "down_ask_depth_top5", "down_depth_imbalance",
    # down_book_bids / down_book_asks sont JSON string — pas dans NUMERIC_COLS
    "seconds_remaining", "window_progress_pct",
    "coin_price_at_window_close",
]


def load(
    live_dir: str = "historical_data/live",
    coin: str | None = None,
    interval: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess parquet files.
    Returns clean DataFrame with numeric columns as float64.
    """
    p = Path(live_dir)
    # Fichiers finaux fusionnés (YYYY-MM-DD.parquet) + chunks en cours (*_NNNN.parquet)
    finals = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
    chunks = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.parquet"))
    files  = finals + chunks

    if start_date:
        files = [f for f in files if f.stem >= start_date]
    if end_date:
        files = [f for f in files if f.stem <= end_date]

    if not files:
        raise FileNotFoundError(f"No parquet files found in {live_dir}")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"WARNING: could not read {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    # Normalize timestamps
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"].astype(str).str.replace("Z", "+00:00", regex=False),
        utc=True, errors="coerce"
    )

    # Convert numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan}),
                errors="coerce"
            )

    # Normalize outcome
    df["outcome"] = df["outcome"].astype(str).replace({"": np.nan, "nan": np.nan})

    # NOTE : les colonnes JSON book (up_book_bids/asks, down_book_bids/asks)
    # sont intentionnellement laissées en string.
    # Utiliser get_book(tick, col) dans l'engine pour désérialiser lazily.

    # Filters
    if coin:
        df = df[df["coin"] == coin]
    if interval:
        df = df[df["interval_min"].astype(str) == str(interval)]

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df



def get_book(tick: dict, col: str) -> list:
    """
    Désérialise une colonne book à la demande (lazy).
    Appelé par execution.py tick par tick — pas en bulk.
    Retourne [] si absent ou invalide.

    Args:
        tick : dict d'un tick (from DataFrame.to_dict() ou iterrows())
        col  : nom de la colonne book (ex: 'up_book_asks')
    """
    val = tick.get(col)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip() not in ("", "[]", "None", "nan"):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    return []

def iter_windows(df: pd.DataFrame):
    """
    Itère sur les fenêtres de marché du DataFrame.
    Chaque fenêtre = tous les ticks d'un même slug.

    NOTE performance : utiliser iter_windows_streaming() pour les grands datasets
    (> 1 jour de données) pour éviter le groupby en mémoire.

    Yields:
        DataFrame — tous les ticks d'une fenêtre, triés par timestamp_utc
    """
    if "slug" not in df.columns:
        raise ValueError("Colonne 'slug' manquante — nécessaire pour iter_windows()")

    # Trier une seule fois par (slug, timestamp)
    df_sorted = df.sort_values(["slug", "timestamp_utc"])
    for slug, group in df_sorted.groupby("slug", sort=False):
        yield group.reset_index(drop=True)


def iter_windows_streaming(
    live_dir: str,
    coin:     Optional[str] = None,
    interval: Optional[int] = None,
    start:    Optional[str] = None,
    end:      Optional[str] = None,
):
    """
    Version streaming de iter_windows — RAM constante quelle que soit la durée.

    Au lieu de charger tout le DataFrame, lit les parquets slug par slug.
    RAM utilisée : ~1 fenêtre (~500 ticks) à la fois.

    Utilisé par engine.run_streaming() pour les backtests sur grandes données.

    Yields:
        (slug, DataFrame) — une fenêtre à la fois
    """
    p = Path(live_dir)
    finals = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
    chunks = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.parquet"))

    # Filtrer par date
    def in_range(f):
        d = f.name[:10]
        if start and d < start: return False
        if end   and d > end:   return False
        return True

    finals = [f for f in finals if in_range(f)]
    chunks = [f for f in chunks if in_range(f)]
    final_dates = {f.name[:10] for f in finals}
    files = finals + [c for c in chunks if c.name[:10] not in final_dates]
    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"Aucun parquet dans {live_dir}")

    # ── Vrai streaming : yield par slug au fur et à mesure ──────────────────
    # Chaque slug = une fenêtre de marché (ex: btc-updown-5m-1773863100)
    # Le timestamp de fin de fenêtre est encodé dans le nom du slug.
    # On yield un slug dès qu'on détecte qu'on est passé à une fenêtre suivante
    # pour ce même marché → RAM = O(1 fenêtre par marché) au lieu de O(tout)

    slug_buffers: dict = {}   # slug → list[DataFrame]
    slug_market:  dict = {}   # slug → (coin, interval, window_start_ts)
    all_slugs_seen: set = set()

    def _parse_window_ts(slug: str) -> int:
        """Extrait le timestamp de début de fenêtre depuis le slug name."""
        try:
            return int(slug.rsplit("-", 1)[-1])
        except Exception:
            return 0

    def _market_key(slug: str) -> str:
        """Retourne la clé marché sans le timestamp (ex: btc-updown-5m)."""
        parts = slug.rsplit("-", 1)
        return parts[0] if len(parts) == 2 else slug

    # market_key → slug actif pour ce marché
    active_slug: dict = {}

    for fpath in files:
        try:
            df_file = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  ⚠ {fpath.name} illisible : {e}")
            continue

        if coin:
            df_file = df_file[df_file["coin"] == coin]
        if interval:
            df_file = df_file[df_file["interval_min"] == str(interval)]

        if df_file.empty:
            continue

        df_file["timestamp_utc"] = pd.to_datetime(
            df_file["timestamp_utc"], utc=True, errors="coerce"
        )
        for col in NUMERIC_COLS:
            if col in df_file.columns:
                df_file[col] = pd.to_numeric(df_file[col], errors="coerce")

        for slug, grp in df_file.groupby("slug", sort=False):
            mk = _market_key(slug)
            prev_slug = active_slug.get(mk)

            # Nouveau slug pour ce marché → yield le précédent si complet
            if prev_slug is not None and prev_slug != slug:
                if prev_slug in slug_buffers and slug_buffers[prev_slug]:
                    window = pd.concat(slug_buffers[prev_slug], ignore_index=True)
                    window = window.sort_values("timestamp_utc").reset_index(drop=True)
                    del slug_buffers[prev_slug]
                    yield window

            active_slug[mk] = slug
            if slug not in slug_buffers:
                slug_buffers[slug] = []
            slug_buffers[slug].append(grp)

    # Yield les fenêtres restantes (dernières de chaque marché)
    for slug in sorted(slug_buffers.keys()):
        if slug_buffers[slug]:
            window = pd.concat(slug_buffers[slug], ignore_index=True)
            window = window.sort_values("timestamp_utc").reset_index(drop=True)
            yield window


def window_summary(df: pd.DataFrame) -> dict:
    """
    Résumé rapide des fenêtres disponibles dans le DataFrame.
    Utilisé par run_backtest.py pour afficher les stats au démarrage.
    """
    total_rows   = len(df)
    labeled      = df["outcome"].notna().sum() if "outcome" in df.columns else 0
    total_windows = df["slug"].nunique() if "slug" in df.columns else 0
    labeled_windows = (
        df[df["outcome"].notna()]["slug"].nunique()
        if "outcome" in df.columns and "slug" in df.columns else 0
    )
    ts = df["timestamp_utc"] if "timestamp_utc" in df.columns else pd.Series([])
    date_range = (ts.min(), ts.max()) if len(ts) > 0 else (None, None)
    return {
        "total_rows":      total_rows,
        "labeled_windows": labeled_windows,
        "total_windows":   total_windows,
        "date_range":      date_range,
    }
