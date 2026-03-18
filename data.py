"""
data.py — Chargement et préprocessing des données parquet pm_watcher v4.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
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

    # Désérialiser les colonnes JSON book complet → listes Python [[price, size], ...]
    for col in ("up_book_bids", "up_book_asks", "down_book_bids", "down_book_asks"):
        if col in df.columns:
            def _parse_book(val):
                if not val or str(val) in ("", "nan", "None", "[]"):
                    return []
                try:
                    return json.loads(val)
                except Exception:
                    return []
            df[col] = df[col].apply(_parse_book)

    # Filters
    if coin:
        df = df[df["coin"] == coin]
    if interval:
        df = df[df["interval_min"].astype(str) == str(interval)]

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def iter_windows(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """
    Iterate over individual market windows (by slug).
    Each window = all ticks for one slug, sorted by time.
    Only yields windows with known outcome.
    """
    for slug, group in df.groupby("slug"):
        group = group.sort_values("timestamp_utc").reset_index(drop=True)
        # Only yield windows with known outcome
        outcome = group["outcome"].dropna()
        if len(outcome) == 0:
            continue
        if not set(outcome.unique()) <= {"0", "1", "0.0", "1.0"}:
            continue
        yield group


def window_summary(df: pd.DataFrame) -> dict:
    """Summary stats for a loaded dataset."""
    return {
        "total_rows":    len(df),
        "total_windows": df["slug"].nunique(),
        "labeled_windows": df.groupby("slug")["outcome"].apply(
            lambda x: x.dropna().isin(["0","1","0.0","1.0"]).any()
        ).sum(),
        "date_range":    (df["timestamp_utc"].min(), df["timestamp_utc"].max()),
        "coins":         sorted(df["coin"].unique().tolist()),
        "intervals":     sorted(df["interval_min"].astype(str).unique().tolist()),
    }
