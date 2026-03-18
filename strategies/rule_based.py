"""
strategies/rule_based.py — Stratégie rule-based exemple.

Logique :
  - Entrer UP si edge_bps > seuil ET z_score > seuil ET distance pas trop loin
  - Ne pas entrer dans les dernières 30s (spread s'élargit)
  - Taille fixe par trade
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from strategies.base import Strategy, Signal, ExitSignal, Direction


class EdgeZScoreStrategy(Strategy):
    """
    Stratégie simple basée sur edge_bps et z_score.

    Paramètres :
      min_edge_bps    : edge minimum pour entrer (ex: 300)
      min_z_score     : z_score minimum (ex: 1.5)
      max_distance_bps: distance max par rapport au target (ex: 50)
      min_seconds     : secondes minimum restantes pour entrer (ex: 30)
      size_usdc       : montant par trade en USDC (ex: 100)
      stop_loss_mid   : exit si up_mid tombe sous ce seuil (ex: 0.3)
    """

    def __init__(
        self,
        min_edge_bps:     float = 300.0,
        min_z_score:      float = 1.5,
        max_distance_bps: float = 100.0,
        min_seconds:      float = 30.0,
        size_usdc:        float = 100.0,
        stop_loss_mid:    Optional[float] = None,
    ):
        super().__init__(name="EdgeZScore")
        self.min_edge_bps     = min_edge_bps
        self.min_z_score      = min_z_score
        self.max_distance_bps = max_distance_bps
        self.min_seconds      = min_seconds
        self.size_usdc        = size_usdc
        self.stop_loss_mid    = stop_loss_mid

    def on_tick(
        self,
        slug:         str,
        tick:         pd.Series,
        has_position: bool,
    ) -> Optional[Signal | ExitSignal]:

        # ── Exit signal ───────────────────────────────────────────────────────
        if has_position and self.stop_loss_mid is not None:
            up_mid = tick.get("up_mid")
            if up_mid is not None and not pd.isna(up_mid):
                if float(up_mid) < self.stop_loss_mid:
                    return ExitSignal(slug=slug, reason="stop_loss")

        # ── Entry signal ──────────────────────────────────────────────────────
        if has_position:
            return None  # une seule position par marché

        edge     = tick.get("edge_bps")
        z        = tick.get("z_score")
        dist     = tick.get("distance_bps")
        sec      = tick.get("seconds_remaining")
        up_mid   = tick.get("up_mid")

        # Vérifications nullité
        if any(v is None or pd.isna(v) for v in [edge, z, dist, sec, up_mid]):
            return None

        edge = float(edge)
        z    = float(z)
        dist = float(dist)
        sec  = float(sec)

        # Conditions d'entrée
        if sec < self.min_seconds:
            return None  # trop tard dans la fenêtre
        if abs(dist) > self.max_distance_bps:
            return None  # trop loin du target
        if edge > self.min_edge_bps and z > self.min_z_score:
            return Signal(
                direction  = Direction.UP,
                size_usdc  = self.size_usdc,
                confidence = min(1.0, edge / 1000),
                reason     = f"edge={edge:.0f}bps z={z:.2f}",
                stop_loss  = self.stop_loss_mid,
            )

        return None


class FlowMomentumStrategy(Strategy):
    """
    Stratégie basée sur le flow Binance + momentum Polymarket.

    Entrer UP si :
      - flow_imbalance_60s > seuil (pression acheteuse)
      - up_mid_change_10s > 0 (momentum positif)
      - edge_bps > 0 (prix attractif)
    """

    def __init__(
        self,
        min_flow_imbalance: float = 0.3,
        min_mid_change:     float = 10.0,   # bps
        min_edge_bps:       float = 100.0,
        min_seconds:        float = 60.0,
        size_usdc:          float = 100.0,
    ):
        super().__init__(name="FlowMomentum")
        self.min_flow_imbalance = min_flow_imbalance
        self.min_mid_change     = min_mid_change
        self.min_edge_bps       = min_edge_bps
        self.min_seconds        = min_seconds
        self.size_usdc          = size_usdc

    def on_tick(
        self,
        slug:         str,
        tick:         pd.Series,
        has_position: bool,
    ) -> Optional[Signal | ExitSignal]:

        if has_position:
            return None

        flow  = tick.get("coin_flow_imbalance_60s")
        chg   = tick.get("up_mid_change_10s")
        edge  = tick.get("edge_bps")
        sec   = tick.get("seconds_remaining")

        if any(v is None or pd.isna(v) for v in [flow, chg, edge, sec]):
            return None

        flow = float(flow)
        chg  = float(chg)
        edge = float(edge)
        sec  = float(sec)

        if sec < self.min_seconds:
            return None

        if (flow > self.min_flow_imbalance
                and chg > self.min_mid_change
                and edge > self.min_edge_bps):
            return Signal(
                direction  = Direction.UP,
                size_usdc  = self.size_usdc,
                confidence = min(1.0, flow),
                reason     = f"flow={flow:.2f} chg={chg:.1f}bps edge={edge:.0f}",
            )

        return None
