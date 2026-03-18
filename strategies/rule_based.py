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

from strategies.base import Strategy, Signal, LimitSignal, ExitSignal, Direction


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


class LeadBinanceStrategy(Strategy):
    """
    Stratégie exploitant le lead signal Binance → Polymarket (~500ms).

    Logique :
      1. Binance fait un move fort (coin_return_10s_bps > seuil)
      2. On anticipe que Polymarket va suivre dans ~500ms
      3. On pose un limit order UP ou DOWN juste au-dessus du bid actuel
         (maker, 0 frais) avant que le mid ajuste
      4. Si fillé, on tient jusqu'à la clôture de la fenêtre
         (ou exit anticipé si take_profit atteint)

    Paramètres :
      bn_move_bps      : seuil de move Binance 10s pour déclencher (ex: 20)
      limit_offset     : offset du limit par rapport au bid (ex: 0.01 = +1 tick)
      min_seconds      : secondes min restantes dans la fenêtre (ex: 60)
      take_profit_bps  : exit si mid a monté de X bps depuis entrée (0 = hold)
      stop_loss_bps    : exit si mid a baissé de X bps depuis entrée (0 = non)
      size_usdc        : montant par trade en USDC
      queue_model      : "conservative" ou "optimistic"
      cooldown_ticks   : attendre N ticks après un signal avant de re-signaler
    """

    def __init__(
        self,
        bn_move_bps:          float = 20.0,
        limit_offset:         float = 0.01,
        min_seconds:          float = 60.0,
        max_seconds:          float = 270.0,  # pas d'entree dans les 30 dernieres sec
        min_prob:             float = 0.0,    # up_mid min pour entrer UP
        max_prob:             float = 1.0,    # up_mid max pour entrer UP
        take_profit_bps:      float = 500.0,
        stop_loss_bps:        float = 0.0,
        max_hold_seconds:     float = 30.0,
        size_usdc:            float = 100.0,
        queue_model:          str   = "conservative",
        cooldown_ticks:       int   = 20,
        cooldown_after_loss_s:float = 60.0,
    ):
        super().__init__(name="LeadBinance")
        self.bn_move_bps           = bn_move_bps
        self.limit_offset          = limit_offset
        self.min_seconds           = min_seconds
        self.max_seconds           = max_seconds
        self.min_prob              = min_prob
        self.max_prob              = max_prob
        self.take_profit_bps       = take_profit_bps
        self.stop_loss_bps         = stop_loss_bps
        self.max_hold_seconds      = max_hold_seconds
        self.size_usdc             = size_usdc
        self.queue_model           = queue_model
        self.cooldown_ticks        = cooldown_ticks
        self.cooldown_after_loss_s = cooldown_after_loss_s
        self._last_signal_tick: dict = {}
        self._entry_mid:        dict = {}
        self._entry_ts:         dict = {}
        self._loss_ts:          dict = {}   # slug -> ts derniere perte

    def on_window_start(self, slug: str, tick) -> None:
        self._last_signal_tick[slug] = 999  # ready dès le début
        self._entry_mid.pop(slug, None)
        self._entry_ts.pop(slug, None)
        self._loss_ts.pop(slug, None)

    def on_tick(
        self,
        slug:         str,
        tick,
        has_position: bool,
    ) -> Optional[Signal | LimitSignal | ExitSignal]:

        # ── Incrémenter le cooldown ───────────────────────────────────────────
        self._last_signal_tick[slug] = self._last_signal_tick.get(slug, 999) + 1

        # ── Exit : take profit / stop loss / max hold ──────────────────────────
        if has_position:
            up_mid    = tick.get("up_mid")
            # entry_fill_price est injecté par l'engine dans tick_dict au moment du fill
            # → référence correcte pour calculer le move depuis l'entrée réelle
            entry_price = tick.get("_entry_fill_price")  # injecté par engine
            if entry_price is None:
                entry_price = self._entry_mid.get(slug)  # fallback signal
            entry_ts  = tick.get("_entry_ts") or self._entry_ts.get(slug)
            cur_ts    = tick.get("timestamp_utc")

            direction = tick.get("_entry_direction", "UP")
            current_mid = tick.get("down_mid") if direction == "DOWN" else tick.get("up_mid")
            if current_mid is not None and not pd.isna(current_mid) and entry_price is not None and float(entry_price) > 0:
                current_mid = float(current_mid)
                move_bps = (current_mid - float(entry_price)) / float(entry_price) * 10000
                if self.take_profit_bps > 0 and move_bps >= self.take_profit_bps:
                    return ExitSignal(slug=slug, reason=f"take_profit +{move_bps:.0f}bps")
                if self.stop_loss_bps > 0 and move_bps <= -self.stop_loss_bps:
                    self._loss_ts[slug] = cur_ts
                    return ExitSignal(slug=slug, reason=f"stop_loss {move_bps:.0f}bps")

            # Max hold — calculé depuis le fill réel (injecté par engine)
            if (self.max_hold_seconds > 0 and entry_ts is not None
                    and cur_ts is not None):
                try:
                    held = (pd.Timestamp(cur_ts) - pd.Timestamp(entry_ts)).total_seconds()
                    if held >= self.max_hold_seconds:
                        return ExitSignal(slug=slug, reason=f"max_hold {held:.0f}s")
                except Exception:
                    pass
            return None

        # ── Filtres pre-entry ─────────────────────────────────────────────────
        sec = tick.get("seconds_remaining")
        if sec is None or pd.isna(sec):
            return None
        sec = float(sec)
        if sec < self.min_seconds or sec > self.max_seconds:
            return None

        # Cooldown ticks
        if self._last_signal_tick.get(slug, 999) < self.cooldown_ticks:
            return None

        # Cooldown apres perte
        if self.cooldown_after_loss_s > 0:
            loss_ts = self._loss_ts.get(slug)
            cur_ts2 = tick.get("timestamp_utc")
            if loss_ts is not None and cur_ts2 is not None:
                try:
                    elapsed = (pd.Timestamp(cur_ts2) - pd.Timestamp(loss_ts)).total_seconds()
                    if elapsed < self.cooldown_after_loss_s:
                        return None
                except Exception:
                    pass

        # ── Signal Binance ────────────────────────────────────────────────────
        bn_ret = tick.get("coin_return_10s_bps")
        if bn_ret is None or pd.isna(bn_ret):
            return None
        bn_ret = float(bn_ret)

        up_bid  = tick.get("up_bid")
        up_mid  = tick.get("up_mid")
        dn_bid  = tick.get("down_bid")

        if any(v is None or pd.isna(v) for v in [up_bid, up_mid, dn_bid]):
            return None

        up_bid = float(up_bid)
        up_mid = float(up_mid)
        dn_bid = float(dn_bid)

        # Move UP sur Binance → acheter UP (filtre prob)
        if bn_ret >= self.bn_move_bps and self.min_prob <= up_mid <= self.max_prob:
            limit_price = round(up_bid + self.limit_offset, 4)
            # S'assurer que le limit est sous l'ask (sinon ordre marketable)
            up_ask = tick.get("up_ask")
            if up_ask is not None and not pd.isna(up_ask):
                limit_price = min(limit_price, float(up_ask) - 0.005)
            if limit_price <= 0 or limit_price >= 1:
                return None

            self._last_signal_tick[slug] = 0
            self._entry_mid[slug] = up_mid
            self._entry_ts[slug]  = tick.get("timestamp_utc")
            return LimitSignal(
                direction    = Direction.UP,
                size_usdc    = self.size_usdc,
                limit_price  = limit_price,
                queue_model  = self.queue_model,
                confidence   = min(1.0, bn_ret / 100),
                reason       = f"bn_ret={bn_ret:.1f}bps limit@{limit_price:.4f}",
            )

        # Move DOWN sur Binance → acheter DOWN (filtre prob inverse)
        dn_mid = 1.0 - up_mid
        if bn_ret <= -self.bn_move_bps and self.min_prob <= dn_mid <= self.max_prob:
            limit_price = round(dn_bid + self.limit_offset, 4)
            dn_ask = tick.get("down_ask")
            if dn_ask is not None and not pd.isna(dn_ask):
                limit_price = min(limit_price, float(dn_ask) - 0.005)
            if limit_price <= 0 or limit_price >= 1:
                return None

            self._last_signal_tick[slug] = 0
            self._entry_mid[slug] = dn_bid
            self._entry_ts[slug]  = tick.get("timestamp_utc")
            return LimitSignal(
                direction    = Direction.DOWN,
                size_usdc    = self.size_usdc,
                limit_price  = limit_price,
                queue_model  = self.queue_model,
                confidence   = min(1.0, abs(bn_ret) / 100),
                reason       = f"bn_ret={bn_ret:.1f}bps DOWN limit@{limit_price:.4f}",
            )

        return None
