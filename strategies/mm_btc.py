"""
strategies/mm_btc.py — InformedMarketMaker pour BTC Polymarket.

Stratégie mixte : Market Making + Signal Directionnel Binance

Modes :
  FLAT   : Marché calme (|bn_ret| < flat_threshold)
           → Placer bid UP + ask UP symétriques autour du mid
           → Gagner le spread, risque d'adverse selection faible

  SKEWED : Move Binance modéré (flat < |bn_ret| < strong_threshold)
           → Garder le côté dans le sens du move
           → Retirer/élargir le côté contre le move
           → Spread effectif plus large

  DIRECTIONAL : Move fort (|bn_ret| > strong_threshold)
           → Retirer complètement les quotes contre-tendance
           → Garder uniquement bid UP (si Binance monte)
           → Option : placer market order si signal très fort
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from strategies.mm_base import MMStrategy, MMQuotes, MMQuote


class InformedMarketMaker(MMStrategy):
    """
    Market Maker informé par le signal Binance sur BTC Polymarket.

    Paramètres :
      base_size_usdc     : taille de base par ordre (ex: 200)
      spread_offset      : offset par rapport au mid pour bid/ask (ex: 0.005 = demi-tick)
      flat_threshold     : |bn_ret| sous lequel on est en mode FLAT (ex: 3bps)
      skew_threshold     : |bn_ret| au-dessus duquel on skew (ex: 5bps)
      strong_threshold   : |bn_ret| pour mode directionnel pur (ex: 10bps)
      min_prob           : up_mid minimum pour trader (ex: 0.2)
      max_prob           : up_mid maximum pour trader (ex: 0.8)
      min_seconds        : secondes min restantes dans la fenêtre
      max_inventory_pct  : % du capital max en inventaire net
      skew_factor        : facteur d'élargissement du spread côté défavorable
    """

    def __init__(
        self,
        base_size_usdc:    float = 200.0,
        spread_offset:     float = 0.005,   # demi-tick
        flat_threshold:    float = 3.0,     # bps
        skew_threshold:    float = 5.0,     # bps
        strong_threshold:  float = 10.0,    # bps
        min_prob:          float = 0.2,
        max_prob:          float = 0.8,
        min_seconds:       float = 30.0,
        max_seconds:       float = 270.0,
        skew_factor:       float = 2.0,     # élargit le spread côté défavorable de 2x
    ):
        super().__init__(name="InformedMM")
        self.base_size_usdc   = base_size_usdc
        self.spread_offset    = spread_offset
        self.flat_threshold   = flat_threshold
        self.skew_threshold   = skew_threshold
        self.strong_threshold = strong_threshold
        self.min_prob         = min_prob
        self.max_prob         = max_prob
        self.min_seconds      = min_seconds
        self.max_seconds      = max_seconds
        self.skew_factor      = skew_factor

    def on_window_start(self, slug, tick):
        pass

    def on_tick(self, slug, tick, inventory, bn_ret):
        # Données du tick
        up_mid  = tick.get("up_mid")
        up_bid  = tick.get("up_bid")
        up_ask  = tick.get("up_ask")
        dn_bid  = tick.get("down_bid")
        dn_ask  = tick.get("down_ask")
        sec     = tick.get("seconds_remaining")

        if any(v is None or (isinstance(v, float) and v != v)  # isnan
               for v in [up_mid, up_bid, up_ask, dn_bid, sec]):
            return None

        up_mid = float(up_mid)
        up_bid = float(up_bid)
        up_ask = float(up_ask)
        dn_bid = float(dn_bid)
        sec    = float(sec)

        # Filtres de base
        if not (self.min_seconds <= sec <= self.max_seconds):
            return MMQuotes()  # retirer toutes les quotes en dehors de la fenêtre

        if not (self.min_prob <= up_mid <= self.max_prob):
            return MMQuotes()  # marché trop directionnel, pas de MM

        # Spread actuel
        spread = up_ask - up_bid  # typiquement 0.01

        # ── MODE FLAT : placer des 2 côtés symétriques ──────────────────────
        if abs(bn_ret) < self.flat_threshold:
            # Bid légèrement sous le bid actuel, ask légèrement au-dessus
            bid_price = round(up_bid, 4)   # au meilleur bid
            ask_price = round(up_ask, 4)   # au meilleur ask
            
            if bid_price <= 0 or ask_price >= 1 or bid_price >= ask_price:
                return None

            return MMQuotes(
                bid_up = MMQuote("bid", "UP", bid_price, self.base_size_usdc),
                ask_up = MMQuote("ask", "UP", ask_price, self.base_size_usdc),
            )

        # ── MODE SKEWED : Binance donne une direction ───────────────────────
        if abs(bn_ret) < self.strong_threshold:
            if bn_ret > 0:
                # Binance monte → garder bid UP au prix normal
                # Élargir ask UP (plus cher pour les acheteurs)
                bid_price = round(up_bid, 4)
                ask_price = round(min(up_ask + spread * (self.skew_factor - 1), 0.99), 4)
            else:
                # Binance baisse → garder ask UP au prix normal
                # Baisser bid UP (moins attractif à acheter)
                bid_price = round(max(up_bid - spread * (self.skew_factor - 1), 0.01), 4)
                ask_price = round(up_ask, 4)

            if bid_price <= 0 or ask_price >= 1 or bid_price >= ask_price:
                return None

            return MMQuotes(
                bid_up = MMQuote("bid", "UP", bid_price, self.base_size_usdc),
                ask_up = MMQuote("ask", "UP", ask_price, self.base_size_usdc),
            )

        # ── MODE DIRECTIONNEL : move fort → un seul côté ───────────────────
        if bn_ret > 0:
            # Binance monte fort → seulement bid UP (on achète UP à l'ancien prix)
            bid_price = round(up_bid, 4)
            if bid_price <= 0 or bid_price >= 1:
                return MMQuotes()
            return MMQuotes(
                bid_up = MMQuote("bid", "UP", bid_price, self.base_size_usdc * 1.5),
            )
        else:
            # Binance baisse fort → seulement bid DOWN (on achète DOWN = parie sur baisse)
            dn_bid_f = float(dn_bid) if dn_bid else 0
            if dn_bid_f <= 0 or dn_bid_f >= 1:
                return MMQuotes()
            dn_mid = 1.0 - up_mid
            if not (self.min_prob <= dn_mid <= self.max_prob):
                return MMQuotes()
            return MMQuotes(
                bid_down = MMQuote("bid", "DOWN", round(dn_bid_f, 4), self.base_size_usdc * 1.5),
            )
