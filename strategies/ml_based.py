"""
strategies/ml_based.py — Stratégie ML (sklearn/xgboost).

Usage :
  1. Entraîner le modèle avec train_model()
  2. Passer le modèle au constructeur
  3. Le moteur appellera on_tick() à chaque tick

Features disponibles :
  Voir FEATURE_COLS ci-dessous — toutes les colonnes numériques du parquet.
"""
from __future__ import annotations

from typing import Optional, Any
import numpy as np
import pandas as pd

from strategies.base import Strategy, Signal, ExitSignal, Direction


# Colonnes utilisées comme features pour le ML
FEATURE_COLS = [
    "distance_bps",
    "coin_vol_60s_bps", "coin_vol_15m_bps",
    "z_score", "analytical_prob_up", "edge_bps",
    "up_spread_bps", "up_depth_imbalance",
    "poly_trade_count_60s", "poly_trade_size_60s",
    "up_mid_change_5s", "up_mid_change_10s",
    "poly_mid_velocity", "poly_mid_acceleration",
    "coin_return_10s_bps", "coin_return_60s_bps",
    "coin_flow_imbalance_60s", "coin_buy_ratio_60s",
    "btc_return_60s_bps", "eth_return_60s_bps",
    "cross_asset_agreement_60s", "cross_asset_momentum_60s",
    "bn_spread_bps", "bn_depth_imbalance",
    "seconds_remaining", "window_progress_pct",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les features ML depuis un DataFrame de ticks."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    X = X.fillna(0.0)
    return X


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    """
    Prépare les labels ML.
    Label = outcome de la fenêtre (1=UP gagne, 0=DOWN gagne).
    Propagé à tous les ticks de la fenêtre.
    """
    # Propager l'outcome à tous les ticks du même slug
    outcome_map = df.groupby("slug")["outcome"].apply(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
    )
    labels = df["slug"].map(outcome_map)
    return pd.to_numeric(labels, errors="coerce")


def train_model(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    train_ratio: float = 0.7,
) -> Any:
    """
    Entraîne un modèle sur les données historiques avec split temporel.

    IMPORTANT : split temporel (pas aléatoire) pour éviter le data leakage.
    Les 70% premiers ticks servent à l'entraînement, les 30% derniers à la validation.

    df         : DataFrame complet (doit avoir outcome rempli)
    model_type : "xgboost", "random_forest", "logistic"
    train_ratio: proportion des données pour l'entraînement (défaut: 0.7)

    Retourne le modèle entraîné.
    """
    X = prepare_features(df)
    y = prepare_labels(df)

    # Supprimer les lignes sans label
    mask  = y.notna()
    X, y  = X[mask], y[mask]

    if len(X) == 0:
        raise ValueError("No labeled samples — run backfill_outcomes.py first")

    # Split temporel : train sur les premiers 70%, test sur les derniers 30%
    # JAMAIS de split aléatoire — ça introduit du data leakage temporel
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Split trop petit: train={len(X_train)}, test={len(X_test)}")

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42
            )
        except ImportError:
            raise ImportError("pip install xgboost")

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=20, random_state=42, n_jobs=-1
        )

    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    # Validation out-of-sample
    train_acc = (model.predict(X_train) == y_train).mean() * 100
    test_acc  = (model.predict(X_test) == y_test).mean() * 100
    print(f"Model trained on {len(X_train):,} samples ({y_train.mean()*100:.1f}% UP)")
    print(f"  Train accuracy: {train_acc:.1f}%  |  Test accuracy: {test_acc:.1f}%")
    if train_acc - test_acc > 15:
        print(f"  ⚠ Probable overfitting (train-test gap = {train_acc - test_acc:.1f}%)")
    return model


class MLStrategy(Strategy):
    """
    Stratégie basée sur un modèle ML.

    Le modèle prédit P(UP) à chaque tick.
    On entre si P(UP) > threshold ET edge_bps > min_edge.
    """

    def __init__(
        self,
        model: Any,
        threshold:    float = 0.65,   # probabilité minimum pour entrer
        min_edge_bps: float = 200.0,
        min_seconds:  float = 30.0,
        size_usdc:    float = 100.0,
    ):
        super().__init__(name="ML")
        self.model        = model
        self.threshold    = threshold
        self.min_edge_bps = min_edge_bps
        self.min_seconds  = min_seconds
        self.size_usdc    = size_usdc
        self._feature_cols = None  # set on first tick

    def _get_features(self, tick: pd.Series) -> Optional[np.ndarray]:
        """Extrait les features d'un tick pour la prédiction."""
        available = [c for c in FEATURE_COLS if c in tick.index]
        vals = []
        for col in available:
            v = tick.get(col, 0.0)
            vals.append(0.0 if pd.isna(v) else float(v))
        return np.array(vals).reshape(1, -1)

    def on_tick(
        self,
        slug:         str,
        tick:         pd.Series,
        has_position: bool,
    ) -> Optional[Signal | ExitSignal]:

        if has_position:
            return None

        sec  = tick.get("seconds_remaining")
        edge = tick.get("edge_bps")

        if sec is None or pd.isna(sec) or float(sec) < self.min_seconds:
            return None
        if edge is None or pd.isna(edge) or float(edge) < self.min_edge_bps:
            return None

        try:
            X = self._get_features(tick)
            prob_up = self.model.predict_proba(X)[0][1]
        except Exception:
            return None

        if prob_up >= self.threshold:
            return Signal(
                direction  = Direction.UP,
                size_usdc  = self.size_usdc,
                confidence = float(prob_up),
                reason     = f"ml_prob={prob_up:.3f} edge={edge:.0f}",
            )

        return None
