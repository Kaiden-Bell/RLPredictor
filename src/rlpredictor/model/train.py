"""Logistic regression on per-game outcomes, saved with its metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from rlpredictor import config
from rlpredictor.model.features import FEATURES


@dataclass
class GameModel:
    features: list[str]               # active features (subset of FEATURES)
    weights: np.ndarray               # effective weights over FEATURES (0 for inactive)
    trained_through: str | None = None
    n_games: int = 0
    metrics: dict = field(default_factory=dict)

    def logit(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.weights

    def p_game(self, x: np.ndarray) -> np.ndarray | float:
        z = self.logit(x)
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or config.MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "GameModel":
        d = joblib.load(Path(path or config.MODEL_PATH))
        return cls(**d)


def fit(X: np.ndarray, y: np.ndarray, active: list[str] | None = None, C: float = 1.0) -> GameModel:
    active = active or list(FEATURES)
    idx = [FEATURES.index(f) for f in active]
    Xa = X[:, idx]
    # Scale without centering: features are antisymmetric differences, so no intercept.
    scale = np.sqrt((Xa ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    lr = LogisticRegression(fit_intercept=False, C=C, max_iter=1000)
    lr.fit(Xa / scale, y)
    w = np.zeros(len(FEATURES))
    w[idx] = lr.coef_[0] / scale
    return GameModel(features=active, weights=w, n_games=len(y))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))
