"""Rule-based baselines. Every ML model has to beat these after costs."""
from __future__ import annotations

import numpy as np

from .base import BaseModel


def proba_from_up(model: BaseModel, p_up: np.ndarray) -> np.ndarray:
    """Turn 'probability of going up' into a predict_proba matrix in the order of
    ``model.classes_``: P(1) = p_up, P(0 or -1) = 1 - p_up (hold class gets 0)."""
    cols = []
    for c in model.classes_:
        c = float(c)
        if c == 1.0:
            cols.append(p_up)
        elif c == -1.0 or (c == 0.0 and -1.0 not in model.classes_):
            cols.append(1.0 - p_up)
        else:
            cols.append(np.zeros_like(p_up))
    return np.column_stack(cols)


class AlwaysLongModel(BaseModel):
    """Always predicts 'up' -> always invested (= buy and hold)."""
    name = "always_long"

    def _fit(self, X, y):
        pass

    def _predict_proba(self, X):
        return proba_from_up(self, np.ones(len(X)))


class MomentumRuleModel(BaseModel):
    """Predicts 'up' when the chosen momentum feature (default ret_21) is positive."""
    name = "momentum_rule"

    def __init__(self, feature: str = "ret_21", seed: int = 42):
        super().__init__(seed=seed, feature=feature)
        self.feature = feature

    def _fit(self, X, y):
        if self.feature not in X.columns:
            raise ValueError(f"momentum_rule needs feature '{self.feature}'")

    def _predict_proba(self, X):
        return proba_from_up(self, (X[self.feature].to_numpy() > 0).astype(float))
