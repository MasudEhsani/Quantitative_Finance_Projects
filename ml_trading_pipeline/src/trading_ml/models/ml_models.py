"""Machine-learning models behind the common interface.

Add a new model: subclass ``SklearnModel`` (or ``BaseModel``), implement
``_make_estimator``, and register it in ``registry.py``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class SklearnModel(BaseModel):
    """Wraps any sklearn classifier. Scalers live inside the pipeline, so they are
    fitted on the training fold only (no leakage from test data)."""

    def _make_estimator(self):
        raise NotImplementedError

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X.to_numpy(dtype=float), y)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X.to_numpy(dtype=float))

    def feature_importance(self) -> Optional[pd.Series]:
        est = self.estimator_[-1] if hasattr(self.estimator_, "steps") else self.estimator_
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "coef_"):
            vals = np.abs(est.coef_).mean(axis=0)
        else:
            return None
        return pd.Series(vals, index=self.feature_names_).sort_values(ascending=False)


class LogisticModel(SklearnModel):
    name = "logistic"

    def _make_estimator(self):
        p = {"C": 1.0, "max_iter": 2000, **self.params}
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=self.seed, **p))


class RandomForestModel(SklearnModel):
    name = "random_forest"

    def _make_estimator(self):
        p = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50, "n_jobs": -1, **self.params}
        return RandomForestClassifier(random_state=self.seed, **p)


class HistGBModel(SklearnModel):
    """sklearn's LightGBM-style gradient boosting. Always available, handles NaN."""
    name = "hist_gb"

    def _make_estimator(self):
        p = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, **self.params}
        return HistGradientBoostingClassifier(random_state=self.seed, **p)


class LightGBMModel(SklearnModel):
    name = "lightgbm"

    def _make_estimator(self):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is not installed: pip install lightgbm") from exc
        p = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 50,
             "verbose": -1, **self.params}
        return lgb.LGBMClassifier(random_state=self.seed, **p)

    def _fit(self, X, y):
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)                 # keep column names for LightGBM

    def _predict_proba(self, X):
        return self.estimator_.predict_proba(X)
