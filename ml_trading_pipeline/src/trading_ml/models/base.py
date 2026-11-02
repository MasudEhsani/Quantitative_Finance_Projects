"""The single interface every model implements.

    model = build_model("logistic", C=0.1)
    model.fit(X_train, y_train)
    score = model.predict_score(X_test)      # float in [0, 1]; 0.5 = no opinion

``predict_score`` unifies label kinds:
  * binary labels {0, 1}        -> P(class 1)
  * three-class {-1, 0, 1}      -> 0.5 + 0.5 * (P(+1) - P(-1))
so signal rules (buy above 0.55, sell below 0.45, rank for Line B) work for all.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    def __init__(self, seed: int = 42, **params: Any):
        self.seed = seed
        self.params: Dict[str, Any] = dict(params)
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    # ---- to implement --------------------------------------------------------
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    # ---- shared logic -------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "BaseModel":
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if np.isnan(y.astype(float)).any():
            raise ValueError("y contains NaN - drop unlabeled rows before fit")
        self.classes_ = np.unique(y)
        self.feature_names_ = list(X.columns)
        self._fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError(f"{self.name}: call fit() before predict")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: missing features {missing[:5]}...")
        return self._predict_proba(X[self.feature_names_])

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        classes = list(self.classes_)
        if len(classes) == 1:                         # degenerate training set
            return np.full(len(X), 1.0 if classes[0] > 0 else 0.0)
        if set(classes) <= {0.0, 1.0}:
            return proba[:, classes.index(1.0)]
        up = proba[:, classes.index(1.0)] if 1.0 in classes else 0.0
        down = proba[:, classes.index(-1.0)] if -1.0 in classes else 0.0
        return 0.5 + 0.5 * (up - down)

    def feature_importance(self) -> Optional[pd.Series]:
        return None

    def get_params(self) -> Dict[str, Any]:
        return {"model": self.name, "seed": self.seed, **self.params}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"
