"""Name -> model class. The config's ``models:`` section refers to these names."""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Type

from .base import BaseModel
from .baselines import AlwaysLongModel, MomentumRuleModel
from .ml_models import HistGBModel, LightGBMModel, LogisticModel, RandomForestModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "always_long": AlwaysLongModel,
    "momentum_rule": MomentumRuleModel,
    "logistic": LogisticModel,
    "random_forest": RandomForestModel,
    "hist_gb": HistGBModel,
    "lightgbm": LightGBMModel,
}

_OPTIONAL_DEPS = {"lightgbm": "lightgbm"}
BASELINES = {"always_long", "momentum_rule"}


def is_available(name: str) -> bool:
    dep = _OPTIONAL_DEPS.get(name)
    return dep is None or importlib.util.find_spec(dep) is not None


def available_models(requested: List[str], verbose: bool = True) -> List[str]:
    """Keep requested models that are known and whose library is installed."""
    out = []
    for name in requested:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
        if is_available(name):
            out.append(name)
        elif verbose:
            print(f"[models] skipping '{name}': library '{_OPTIONAL_DEPS[name]}' not installed")
    return out


def build_model(name: str, seed: int = 42, **params: Any) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed=seed, **params)
