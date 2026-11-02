"""Shared fixtures. Tests run without installing the package (src/ is put on the path)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_ml.config import load_config  # noqa: E402
from trading_ml.data.synthetic import generate_synthetic_prices  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def banner(title: str) -> None:
    """Print a visible header so the -s output is easy to follow."""
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


@pytest.fixture(scope="session")
def cfg():
    """Small, fast configuration used by most tests."""
    return load_config(
        overrides={
            "data": {"synthetic": {"n_tickers": 5, "n_days": 700}},
            "validation": {"n_splits": 3, "test_days": 63, "min_train_days": 200, "holdout_days": 30},
            "models": {
                "always_long": {},
                "momentum_rule": {"feature": "ret_21"},
                "logistic": {"C": 0.1},
                "hist_gb": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 3},
            },
            "line_b": {"top_k": 2, "bottom_k": 0, "rebalance_every": 5},
        }
    )


@pytest.fixture(scope="session")
def prices(cfg):
    syn = cfg["data"]["synthetic"]
    return generate_synthetic_prices(
        n_tickers=syn["n_tickers"],
        n_days=syn["n_days"],
        start=syn["start"],
        momentum_strength=syn["momentum_strength"],
        seed=cfg["project"]["seed"],
    )
