"""Step 4 - walk-forward splits (no overlap, purge, hold-out) and the model interface."""
import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.features import build_features
from trading_ml.labels import make_labels
from trading_ml.models import MODEL_REGISTRY, available_models, build_model, is_available
from trading_ml.validation import WalkForwardSplitter


# ----------------------------------------------------------------------------- splits
@pytest.fixture(scope="module")
def dates(prices):
    return prices.index.get_level_values("date").unique()


def test_walk_forward_layout(dates, cfg):
    banner("SPLITS: walk-forward folds, purge gap and hold-out")
    sp = WalkForwardSplitter.from_config(cfg["validation"])
    folds = sp.split(dates)
    hold = sp.holdout_dates(dates)
    for f in folds:
        print("  " + f.describe())
    print(f"  holdout: {hold.min().date()}..{hold.max().date()} ({len(hold)} d)  purge={sp.purge_days} d")
    assert len(folds) == cfg["validation"]["n_splits"]
    pos = {d: i for i, d in enumerate(dates)}
    for f in folds:
        gap = pos[f.test_dates.min()] - pos[f.train_dates.max()] - 1
        assert gap == sp.purge_days, "exactly purge_days dates between train and test"
        assert len(f.train_dates) >= sp.min_train_days
        assert len(f.test_dates) == sp.test_days
        assert f.train_dates.max() < f.test_dates.min()
        assert not set(f.test_dates) & set(hold), "hold-out never used for testing"
        assert not set(f.train_dates) & set(hold), "hold-out never used for training"
    for a, b in zip(folds, folds[1:]):
        assert a.test_dates.max() < b.test_dates.min(), "test blocks do not overlap"
        assert len(b.train_dates) > len(a.train_dates), "expanding window grows"


def test_purge_removes_label_leakage(prices, dates, cfg):
    banner("SPLITS: with purge >= horizon no training label 'sees' a test-period price")
    h = cfg["labels"]["horizon"]
    for purge, expect_leak in [(0, True), (h, False)]:
        sp = WalkForwardSplitter(n_splits=3, test_days=63, min_train_days=200, purge_days=purge)
        leaks = 0
        for f in sp.split(dates):
            last_train = f.train_dates.max()
            label_end = dates[dates.get_loc(last_train) + h]      # last price the label uses
            leaks += int(label_end >= f.test_dates.min())
        print(f"  purge={purge}: folds whose last training label reaches into test = {leaks}")
        assert (leaks > 0) == expect_leak


def test_rolling_window_and_errors(dates):
    banner("SPLITS: rolling window caps train length; too little data raises a clear error")
    sp = WalkForwardSplitter(n_splits=2, test_days=50, min_train_days=100, purge_days=5, max_train_days=150)
    lengths = [len(f.train_dates) for f in sp.split(dates)]
    print(f"  rolling train lengths: {lengths}")
    assert all(n == 150 for n in lengths)
    with pytest.raises(ValueError) as exc:
        WalkForwardSplitter(n_splits=20, test_days=60, min_train_days=500, purge_days=5).split(dates)
    print(f"  error message: {exc.value}")


# ----------------------------------------------------------------------------- models
@pytest.fixture(scope="module")
def xy(prices, cfg):
    X = build_features(prices, cfg["features"])
    y = make_labels(prices, cfg["labels"]["horizon"], "direction")
    data = X.join(y).dropna()
    return data.drop(columns="label"), data["label"]


def test_registry():
    banner("MODELS: registry and optional libraries")
    for name in MODEL_REGISTRY:
        print(f"  {name:15s} available={is_available(name)}")
    assert "lightgbm" not in available_models(["lightgbm"], verbose=True) or is_available("lightgbm")
    with pytest.raises(KeyError):
        build_model("does_not_exist")


@pytest.mark.parametrize("name", ["always_long", "momentum_rule", "logistic", "random_forest", "hist_gb"])
def test_model_interface(name, xy):
    banner(f"MODELS: '{name}' fits, predicts probabilities/scores and survives pickling")
    X, y = xy
    split = int(len(X) * 0.7)
    params = {"feature": "ret_21"} if name == "momentum_rule" else {}
    if name == "random_forest":
        params = {"n_estimators": 50}
    m = build_model(name, **params).fit(X.iloc[:split], y.iloc[:split])
    proba = m.predict_proba(X.iloc[split:])
    score = m.predict_score(X.iloc[split:])
    acc = float(((score > 0.5) == y.iloc[split:].to_numpy()).mean())
    print(f"  {name:14s} proba shape={proba.shape} score range=[{score.min():.3f}, {score.max():.3f}] "
          f"mean={score.mean():.3f} accuracy={acc:.3f}")
    assert proba.shape == (len(X) - split, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((score >= 0) & (score <= 1)).all()
    # model survives pickling (needed to log it to MLflow and to reload for paper trading)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.allclose(m2.predict_score(X.iloc[split:]), score)
    imp = m.feature_importance()
    if imp is not None:
        print(f"     top-3 features: {list(imp.index[:3])}")


def test_three_class_score_mapping(xy, prices, cfg):
    banner("MODELS: three-class labels map to a single score (0.5 = neutral)")
    X, _ = xy
    y3 = make_labels(prices, 5, "three_class", dead_zone=0.01).reindex(X.index)
    m = build_model("logistic").fit(X, y3)
    proba = m.predict_proba(X.iloc[:5])
    score = m.predict_score(X.iloc[:5])
    manual = 0.5 + 0.5 * (proba[:, 2] - proba[:, 0])   # classes are sorted [-1, 0, 1]
    print(f"  classes={m.classes_}  proba row0={np.round(proba[0], 3)}  score0={score[0]:.3f}")
    assert list(m.classes_) == [-1.0, 0.0, 1.0]
    assert np.allclose(score, manual)
    base = build_model("always_long").fit(X, y3)
    assert np.allclose(base.predict_score(X.iloc[:5]), 1.0)


def test_fit_guards(xy):
    banner("MODELS: clear errors for NaN labels, missing features, predict before fit")
    X, y = xy
    m = build_model("logistic")
    with pytest.raises(RuntimeError):
        m.predict_proba(X)
    with pytest.raises(ValueError):
        m.fit(X, y.where(y.index != y.index[0]))
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X.drop(columns=["ret_5"]))
    print("  all three guards raise -> OK")


def test_models_learn_planted_signal(cfg):
    banner("MODELS: on data with strong planted momentum, ML beats a coin flip out-of-sample")
    from trading_ml.data import generate_synthetic_prices
    p = generate_synthetic_prices(n_tickers=6, n_days=1200, momentum_strength=0.06, seed=7)
    X = build_features(p, cfg["features"])
    y = make_labels(p, 5, "direction")
    data = X.join(y).dropna()
    dts = data.index.get_level_values("date")
    cut = dts.unique()[800]
    tr, te = data[dts < cut], data[dts >= dates_after(dts, cut, 5)]
    accs = {}
    base = max(te["label"].mean(), 1 - te["label"].mean())
    for name in ["logistic", "hist_gb"]:
        m = build_model(name).fit(tr.drop(columns="label"), tr["label"])
        s = m.predict_score(te.drop(columns="label"))
        accs[name] = float(((s > 0.5) == te["label"].to_numpy()).mean())
        print(f"  {name:9s} test accuracy {accs[name]:.3f}  vs majority-class {base:.3f}")
    assert all(a > base for a in accs.values()), "every model beats the majority class"
    assert max(accs.values()) > base + 0.03, "the best model has a clear edge"


def dates_after(dts, cut, n):
    u = dts.unique()
    return u[list(u).index(cut) + n]


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm not installed (runs on your machine)")
def test_lightgbm_model(xy):
    banner("MODELS: LightGBM through the same interface")
    X, y = xy
    split = int(len(X) * 0.7)
    m = build_model("lightgbm", n_estimators=50).fit(X.iloc[:split], y.iloc[:split])
    s = m.predict_score(X.iloc[split:])
    print(f"  score range [{s.min():.3f}, {s.max():.3f}], top features {list(m.feature_importance().index[:3])}")
    assert ((s >= 0) & (s <= 1)).all()
