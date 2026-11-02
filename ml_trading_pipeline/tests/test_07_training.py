"""Step 7 - the training pipeline: dataset, walk-forward scores, final model, full session with tracking."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.config import load_config
from trading_ml.data import generate_synthetic_prices
from trading_ml.models import available_models
from trading_ml.pipeline import (classification_metrics, fit_final_model, prepare_dataset, run_training,
                                 walk_forward_scores)
from trading_ml.tracking import LocalTracker
from trading_ml.validation import WalkForwardSplitter


@pytest.fixture()
def tcfg(cfg, tmp_path):
    """Test config writing every output (runs, plots, csv) into a temporary folder."""
    return load_config(overrides={**{k: cfg[k] for k in ("data", "validation", "models", "line_b")},
                                  "tracking": {"backend": "local", "local_dir": str(tmp_path / "runs")},
                                  "project": {"output_dir": str(tmp_path / "outputs")}})


def test_prepare_dataset(prices, cfg):
    banner("PIPELINE: datasets for Line A and Line B")
    for line in ["A", "B"]:
        X, y, fwd = prepare_dataset(prices, cfg, line)
        print(f"  Line {line}: X {X.shape}  labelled {int(y.notna().sum())}  unlabelled tail {int(y.isna().sum())}"
              f"  cs_rank features: {sum(c.startswith('cs_rank') for c in X.columns)}")
        assert not X.isna().any().any()
        assert X.index.equals(y.index) and X.index.equals(fwd.index)
        assert int(y.isna().sum()) == 5 * 5          # horizon x tickers
    assert any(c.startswith("cs_rank") for c in prepare_dataset(prices, cfg, "B")[0].columns)


def test_walk_forward_scores_are_out_of_sample(prices, cfg):
    banner("PIPELINE: walk-forward scores exist only for test blocks, one per row")
    X, y, fwd = prepare_dataset(prices, cfg, "A")
    sp = WalkForwardSplitter.from_config(cfg["validation"])
    scores, folds = walk_forward_scores("logistic", {"C": 0.1}, X, y, fwd, sp)
    print(folds[["fold", "train_start", "train_end", "test_start", "test_end", "n_train", "accuracy", "auc",
                 "ic_mean"]].round(3).to_string(index=False))
    test_dates = set().union(*[set(f.test_dates) for f in sp.split(X.index.get_level_values("date").unique())])
    assert set(scores.index.get_level_values("date")) == test_dates
    assert not scores.index.duplicated().any()
    assert scores.between(0, 1).all()
    for _, f in folds.iterrows():
        assert pd.Timestamp(f["train_end"]) < pd.Timestamp(f["test_start"])


def test_random_labels_give_no_edge(prices, cfg):
    banner("PIPELINE: with random labels the out-of-sample accuracy is ~50% (no hidden leakage)")
    X, y, fwd = prepare_dataset(prices, cfg, "A")
    rng = np.random.default_rng(0)
    y_rand = pd.Series(rng.integers(0, 2, len(y)).astype(float), index=y.index).where(y.notna())
    sp = WalkForwardSplitter.from_config(cfg["validation"])
    for name in ["logistic", "hist_gb"]:
        s, _ = walk_forward_scores(name, {}, X, y_rand, fwd, sp)
        acc = classification_metrics(y_rand.reindex(s.index), s)["accuracy"]
        print(f"  {name:9s} accuracy on random labels: {acc:.3f}")
        assert abs(acc - 0.5) < 0.05


def test_pipeline_skill_grows_with_planted_signal(cfg):
    banner("PIPELINE: dose-response - more planted momentum -> higher out-of-sample AUC and IC")
    out = {}
    for mom in [0.0, 0.06]:
        p = generate_synthetic_prices(n_tickers=6, n_days=900, momentum_strength=mom, seed=42)
        X, y, fwd = prepare_dataset(p, cfg, "B")
        sp = WalkForwardSplitter(n_splits=3, test_days=126, min_train_days=252, purge_days=5)
        s, _ = walk_forward_scores("logistic", {}, X, y, fwd, sp)
        out[mom] = classification_metrics(y.reindex(s.index), s, fwd.reindex(s.index))
        print(f"  momentum={mom:.2f}: accuracy={out[mom]['accuracy']:.3f} auc={out[mom]['auc']:.3f} "
              f"ic={out[mom]['ic_mean']:+.3f}")
    assert out[0.06]["auc"] > out[0.0]["auc"] + 0.03
    assert out[0.06]["ic_mean"] > 0.08
    assert abs(out[0.0]["auc"] - 0.5) < 0.05


def test_final_model_never_sees_holdout(prices, cfg):
    banner("PIPELINE: the final (paper-trading) model is trained only before hold-out minus purge")
    X, y, _ = prepare_dataset(prices, cfg, "B")
    sp = WalkForwardSplitter.from_config(cfg["validation"])
    m = fit_final_model("logistic", {}, X, y, sp)
    hold = sp.holdout_dates(X.index.get_level_values("date").unique())
    all_dates = X.index.get_level_values("date").unique()
    gap = all_dates.get_loc(hold.min()) - all_dates.get_loc(m.train_end_) - 1
    print(f"  trained {m.train_start_.date()}..{m.train_end_.date()} | hold-out starts {hold.min().date()} "
          f"| gap {gap} days (purge={sp.purge_days})")
    assert m.train_end_ < hold.min()
    assert gap == sp.purge_days


def test_full_training_session(prices, tcfg, tmp_path):
    banner("PIPELINE: full session - both lines, all models, backtests, tracking, champion tags")
    session, summary, results = run_training(tcfg, prices, lines=("A", "B"), session_id="TEST")
    n_models = len(available_models(list(tcfg["models"]), verbose=False))
    assert len(summary) == n_models * (2 + 1)          # 2 Line-A tickers + 1 universe
    assert summary["run_id"].notna().all()
    # same out-of-sample window for every model of a target -> fair comparison
    for (line, target), g in summary.groupby(["line", "target"]):
        assert g["bench_return"].nunique() == 1, f"{line}/{target}: benchmark must be identical across models"
    # always_long in Line A IS buy & hold: equal to its benchmark
    al = summary[(summary.line == "A") & (summary.model == "always_long")]
    print("\n  always_long vs benchmark (Line A):")
    print(al[["target", "total_return", "bench_return"]].to_string(index=False))
    assert np.allclose(al["total_return"], al["bench_return"], atol=1e-9)

    tracker = LocalTracker("trading-lineA", root=str(tmp_path / "runs"))
    runs = tracker.search_runs({"session_id": "TEST"})
    champs = runs[runs["tags.stage"] == "champion"]
    print(f"\n  Line A runs logged: {len(runs)}   champions: {list(champs['run_name'])}")
    assert len(runs) == n_models * 2 and len(champs) == 2
    rid = runs["run_id"].iloc[0]
    arts = sorted(p.name for p in (tmp_path / "runs" / "trading-lineA" / rid / "artifacts").rglob("*") if p.is_file())
    print(f"  artifacts of one run: {arts}")
    for f in ["equity_curve.png", "fold_metrics.csv", "oos_predictions.csv", "model.pkl", "config.yaml"]:
        assert f in arts
    assert (tmp_path / "outputs" / "training_summary_TEST.csv").exists()
    model = tracker.load_model(rid)
    assert hasattr(model, "train_end_")
