"""Step 6 - experiment tracking: local store, MLflow code path (test double), real MLflow if installed."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import banner
from fake_mlflow import make_fake_mlflow
from trading_ml.config import load_config
from trading_ml.models import build_model
from trading_ml.tracking import LocalTracker, MLflowTracker, flatten, get_tracker


def _exercise(tracker, tmp_path):
    """Same scenario for every backend: 2 runs, params/metrics/steps/artifacts/model, search, reload."""
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) % 3})
    y = (X["a"] > 9).astype(float)
    model = build_model("logistic").fit(X, y)
    ids = []
    for i, name in enumerate(["run-logistic", "run-other"]):
        with tracker.start_run(name, tags={"session_id": "S1", "model": name}):
            ids.append(tracker.run_id)
            tracker.log_params({"model": name, "nested": {"C": 0.1, "list": [1, 2]}})
            tracker.log_metrics({"sharpe": 1.5 - i, "accuracy": 0.6})
            for step in range(3):
                tracker.log_metrics({"fold_accuracy": 0.5 + 0.1 * step}, step=step)
            tracker.log_dataframe(pd.DataFrame({"x": [1, 2]}), "table.csv")
            tracker.log_text("hello", "note.txt")
            tracker.log_model(model)
    with tracker.start_run("other-session", tags={"session_id": "S2"}):
        tracker.log_metrics({"sharpe": 9.0})
    tracker.set_tags({"stage": "champion"}, run_id=ids[0])
    with tracker.start_run("ignored", run_id=ids[0]):       # resume: append more history to run 0
        assert tracker.run_id == ids[0]
        tracker.log_metrics({"fold_accuracy": 0.8}, step=3)

    runs = tracker.search_runs({"session_id": "S1"})
    print(runs[["run_id", "run_name", "metrics.sharpe", "params.nested.C", "tags.model"]].to_string(index=False))
    assert len(runs) == 2
    assert set(runs["run_id"]) == set(ids)
    assert runs.set_index("run_id").loc[ids[0], "tags.stage"] == "champion"
    assert runs.set_index("run_id").loc[ids[0], "params.nested.C"] == "0.1"
    hist = tracker.metric_history(ids[0], "fold_accuracy")
    print(f"  fold_accuracy history: {hist.to_dict()}")
    assert list(hist.values) == pytest.approx([0.5, 0.6, 0.7, 0.8]) and list(hist.index) == [0, 1, 2, 3]
    back = tracker.load_model(ids[0])
    assert np.allclose(back.predict_score(X), model.predict_score(X))
    print("  model reloaded from the tracker predicts identically -> OK")
    return ids


def test_flatten():
    banner("TRACKING: nested config dicts are flattened to string params")
    flat = flatten({"a": 1, "b": {"c": [1, 2], "d": {"e": None}}})
    print(f"  {flat}")
    assert flat == {"a": "1", "b.c": "[1, 2]", "b.d.e": "None"}


def test_local_tracker(tmp_path):
    banner("TRACKING: local folder store (used when mlflow is not installed)")
    t = LocalTracker("unit-test", root=str(tmp_path))
    ids = _exercise(t, tmp_path)
    files = sorted(p.relative_to(tmp_path / "unit-test" / ids[0]).as_posix()
                   for p in (tmp_path / "unit-test" / ids[0]).rglob("*") if p.is_file())
    print(f"  files of one run: {files}")
    assert "artifacts/model/model.pkl" in files and "meta.json" in files


def test_local_tracker_marks_failed_runs(tmp_path):
    banner("TRACKING: a run that raises is stored with status FAILED")
    t = LocalTracker("unit-test", root=str(tmp_path))
    with pytest.raises(ZeroDivisionError):
        with t.start_run("boom"):
            1 / 0
    print(t.search_runs()[["run_name", "status"]])
    assert t.search_runs()["status"].iloc[0] == "FAILED"


def test_mlflow_tracker_with_test_double(tmp_path, monkeypatch):
    banner("TRACKING: MLflowTracker code path against a strict MLflow test double")
    fake = make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    t = MLflowTracker("unit-test", tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
                      artifact_dir=str(tmp_path / "art"))
    print(f"  tracking uri: {t.tracking_uri}")
    _exercise(t, tmp_path)
    t2 = MLflowTracker("unit-test", tracking_uri=t.tracking_uri, artifact_dir=str(tmp_path / "art"))
    assert t2.experiment_id == t.experiment_id, "re-opening an experiment reuses it"


def test_get_tracker_factory(tmp_path, monkeypatch):
    banner("TRACKING: backend 'auto' picks mlflow when importable, else the local store")
    cfg = load_config(overrides={"tracking": {"local_dir": str(tmp_path / "runs"),
                                              "mlflow_uri": "sqlite:///" + str(tmp_path / "m.db"),
                                              "artifact_dir": str(tmp_path / "art")}})
    t_local = get_tracker(load_config(overrides={"tracking": {"backend": "local",
                                                              "local_dir": str(tmp_path / "runs")}}), "x")
    assert t_local.backend == "local"
    monkeypatch.setitem(sys.modules, "mlflow", make_fake_mlflow())
    t_ml = get_tracker(load_config(overrides={"tracking": {**cfg["tracking"], "backend": "mlflow"}}), "x")
    assert t_ml.backend == "mlflow"
    with pytest.raises(ValueError):
        get_tracker(load_config(overrides={"tracking": {"backend": "nope"}}), "x")


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None,
                    reason="mlflow not installed (this test runs on your machine)")
def test_real_mlflow(tmp_path):
    banner("TRACKING: the REAL MLflow library (sqlite backend)")
    t = MLflowTracker("unit-test-real", tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
                      artifact_dir=str(tmp_path / "art"))
    _exercise(t, tmp_path)
