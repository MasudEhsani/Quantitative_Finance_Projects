"""A strict, in-memory stand-in for the parts of the MLflow API this project uses.

It lets the MLflowTracker code path be tested where mlflow is not installed.
It is strict on purpose: wrong argument names, logging outside a run, nested
runs or changing a param value raise, like real MLflow would.
On a machine with mlflow installed, test_06 also runs against the real thing.
"""
from __future__ import annotations

import shutil
import tempfile
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


class _Store:
    def __init__(self):
        self.uri = None
        self.experiments = {}     # name -> dict(id, artifact_location)
        self.runs = {}            # run_id -> dict
        self.active = None
        self.calls = []


def make_fake_mlflow():
    st = _Store()
    m = types.ModuleType("mlflow")
    m._store = st

    def set_tracking_uri(uri):
        st.calls.append("set_tracking_uri")
        st.uri = uri

    def get_experiment_by_name(name):
        e = st.experiments.get(name)
        return None if e is None else types.SimpleNamespace(experiment_id=e["id"], name=name)

    def create_experiment(name, artifact_location=None, tags=None):
        if name in st.experiments:
            raise ValueError("experiment exists")
        eid = str(len(st.experiments) + 1)
        st.experiments[name] = {"id": eid, "artifact_location": artifact_location}
        return eid

    def _require_run():
        if st.active is None:
            raise RuntimeError("no active run")
        return st.runs[st.active]

    @contextmanager
    def start_run(run_id=None, experiment_id=None, run_name=None, nested=False, tags=None, description=None):
        if st.active is not None and not nested:
            raise RuntimeError("run already active")
        if run_id is not None:                          # resume an existing run
            if run_id not in st.runs:
                raise KeyError(f"run {run_id} not found")
            st.active = run_id
            try:
                yield types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id))
            finally:
                st.active = None
            return
        exp = next(e for e in st.experiments.values() if e["id"] == experiment_id)
        rid = uuid.uuid4().hex
        art = Path(urlparse(exp["artifact_location"]).path) / rid / "artifacts"
        st.runs[rid] = {"experiment_id": experiment_id, "params": {}, "metrics": {}, "history": [],
                        "tags": {"mlflow.runName": run_name, **(tags or {})}, "artifact_dir": art,
                        "start_time": pd.Timestamp.now()}
        st.active = rid
        try:
            yield types.SimpleNamespace(info=types.SimpleNamespace(run_id=rid))
        finally:
            st.active = None

    def log_params(params):
        run = _require_run()
        for k, v in params.items():
            if not isinstance(v, str):
                raise TypeError("tracker must stringify params")
            if k in run["params"] and run["params"][k] != v:
                raise ValueError(f"param {k} changed value")
            run["params"][k] = v

    def log_metrics(metrics, step=None):
        run = _require_run()
        for k, v in metrics.items():
            if not isinstance(v, float):
                raise TypeError(f"metric {k} must be float")
            run["metrics"][k] = v
            run["history"].append((k, v, step or 0))

    def set_tags(tags):
        _require_run()["tags"].update({k: str(v) for k, v in tags.items()})

    def log_artifact(local_path, artifact_path=None):
        run = _require_run()
        dest = run["artifact_dir"] / (artifact_path or "")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dest)

    def search_runs(experiment_ids=None, filter_string=""):
        conds = []
        for part in [p for p in filter_string.split(" and ") if p.strip()]:
            k, v = part.split(" = ")
            assert k.startswith("tags."), k
            conds.append((k[5:], v.strip("'")))
        rows = []
        for rid, r in st.runs.items():
            if r["experiment_id"] not in experiment_ids:
                continue
            if any(r["tags"].get(k) != v for k, v in conds):
                continue
            row = {"run_id": rid, "start_time": r["start_time"]}
            row.update({f"params.{k}": v for k, v in r["params"].items()})
            row.update({f"metrics.{k}": v for k, v in r["metrics"].items()})
            row.update({f"tags.{k}": v for k, v in r["tags"].items()})
            rows.append(row)
        return pd.DataFrame(rows)

    class MlflowClient:
        def __init__(self, tracking_uri=None):
            self.uri = tracking_uri

        def set_tag(self, run_id, key, value):
            st.runs[run_id]["tags"][key] = str(value)

        def get_metric_history(self, run_id, key):
            return [types.SimpleNamespace(key=k, value=v, step=s) for k, v, s in st.runs[run_id]["history"] if k == key]

    def download_artifacts(artifact_uri=None, run_id=None, artifact_path=None, dst_path=None, tracking_uri=None):
        src = st.runs[run_id]["artifact_dir"] / artifact_path
        if not src.exists():
            raise FileNotFoundError(src)
        tmp = Path(tempfile.mkdtemp()) / src.name
        shutil.copy(src, tmp)
        return str(tmp)

    m.set_tracking_uri = set_tracking_uri
    m.get_experiment_by_name = get_experiment_by_name
    m.create_experiment = create_experiment
    m.start_run = start_run
    m.log_params = log_params
    m.log_metrics = log_metrics
    m.set_tags = set_tags
    m.log_artifact = log_artifact
    m.search_runs = search_runs
    m.tracking = types.SimpleNamespace(MlflowClient=MlflowClient)
    m.artifacts = types.SimpleNamespace(download_artifacts=download_artifacts)
    return m
