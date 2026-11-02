"""Experiment tracking: MLflow when installed, otherwise a local folder store
with the same interface (so the pipeline always runs).

    tracker = get_tracker(cfg, "lineA")
    with tracker.start_run("logistic-AAPL", tags={"line": "A"}):
        tracker.log_params({...}); tracker.log_metrics({...}); tracker.log_model(model)
    runs = tracker.search_runs({"session_id": "..."})     # DataFrame like mlflow.search_runs
    model = tracker.load_model(run_id)

Models are stored as a pickle artifact ``model/model.pkl`` - works the same in
both backends and for every model type in this project.
"""
from __future__ import annotations

import json
import pickle
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from ..config import resolve_path

MODEL_ARTIFACT = "model/model.pkl"


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Nested dict -> flat {'a.b': 'value'} with string values (MLflow params are strings)."""
    out: Dict[str, str] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:5000]
    return out


class BaseTracker:
    backend = "base"

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.run_id: Optional[str] = None

    # helpers shared by both backends ------------------------------------
    def log_dataframe(self, df: pd.DataFrame, filename: str, artifact_path: Optional[str] = None) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / filename
            df.to_csv(p)
            self.log_artifact(str(p), artifact_path)

    def log_text(self, text: str, filename: str, artifact_path: Optional[str] = None) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / filename
            p.write_text(text, encoding="utf-8")
            self.log_artifact(str(p), artifact_path)

    def log_figure(self, fig, filename: str, artifact_path: Optional[str] = "plots") -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / filename
            fig.savefig(p, dpi=110, bbox_inches="tight")
            self.log_artifact(str(p), artifact_path)

    def log_model(self, model) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pkl"
            with open(p, "wb") as fh:
                pickle.dump(model, fh)
            self.log_artifact(str(p), "model")

    # backend specific -----------------------------------------------------
    def start_run(self, run_name: str, tags: Optional[Dict[str, Any]] = None,
                  run_id: Optional[str] = None): raise NotImplementedError
    def log_params(self, params: Dict[str, Any]) -> None: raise NotImplementedError
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: raise NotImplementedError
    def set_tags(self, tags: Dict[str, Any], run_id: Optional[str] = None) -> None: raise NotImplementedError
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None: raise NotImplementedError
    def search_runs(self, tags: Optional[Dict[str, str]] = None) -> pd.DataFrame: raise NotImplementedError
    def load_model(self, run_id: str): raise NotImplementedError
    def metric_history(self, run_id: str, key: str) -> pd.Series: raise NotImplementedError


# =============================================================================== local
class LocalTracker(BaseTracker):
    """Folder layout: <root>/<experiment>/<run_id>/{meta.json, metrics.csv, artifacts/}"""
    backend = "local"

    def __init__(self, experiment: str, root: str = "runs"):
        super().__init__(experiment)
        self.exp_dir = resolve_path(root) / experiment
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: Optional[str] = None) -> Path:
        rid = run_id or self.run_id
        if rid is None:
            raise RuntimeError("no active run - use `with tracker.start_run(...)`")
        return self.exp_dir / rid

    def _meta(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        return json.loads((self._run_dir(run_id) / "meta.json").read_text())

    def _save_meta(self, meta: Dict[str, Any], run_id: Optional[str] = None) -> None:
        (self._run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    @contextmanager
    def start_run(self, run_name: str, tags: Optional[Dict[str, Any]] = None,
                  run_id: Optional[str] = None) -> Iterator["LocalTracker"]:
        """New run, or - with ``run_id`` - resume an existing one (used by daily paper trading)."""
        if run_id is not None and (self.exp_dir / run_id / "meta.json").exists():
            self.run_id = run_id
            meta = self._meta()
            meta["status"] = "RUNNING"
            meta["tags"].update(flatten(tags or {}))
            self._save_meta(meta)
        else:
            self.run_id = uuid.uuid4().hex
            d = self._run_dir()
            (d / "artifacts").mkdir(parents=True)
            self._save_meta({"run_id": self.run_id, "run_name": run_name, "status": "RUNNING",
                             "start_time": pd.Timestamp.now().isoformat(), "params": {}, "metrics": {},
                             "tags": flatten(tags or {})})
        try:
            yield self
            status = "FINISHED"
        except BaseException:
            status = "FAILED"
            raise
        finally:
            meta = self._meta()
            meta["status"] = status
            self._save_meta(meta)
            self.run_id = None

    def log_params(self, params):
        meta = self._meta()
        meta["params"].update(flatten(params))
        self._save_meta(meta)

    def log_metrics(self, metrics, step=None):
        meta = self._meta()
        rows = []
        for k, v in metrics.items():
            v = float(v)
            meta["metrics"][k] = v
            rows.append({"key": k, "value": v, "step": 0 if step is None else int(step), "timestamp": time.time()})
        self._save_meta(meta)
        p = self._run_dir() / "metrics.csv"
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)

    def set_tags(self, tags, run_id=None):
        meta = self._meta(run_id)
        meta["tags"].update(flatten(tags))
        self._save_meta(meta, run_id)

    def log_artifact(self, local_path, artifact_path=None):
        dest = self._run_dir() / "artifacts" / (artifact_path or "")
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(local_path)
        (dest / src.name).write_bytes(src.read_bytes())

    def search_runs(self, tags=None):
        rows = []
        for meta_file in self.exp_dir.glob("*/meta.json"):
            m = json.loads(meta_file.read_text())
            if tags and any(m["tags"].get(k) != str(v) for k, v in tags.items()):
                continue
            row = {"run_id": m["run_id"], "run_name": m["run_name"], "status": m["status"],
                   "start_time": pd.Timestamp(m["start_time"])}
            row.update({f"params.{k}": v for k, v in m["params"].items()})
            row.update({f"metrics.{k}": v for k, v in m["metrics"].items()})
            row.update({f"tags.{k}": v for k, v in m["tags"].items()})
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values("start_time", ascending=False).reset_index(drop=True) if len(df) else df

    def artifact_path(self, run_id: str, rel: str) -> Path:
        return self._run_dir(run_id) / "artifacts" / rel

    def load_model(self, run_id):
        with open(self.artifact_path(run_id, MODEL_ARTIFACT), "rb") as fh:
            return pickle.load(fh)

    def metric_history(self, run_id, key):
        df = pd.read_csv(self._run_dir(run_id) / "metrics.csv")
        df = df[df["key"] == key]
        return pd.Series(df["value"].to_numpy(), index=df["step"].to_numpy(), name=key)


# =============================================================================== mlflow
def _resolve_mlflow_uri(uri: str) -> str:
    """Make a relative sqlite path absolute (relative to the project root)."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        return prefix + str(resolve_path(uri[len(prefix):]))
    if "://" not in uri:                        # plain folder -> file store
        return str(resolve_path(uri))
    return uri


class MLflowTracker(BaseTracker):
    backend = "mlflow"

    def __init__(self, experiment: str, tracking_uri: str = "sqlite:///mlflow.db",
                 artifact_dir: str = "mlartifacts"):
        super().__init__(experiment)
        import mlflow
        self.mlflow = mlflow
        self.tracking_uri = _resolve_mlflow_uri(tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            location = (resolve_path(artifact_dir) / experiment).resolve()
            location.mkdir(parents=True, exist_ok=True)
            self.experiment_id = mlflow.create_experiment(experiment, artifact_location=location.as_uri())
        else:
            self.experiment_id = exp.experiment_id

    @contextmanager
    def start_run(self, run_name, tags=None, run_id=None):
        self.mlflow.set_tracking_uri(self.tracking_uri)
        if run_id is not None:       # resume (daily paper trading appends to the same run)
            ctx = self.mlflow.start_run(run_id=run_id)
        else:
            ctx = self.mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id,
                                        tags=flatten(tags or {}))
        with ctx as run:
            self.run_id = run.info.run_id
            try:
                yield self
            finally:
                self.run_id = None

    def log_params(self, params):
        self.mlflow.log_params(flatten(params))

    def log_metrics(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def set_tags(self, tags, run_id=None):
        if run_id is None:
            self.mlflow.set_tags(flatten(tags))
        else:
            client = self.mlflow.tracking.MlflowClient(self.tracking_uri)
            for k, v in flatten(tags).items():
                client.set_tag(run_id, k, v)

    def log_artifact(self, local_path, artifact_path=None):
        self.mlflow.log_artifact(local_path, artifact_path)

    def search_runs(self, tags=None):
        self.mlflow.set_tracking_uri(self.tracking_uri)
        flt = " and ".join(f"tags.{k} = '{v}'" for k, v in (tags or {}).items())
        df = self.mlflow.search_runs(experiment_ids=[self.experiment_id], filter_string=flt)
        if len(df):
            df = df.rename(columns={"tags.mlflow.runName": "run_name"})
        return df

    def load_model(self, run_id):
        local = self.mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=MODEL_ARTIFACT,
                                                        tracking_uri=self.tracking_uri)
        with open(local, "rb") as fh:
            return pickle.load(fh)

    def metric_history(self, run_id, key):
        client = self.mlflow.tracking.MlflowClient(self.tracking_uri)
        hist = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
        return pd.Series([m.value for m in hist], index=[m.step for m in hist], name=key)


# =============================================================================== factory
def mlflow_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("mlflow") is not None


def get_tracker(cfg: Dict[str, Any], experiment_suffix: str, verbose: bool = True) -> BaseTracker:
    t = cfg["tracking"]
    name = f"{t.get('experiment_prefix', 'trading')}-{experiment_suffix}"
    backend = t.get("backend", "auto")
    if backend == "auto":
        backend = "mlflow" if mlflow_available() else "local"
        if backend == "local" and verbose:
            print("[tracking] mlflow not installed -> using local run store "
                  f"'{t.get('local_dir', 'runs')}/' (pip install mlflow for the MLflow UI)")
    if backend == "mlflow":
        tracker: BaseTracker = MLflowTracker(name, t.get("mlflow_uri", "sqlite:///mlflow.db"),
                                             t.get("artifact_dir", "mlartifacts"))
    elif backend == "local":
        tracker = LocalTracker(name, t.get("local_dir", "runs"))
    else:
        raise ValueError(f"unknown tracking.backend '{backend}'")
    if verbose:
        print(f"[tracking] backend={tracker.backend} experiment={name}")
    return tracker
