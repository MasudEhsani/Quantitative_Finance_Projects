"""Training + backtest pipeline for both lines.

For every (line, ticker-or-universe, model):
  1. walk-forward: fit on each train window, score the next test block (out-of-sample)
  2. ML metrics per fold and overall (accuracy, AUC, information coefficient)
  3. scores -> target weights (Line A thresholds / Line B top-k) -> backtest with costs
  4. same backtest for the benchmark (buy & hold / equal weight) on the same dates
  5. fit a FINAL model on all non-hold-out data (used later for paper trading)
  6. log params, metrics, fold metrics, plots, predictions and the model to the tracker
All runs of one call share a ``session_id`` tag, so they can be compared and paper-traded together.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score  # noqa: E402

from ..backtest import BacktestResult, format_metrics, run_backtest  # noqa: E402
from ..config import resolve_path  # noqa: E402
from ..features import add_cross_sectional_ranks, build_features  # noqa: E402
from ..labels import forward_return, make_labels  # noqa: E402
from ..models import BASELINES, BaseModel, available_models, build_model  # noqa: E402
from ..signals import equal_weight, line_a_weights, line_b_weights  # noqa: E402
from ..tracking import BaseTracker, get_tracker  # noqa: E402
from ..validation import WalkForwardSplitter  # noqa: E402


# ----------------------------------------------------------------------------- dataset
def prepare_dataset(prices: pd.DataFrame, cfg: Dict[str, Any], line: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X: features (rows with complete features only), y: label (NaN where future unknown), fwd: forward return."""
    feats = build_features(prices, cfg["features"])
    if line == "B":
        feats = add_cross_sectional_ranks(feats)
    h = cfg["labels"]["horizon"]
    kind = cfg["labels"]["line_a_kind" if line == "A" else "line_b_kind"]
    y = make_labels(prices, h, kind, cfg["labels"].get("dead_zone", 0.0))
    fwd = forward_return(prices, h)
    X = feats.dropna()
    return X, y.reindex(X.index), fwd.reindex(X.index)


def line_a_tickers(cfg: Dict[str, Any], prices: pd.DataFrame) -> List[str]:
    universe = sorted(prices.index.get_level_values("ticker").unique())
    t = cfg["line_a"].get("tickers")
    if not t:
        return universe[:2]
    missing = [x for x in t if x not in universe]
    if missing:
        raise ValueError(f"line_a.tickers not in data: {missing}")
    return list(t)


# ----------------------------------------------------------------------------- metrics
def classification_metrics(y: pd.Series, score: pd.Series, fwd: Optional[pd.Series] = None) -> Dict[str, float]:
    """Metrics of 'score predicts up-move' on labelled rows. Works for every label kind."""
    mask = y.notna() & score.notna()
    y, score = y[mask], score[mask]
    if len(y) == 0:
        return {}
    up = (y > 0).astype(int).to_numpy()
    s = score.to_numpy()
    m = {"n_obs": float(len(y)), "accuracy": float(((s > 0.5).astype(int) == up).mean()),
         "up_rate": float(up.mean())}
    if len(np.unique(up)) == 2:
        m["balanced_accuracy"] = float(balanced_accuracy_score(up, (s > 0.5).astype(int)))
        m["auc"] = float(roc_auc_score(up, s)) if len(np.unique(s)) > 1 else 0.5
        m["log_loss"] = float(log_loss(up, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1]))
    if fwd is not None:
        f = fwd.reindex(score.index)
        df = pd.DataFrame({"s": score, "f": f}).dropna()
        n_tickers = df.index.get_level_values("ticker").nunique()
        if n_tickers >= 3:     # information coefficient: daily cross-sectional rank correlation
            ics = [g["s"].rank().corr(g["f"].rank()) for _, g in df.groupby(level="date")
                   if len(g) >= 3 and g["s"].nunique() > 1 and g["f"].nunique() > 1]
            m["ic_mean"] = float(np.mean(ics)) if ics else 0.0
        elif len(df) > 2 and df["s"].nunique() > 1:      # one ticker: rank correlation over time
            m["ic_mean"] = float(df["s"].rank().corr(df["f"].rank()))
        else:
            m["ic_mean"] = 0.0                            # constant score (e.g. always_long)
    return m


# ----------------------------------------------------------------------------- walk-forward
def walk_forward_scores(model_name: str, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                        fwd: pd.Series, splitter: WalkForwardSplitter, seed: int = 42
                        ) -> Tuple[pd.Series, pd.DataFrame]:
    dates = X.index.get_level_values("date")
    scores, rows = [], []
    for fold in splitter.split(dates.unique()):
        tr = dates.isin(fold.train_dates) & y.notna().to_numpy()
        te = dates.isin(fold.test_dates)
        model = build_model(model_name, seed=seed, **params).fit(X[tr], y[tr])
        s = pd.Series(model.predict_score(X[te]), index=X.index[te], name="score")
        scores.append(s)
        fm = classification_metrics(y[te], s, fwd[te])
        rows.append({"fold": fold.fold, "train_start": fold.train_dates.min().date(),
                     "train_end": fold.train_dates.max().date(), "test_start": fold.test_dates.min().date(),
                     "test_end": fold.test_dates.max().date(), "n_train": int(tr.sum()), **fm})
    return pd.concat(scores).sort_index(), pd.DataFrame(rows)


def fit_final_model(model_name: str, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                    splitter: WalkForwardSplitter, seed: int = 42) -> BaseModel:
    """Train on all data before the hold-out, minus a purge gap (labels must not see hold-out prices)."""
    usable = splitter.usable_dates(X.index.get_level_values("date").unique())
    train_dates = usable[: len(usable) - splitter.purge_days] if splitter.purge_days else usable
    mask = X.index.get_level_values("date").isin(train_dates) & y.notna().to_numpy()
    model = build_model(model_name, seed=seed, **params).fit(X[mask], y[mask])
    model.train_start_ = X.index.get_level_values("date")[mask].min()
    model.train_end_ = X.index.get_level_values("date")[mask].max()   # paper trading must start after this
    return model


def scores_to_weights(scores: pd.Series, cfg: Dict[str, Any], line: str) -> pd.DataFrame:
    if line == "A":
        a = cfg["line_a"]
        return line_a_weights(scores, a["buy_threshold"], a["sell_threshold"], a["allow_short"])
    b = cfg["line_b"]
    return line_b_weights(scores, b["top_k"], b["bottom_k"], b["rebalance_every"])


# ----------------------------------------------------------------------------- one run
@dataclass
class RunResult:
    line: str
    target: str               # ticker (Line A) or "universe" (Line B)
    model: str
    run_id: Optional[str]
    scores: pd.Series
    folds: pd.DataFrame
    ml_metrics: Dict[str, float]
    backtest: BacktestResult
    benchmark: BacktestResult
    seconds: float


def equity_figure(res: BacktestResult, bench: BacktestResult, title: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(res.equity.index, res.equity.values, label="strategy (after costs)", lw=1.4)
    ax1.plot(bench.equity.index, bench.equity.values, label="benchmark", lw=1.1, alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("equity")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    dd = res.equity / res.equity.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="tab:red")
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run_one(model_name: str, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series, fwd: pd.Series,
            prices: pd.DataFrame, cfg: Dict[str, Any], line: str, target: str,
            tracker: Optional[BaseTracker], session_id: str) -> RunResult:
    t0 = time.time()
    splitter = WalkForwardSplitter.from_config(cfg["validation"])
    seed = cfg["project"]["seed"]
    scores, folds = walk_forward_scores(model_name, params, X, y, fwd, splitter, seed)
    ml = classification_metrics(y.reindex(scores.index), scores, fwd.reindex(scores.index))

    tickers = sorted(scores.index.get_level_values("ticker").unique())
    weights = scores_to_weights(scores, cfg, line)
    bt_cfg = cfg["backtest"]
    bt = run_backtest(weights, prices, bt_cfg["cost_bps"], bt_cfg["initial_capital"])
    bench = run_backtest(equal_weight(weights.index, tickers), prices, bt_cfg["cost_bps"], bt_cfg["initial_capital"])
    final_model = fit_final_model(model_name, params, X, y, splitter, seed)
    run_id = None

    if tracker is not None:
        tags = {"session_id": session_id, "line": line, "target": target, "model": model_name,
                "is_baseline": str(model_name in BASELINES), "stage": "candidate",
                "beats_benchmark": str(bt.metrics["sharpe"] > bench.metrics["sharpe"])}
        with tracker.start_run(f"{line}-{target}-{model_name}", tags=tags):
            run_id = tracker.run_id
            tracker.log_params({
                "line": line, "target": target, "model": model_name, "model_params": params,
                "label_kind": cfg["labels"]["line_a_kind" if line == "A" else "line_b_kind"],
                "horizon": cfg["labels"]["horizon"], "n_features": X.shape[1], "tickers": ",".join(tickers),
                "validation": cfg["validation"], "signals": cfg["line_a"] if line == "A" else cfg["line_b"],
                "cost_bps": bt_cfg["cost_bps"], "data_source": cfg["data"]["source"],
                "oos_start": str(scores.index.get_level_values("date").min().date()),
                "oos_end": str(scores.index.get_level_values("date").max().date()),
            })
            tracker.log_metrics({f"ml_{k}": v for k, v in ml.items()})
            tracker.log_metrics({f"bt_{k}": v for k, v in bt.metrics.items()})
            tracker.log_metrics({f"bench_{k}": v for k, v in bench.metrics.items()})
            tracker.log_metrics({"excess_return": bt.metrics["total_return"] - bench.metrics["total_return"],
                                 "excess_sharpe": bt.metrics["sharpe"] - bench.metrics["sharpe"]})
            for _, row in folds.iterrows():                      # one point per fold -> line chart in the UI
                tracker.log_metrics({f"fold_{k}": row[k] for k in ("accuracy", "auc", "ic_mean") if k in row
                                     and pd.notna(row[k])}, step=int(row["fold"]))
            tracker.log_dataframe(folds, "fold_metrics.csv")
            tracker.log_dataframe(pd.DataFrame({"score": scores, "label": y.reindex(scores.index),
                                                "fwd_return": fwd.reindex(scores.index)}), "oos_predictions.csv")
            tracker.log_dataframe(pd.DataFrame({"strategy": bt.equity, "benchmark": bench.equity}), "equity.csv")
            imp = final_model.feature_importance()
            if imp is not None:
                tracker.log_dataframe(imp.to_frame("importance"), "feature_importance.csv")
            fig = equity_figure(bt, bench, f"Line {line} | {target} | {model_name}")
            tracker.log_figure(fig, "equity_curve.png")
            plt.close(fig)
            tracker.log_text(yaml.safe_dump(cfg, sort_keys=False), "config.yaml")
            tracker.log_model(final_model)

    return RunResult(line, target, model_name, run_id, scores, folds, ml, bt, bench, time.time() - t0)


# ----------------------------------------------------------------------------- whole session
def summary_row(r: RunResult) -> Dict[str, Any]:
    return {"line": r.line, "target": r.target, "model": r.model,
            "accuracy": r.ml_metrics.get("accuracy"), "auc": r.ml_metrics.get("auc"),
            "ic": r.ml_metrics.get("ic_mean"),
            "total_return": r.backtest.metrics["total_return"], "sharpe": r.backtest.metrics["sharpe"],
            "max_dd": r.backtest.metrics["max_drawdown"],
            "turnover": r.backtest.metrics.get("avg_daily_turnover"),
            "bench_return": r.benchmark.metrics["total_return"], "bench_sharpe": r.benchmark.metrics["sharpe"],
            "run_id": r.run_id, "sec": round(r.seconds, 1)}


def run_training(cfg: Dict[str, Any], prices: pd.DataFrame, lines: Sequence[str] = ("A", "B"),
                 models: Optional[Sequence[str]] = None, track: bool = True, session_id: Optional[str] = None,
                 verbose: bool = True) -> Tuple[str, pd.DataFrame, List[RunResult]]:
    session_id = session_id or pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    model_names = available_models(list(models or cfg["models"].keys()), verbose=verbose)
    results: List[RunResult] = []
    if verbose:
        print(f"\n[train] session={session_id} lines={list(lines)} models={model_names}")

    for line in lines:
        tracker = get_tracker(cfg, f"line{line}", verbose=verbose) if track else None
        X, y, fwd = prepare_dataset(prices, cfg, line)
        if verbose:
            print(f"[train] Line {line}: X={X.shape}, labelled rows={int(y.notna().sum())}, "
                  f"label balance={y.mean():.3f}")
        if line == "A":
            groups = [(t, X.index.get_level_values("ticker") == t) for t in line_a_tickers(cfg, prices)]
        else:
            groups = [("universe", np.ones(len(X), dtype=bool))]
        for target, mask in groups:
            Xg, yg, fg = X[mask], y[mask], fwd[mask]
            for name in model_names:
                r = run_one(name, cfg["models"].get(name) or {}, Xg, yg, fg, prices, cfg, line, target,
                            tracker, session_id)
                results.append(r)
                if verbose:
                    print(f"  [{line}|{target}|{name:13s}] acc={r.ml_metrics.get('accuracy', float('nan')):.3f} "
                          f"{format_metrics(r.backtest.metrics)}  ({r.seconds:.1f}s)")
        if tracker is not None:
            _tag_best(tracker, session_id, [r for r in results if r.line == line])

    summary = pd.DataFrame([summary_row(r) for r in results])
    out_dir = resolve_path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"training_summary_{session_id}.csv", index=False)
    if verbose:
        cols = ["line", "target", "model", "accuracy", "auc", "ic", "total_return", "sharpe", "max_dd",
                "bench_return", "bench_sharpe"]
        print("\n[train] SUMMARY (out-of-sample, after costs)")
        print(summary[cols].round(3).to_string(index=False))
        print(f"[train] saved {out_dir / f'training_summary_{session_id}.csv'}")
    return session_id, summary, results


def _tag_best(tracker: BaseTracker, session_id: str, results: List[RunResult]) -> None:
    """Tag the best non-baseline model (by backtest Sharpe) per target as stage=champion.
    Check the ``beats_benchmark`` tag too: the best ML model can still lose to buy & hold."""
    for target in sorted({r.target for r in results}):
        ml = [r for r in results if r.target == target and r.model not in BASELINES and r.run_id]
        if ml:
            best = max(ml, key=lambda r: r.backtest.metrics["sharpe"])
            tracker.set_tags({"stage": "champion"}, run_id=best.run_id)
