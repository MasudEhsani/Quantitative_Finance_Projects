"""Comparison reports: training leaderboard (backtests) and paper-trading leaderboard + plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .tracking import get_tracker  # noqa: E402

TRAIN_COLS = {
    "tags.line": "line", "tags.target": "target", "tags.model": "model", "tags.stage": "stage",
    "tags.beats_benchmark": "beats_bench",
    "metrics.ml_accuracy": "accuracy", "metrics.ml_auc": "auc", "metrics.ml_ic_mean": "ic",
    "metrics.bt_total_return": "bt_return", "metrics.bt_sharpe": "bt_sharpe",
    "metrics.bt_max_drawdown": "bt_max_dd", "metrics.bench_total_return": "bench_return",
    "metrics.excess_return": "excess_return",
}


def training_leaderboard(cfg: Dict[str, Any], session_id: str) -> pd.DataFrame:
    frames = []
    for line in ("A", "B"):
        runs = get_tracker(cfg, f"line{line}", verbose=False).search_runs({"session_id": session_id})
        if len(runs):
            frames.append(runs[[c for c in TRAIN_COLS if c in runs.columns]].rename(columns=TRAIN_COLS))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["line", "target", "bt_sharpe"], ascending=[True, True, False]).reset_index(drop=True)


def plot_paper_equity(curves: pd.DataFrame, path: Path, title: str = "Paper trading - equity by strategy") -> Path:
    """One panel per (line, target); every strategy normalised to 1.0 at its first day."""
    groups: Dict[str, list] = {}
    for col in curves.columns:
        line, target, _ = col.split("|")
        groups.setdefault(f"Line {line} - {target}", []).append(col)
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)
    for ax, (name, cols) in zip(axes[:, 0], sorted(groups.items())):
        for c in sorted(cols):
            s = curves[c].dropna()
            is_bench = c.endswith(("buy_and_hold", "equal_weight"))
            ax.plot(s.index, s / s.iloc[0], label=c.split("|")[2], lw=2.2 if is_bench else 1.3,
                    ls="--" if is_bench else "-", color="black" if is_bench else None)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.set_title(name)
        ax.set_ylabel("equity (start = 1)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def compare_paper_to_backtest(paper_lb: pd.DataFrame, train_lb: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Side by side: backtest Sharpe vs paper Sharpe per model (big gaps hint at overfitting)."""
    if paper_lb.empty or train_lb.empty:
        return None
    m = paper_lb.merge(train_lb[["line", "target", "model", "bt_sharpe", "bt_return"]],
                       on=["line", "target", "model"], how="left")
    return m[["strategy", "days", "total_return", "sharpe", "bt_return", "bt_sharpe"]].rename(
        columns={"total_return": "paper_return", "sharpe": "paper_sharpe"})
