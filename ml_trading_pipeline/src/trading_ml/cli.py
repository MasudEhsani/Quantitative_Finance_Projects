"""Command-line entry points (the scripts/ folder just calls these).

    python scripts/train.py            [--config C] [--lines A B] [--models logistic hist_gb] [--session NAME]
    python scripts/paper_trade_daily.py [--config C] [--session NAME] [--simulate] [--reset]
    python scripts/compare_models.py   [--config C] [--session NAME]
    python scripts/run_demo.py         [--config C]      # train -> simulate paper trading -> compare
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import pandas as pd

from .config import load_config, resolve_path
from .data import load_prices, summarize
from .paper import PaperTrader, load_session_strategies
from .pipeline import run_training
from .reporting import compare_paper_to_backtest, plot_paper_equity, training_leaderboard
from .validation import WalkForwardSplitter

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 30)


def _parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="YAML file merged on top of configs/default.yaml")
    return p


def _header(text: str) -> None:
    print("\n" + "#" * 90 + f"\n# {text}\n" + "#" * 90)


# ----------------------------------------------------------------------------- train
def train_main(argv: Optional[List[str]] = None) -> str:
    p = _parser("Walk-forward training + backtest of all configured models, logged to MLflow")
    p.add_argument("--lines", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--models", nargs="+", default=None, help="subset of the models in the config")
    p.add_argument("--session", default=None, help="session id (default: timestamp)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    _header("1/3  DATA")
    prices = load_prices(cfg)
    print(summarize(prices))
    _header("2/3  WALK-FORWARD TRAINING + BACKTEST")
    session, summary, _ = run_training(cfg, prices, lines=a.lines, models=a.models, session_id=a.session)
    _header("3/3  DONE")
    print(f"session id: {session}")
    print("champions (best ML model by backtest Sharpe per target) are tagged stage=champion in the tracker")
    print("next: python scripts/paper_trade_daily.py --simulate   (replay the hold-out period)")
    return session


# ----------------------------------------------------------------------------- paper
def paper_main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    p = _parser("Daily paper trading of every model of a training session (+ benchmarks)")
    p.add_argument("--session", default=None, help="training session to trade (default: latest)")
    p.add_argument("--simulate", action="store_true",
                   help="reset ledgers and replay the hold-out period day by day (offline demo / sanity check)")
    p.add_argument("--reset", action="store_true", help="delete existing paper ledgers first")
    p.add_argument("--start", default=None, help="first date to process (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="last date to process (YYYY-MM-DD)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    prices = load_prices(cfg)
    session, strategies = load_session_strategies(cfg, a.session)
    trader = PaperTrader(cfg, strategies)
    dates = prices.index.get_level_values("date").unique().sort_values()
    start = a.start
    if a.simulate or a.reset:
        trader.reset()
    if a.simulate and start is None:
        hold = WalkForwardSplitter.from_config(cfg["validation"]).holdout_dates(dates)
        if len(hold) == 0:
            raise SystemExit("validation.holdout_days is 0 - nothing to simulate")
        start = hold.min()
        print(f"[paper] SIMULATION: replaying hold-out {hold.min().date()}..{hold.max().date()} ({len(hold)} days)")
    elif start is None and all(b.last_date is None for b in trader.brokers.values()):
        start = dates[-1]          # fresh live start: decide today, first fills at the next open
        print(f"[paper] new ledgers: first decisions on {start.date()}, orders fill at the next open")
    _header(f"PAPER TRADING  session={session}")
    report = trader.run(prices, start=start, end=a.end)
    if report.empty:
        print("[paper] nothing new to process (ledgers are up to date)")
    else:
        last = report[report["date"] == report["date"].max()]
        print(f"\n[paper] decisions taken at the close of {report['date'].max()} (fill at next open):")
        print(last[["strategy", "equity", "total_return", "score", "action_for_tomorrow", "target_weights"]]
              .round(4).to_string(index=False))
    lb = trader.leaderboard()
    print("\n[paper] LEADERBOARD")
    print(lb.round(4).to_string(index=False))
    out = resolve_path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out / "paper_leaderboard.csv", index=False)
    if not lb.empty:
        print(f"[paper] plot: {plot_paper_equity(trader.equity_curves(), out / 'paper_equity.png')}")
    return lb


# ----------------------------------------------------------------------------- compare
def compare_main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    p = _parser("Compare models: backtest leaderboard and paper-trading leaderboard")
    p.add_argument("--session", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    session, strategies = load_session_strategies(cfg, a.session, verbose=False)
    tl = training_leaderboard(cfg, session)
    _header(f"BACKTEST LEADERBOARD  session={session}  (out-of-sample, after costs)")
    print(tl.round(3).to_string(index=False))
    trader = PaperTrader(cfg, strategies, track=False, verbose=False)
    pl = trader.leaderboard()
    _header("PAPER-TRADING LEADERBOARD")
    print(pl.round(4).to_string(index=False) if len(pl) else "no paper trading yet")
    both = compare_paper_to_backtest(pl, tl)
    if both is not None:
        _header("PAPER vs BACKTEST  (a large drop from backtest to paper = overfitting warning)")
        print(both.round(3).to_string(index=False))
    out = resolve_path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tl.to_csv(out / f"backtest_leaderboard_{session}.csv", index=False)
    if len(pl):
        print(f"\nplot: {plot_paper_equity(trader.equity_curves(), out / 'paper_equity.png')}")
    return tl


# ----------------------------------------------------------------------------- demo
def demo_main(argv: Optional[List[str]] = None) -> None:
    p = _parser("End-to-end demo: train -> simulated paper trading over the hold-out -> compare")
    a = p.parse_args(argv)
    extra = ["--config", a.config] if a.config else []
    session = train_main(extra + ["--session", "demo-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")])
    paper_main(extra + ["--session", session, "--simulate"])
    compare_main(extra + ["--session", session])
