"""Step 8 - paper trading: broker ledger, equivalence with the backtest, daily multi-model runner."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.backtest import run_backtest
from trading_ml.config import load_config
from trading_ml.data import generate_synthetic_prices, to_wide
from trading_ml.features import add_cross_sectional_ranks, build_features
from trading_ml.paper import PaperBroker, PaperTrader, load_session_strategies
from trading_ml.pipeline import run_training
from trading_ml.validation import WalkForwardSplitter


# ----------------------------------------------------------------------------- broker
def test_broker_hand_calculation(tmp_path):
    banner("BROKER: order queued at close fills next open, costs reserved, marked at close")
    b = PaperBroker("demo", initial_cash=1000, cost_bps=10)
    d0 = b.process_day("2024-01-01", {"X": 100.0}, {"X": 100.0})
    b.queue_target_weights("2024-01-01", {"X": 1.0})
    d1 = b.process_day("2024-01-02", {"X": 100.0}, {"X": 110.0})
    sh = 1000 / 1.001 / 100
    print(f"  day0 equity {d0['equity']:.2f} | day1 shares {b.positions['X']:.6f} (expected {sh:.6f}) "
          f"cash {b.cash:.6f} equity {d1['equity']:.4f}")
    print(f"  trade: {b.trades[0]}")
    assert b.positions["X"] == pytest.approx(sh)
    assert b.cash == pytest.approx(0.0, abs=1e-9)          # never negative
    assert d1["equity"] == pytest.approx(sh * 110)
    assert d1["costs"] == pytest.approx(sh * 100 * 0.001)
    b.queue_target_weights("2024-01-02", {})               # sell everything
    d2 = b.process_day("2024-01-03", {"X": 120.0}, {"X": 125.0})
    print(f"  day2 after selling at 120: cash {b.cash:.4f} positions {b.positions} equity {d2['equity']:.4f}")
    assert b.positions == {} and d2["equity"] == pytest.approx(sh * 120 * 0.999)


def test_broker_guards_and_persistence(tmp_path):
    banner("BROKER: no same-day fills, no double processing, no leverage, JSON save/load")
    b = PaperBroker("demo", 1000, 10)
    b.queue_target_weights("2024-01-02", {"X": 0.5})
    b.process_day("2024-01-02", {"X": 10.0}, {"X": 10.0})
    assert b.positions == {} and b.pending is not None, "decision of today must NOT fill today"
    with pytest.raises(ValueError):
        b.process_day("2024-01-02", {"X": 10.0}, {"X": 10.0})
    with pytest.raises(ValueError):
        b.queue_target_weights("2024-01-02", {"X": 0.8, "Y": 0.5})
    b.process_day("2024-01-03", {"X": 10.0}, {"X": 11.0})
    p = tmp_path / "demo.json"
    b.save(p)
    b2 = PaperBroker.load(p)
    assert b2.to_dict() == b.to_dict()
    print(f"  all guards raise; ledger round-trips through {p.name} ({p.stat().st_size} bytes)")


def test_broker_matches_backtest_exactly():
    banner("BROKER vs BACKTEST: same weights, no costs -> identical equity path")
    p = generate_synthetic_prices(n_tickers=3, n_days=60, seed=21)
    dates = p.index.get_level_values("date").unique()
    rng = np.random.default_rng(1)
    raw = rng.random((len(dates) - 1, 3))
    W = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True) * rng.uniform(0.3, 1.0, (len(dates) - 1, 1)),
                     index=dates[:-1], columns=["SYN00", "SYN01", "SYN02"])
    bt = run_backtest(W, p, cost_bps=0, initial_capital=100_000)
    opens, closes = to_wide(p, "open"), to_wide(p, "close")
    b = PaperBroker("eq", 100_000, cost_bps=0)
    for d in dates:
        b.process_day(d, opens.loc[d].to_dict(), closes.loc[d].to_dict())
        if d in W.index:
            b.queue_target_weights(d, W.loc[d].to_dict())
    paper_open = b.history_frame()["equity_open"]
    # paper equity at the open of day k+1 == backtest equity at the end of day k
    a = paper_open.iloc[1:].to_numpy()
    e = bt.equity.to_numpy()[: len(a)]
    print(f"  compared {len(a)} days, max abs difference = {np.abs(a - e).max():.2e} EUR")
    assert np.allclose(a, e, rtol=1e-10)


# ----------------------------------------------------------------------------- runner
@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Train a small session once (local tracker in a temp folder) and reuse it."""
    tmp = tmp_path_factory.mktemp("paper")
    cfg = load_config(overrides={
        "data": {"synthetic": {"n_tickers": 5, "n_days": 700}},
        "validation": {"n_splits": 2, "test_days": 63, "min_train_days": 200, "holdout_days": 20},
        "models": {"momentum_rule": {"feature": "ret_21"}, "logistic": {"C": 0.1}},
        "line_b": {"top_k": 2, "rebalance_every": 5},
        "tracking": {"backend": "local", "local_dir": str(tmp / "runs")},
        "project": {"output_dir": str(tmp / "out")},
        "paper": {"state_dir": str(tmp / "paper_state")},
    })
    prices = generate_synthetic_prices(5, 700, seed=42)
    sid, _, _ = run_training(cfg, prices, session_id="PAPERTEST", verbose=False)
    return cfg, prices, sid, tmp


def test_load_session_strategies(session):
    banner("PAPER: strategies = every model of the latest session + benchmarks")
    cfg, _, sid, _ = session
    found, strategies = load_session_strategies(cfg)
    for s in strategies:
        te = getattr(s.model, "train_end_", None)
        print(f"  {s.name:28s} benchmark={s.is_benchmark!s:5s} trained_until={te.date() if te is not None else '-'}")
    assert found == sid
    assert len(strategies) == 2 * 3 + 3              # (2 tickers + universe) x 2 models + 3 benchmarks


def test_paper_features_match_full_history(session):
    banner("PAPER: features from the 400-day look-back window equal features from the full history")
    cfg, prices, _, _ = session
    _, strategies = load_session_strategies(cfg, verbose=False)
    pt = PaperTrader(cfg, strategies, track=False, verbose=False)
    d = prices.index.get_level_values("date").unique()[-1]
    win = pt._features_at(prices, d)["B"]
    full = add_cross_sectional_ranks(build_features(prices, cfg["features"])).xs(d, level="date", drop_level=False)
    diff = (win - full[win.columns]).abs().max().max()
    print(f"  date {d.date()}: max |window - full| = {diff:.2e}")
    assert diff < 1e-8


def test_paper_simulation_over_holdout(session):
    banner("PAPER: replay the hold-out period day by day for every strategy")
    cfg, prices, _, tmp = session
    _, strategies = load_session_strategies(cfg, verbose=False)
    pt = PaperTrader(cfg, strategies, verbose=True)
    pt.reset()
    sp = WalkForwardSplitter.from_config(cfg["validation"])
    hold = sp.holdout_dates(prices.index.get_level_values("date").unique())
    report = pt.run(prices, start=hold.min())
    lb = pt.leaderboard()
    print("\n  LEADERBOARD")
    print(lb.drop(columns=["line"]).round(4).to_string(index=False))
    print("\n  last day of the daily report (Line A, SYN00):")
    last = report[report["date"] == report["date"].max()]
    print(last[last["target"] == "SYN00"][["strategy", "equity", "score", "action_for_tomorrow",
                                           "target_weights"]].to_string(index=False))
    assert len(report) == len(hold) * len(strategies)
    assert (lb["days"] == len(hold)).all()
    assert (lb["equity"] > 0).all()
    bh = pt.brokers["A|SYN00|buy_and_hold"]
    assert len(bh.trades) == 1 and bh.trades[0]["date"] == str(hold[1].date()), "benchmark buys at day-2 open"
    for s in strategies:                              # every model day is after its training data
        te = getattr(s.model, "train_end_", None)
        if te is not None:
            assert pd.Timestamp(te) < hold.min()
    assert (tmp / "paper_state" / "daily_report.csv").exists()


def test_paper_is_idempotent_and_resumes(session):
    banner("PAPER: re-running processes nothing new; a new day processes only that day")
    cfg, prices, _, _ = session
    _, strategies = load_session_strategies(cfg, verbose=False)
    dates = prices.index.get_level_values("date").unique()
    pt = PaperTrader(cfg, strategies, verbose=False)
    pt.reset()
    pt.run(prices, start=dates[-10], end=dates[-2])
    pt2 = PaperTrader(cfg, strategies, verbose=False)            # fresh object, ledgers loaded from disk
    again = pt2.run(prices, start=dates[-10], end=dates[-2])
    new = pt2.run(prices, start=dates[-10])
    print(f"  second run: {len(again)} rows (expected 0); next run: {len(new)} rows for {new['date'].nunique()} new day")
    assert len(again) == 0
    assert new["date"].nunique() == 1 and len(new) == len(strategies)
    hist = pt2.tracker.metric_history(pt2.brokers["B|universe|logistic"].meta["paper_run_id"], "equity")
    print(f"  tracked equity points for B|universe|logistic: {len(hist)} (one per day)")
    assert len(hist) == 10


def test_in_sample_warning(session, capsys):
    cfg, prices, _, _ = session
    _, strategies = load_session_strategies(cfg, verbose=False)
    pt = PaperTrader(cfg, [s for s in strategies if s.name == "A|SYN00|logistic"], track=False, verbose=False)
    pt.reset()
    early = prices.index.get_level_values("date").unique()[300]
    pt.step(early, prices)
    out = capsys.readouterr().out
    banner("PAPER: trading a day inside a model's training period prints a warning")
    print("  " + [ln for ln in out.splitlines() if "WARNING" in ln][0])
    assert "inside the model's training period" in out
