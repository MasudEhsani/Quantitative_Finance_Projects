"""Step 5 - signal rules and the backtest engine, checked against hand calculations."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.backtest import max_drawdown, performance_metrics, run_backtest
from trading_ml.data import generate_synthetic_prices, to_wide
from trading_ml.signals import (equal_weight, line_a_weights, line_b_weights, scores_to_signals,
                                signals_to_positions)


def _prices_from_opens(opens: dict) -> pd.DataFrame:
    n = len(next(iter(opens.values())))
    dates = pd.bdate_range("2024-01-01", periods=n, name="date")
    rows = []
    for t, ops in opens.items():
        for d, o in zip(dates, ops):
            rows.append({"date": d, "ticker": t, "open": o, "high": o * 1.01, "low": o * 0.99,
                         "close": o, "volume": 1.0})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index().astype(float)


# ----------------------------------------------------------------------------- signals
def test_threshold_signals_and_stateful_positions():
    banner("SIGNALS: buy > 0.55, sell < 0.45, hold in between keeps the previous position")
    idx = pd.bdate_range("2024-01-01", periods=7, name="date")
    scores = pd.Series([0.5, 0.6, 0.5, 0.52, 0.4, 0.5, 0.7], index=idx)
    sig = scores_to_signals(scores, 0.55, 0.45)
    pos_long = signals_to_positions(sig, allow_short=False)
    pos_short = signals_to_positions(sig, allow_short=True)
    print(pd.DataFrame({"score": scores, "signal": sig, "pos_long_only": pos_long, "pos_long_short": pos_short}))
    assert list(sig) == [0, 1, 0, 0, -1, 0, 1]
    assert list(pos_long) == [0, 1, 1, 1, 0, 0, 1]
    assert list(pos_short) == [0, 1, 1, 1, -1, -1, 1]
    with pytest.raises(ValueError):
        scores_to_signals(scores, 0.4, 0.6)


def test_line_a_weights_split_capital():
    banner("SIGNALS: Line A with 2 tickers -> each gets half the capital when long")
    idx = pd.MultiIndex.from_product([pd.bdate_range("2024-01-01", periods=3, name="date"), ["A", "B"]],
                                     names=["date", "ticker"])
    scores = pd.Series([0.6, 0.3, 0.5, 0.7, 0.2, 0.5], index=idx)
    w = line_a_weights(scores)
    print(w)
    assert list(w["A"]) == [0.5, 0.5, 0.0]
    assert list(w["B"]) == [0.0, 0.5, 0.5]


def test_line_b_top_k_and_rebalance():
    banner("SIGNALS: Line B longs the top-k, holds between rebalances, long/short is dollar-neutral")
    dates = pd.bdate_range("2024-01-01", periods=4, name="date")
    tick = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    vals = [0.9, 0.8, 0.1, 0.2,   0.1, 0.2, 0.9, 0.8,   0.5, 0.5, 0.5, 0.5,   0.1, 0.9, 0.8, 0.2]
    scores = pd.Series(vals, index=idx)
    w = line_b_weights(scores, top_k=2, rebalance_every=2)
    print("long-only, top_k=2, rebalance every 2 days:\n", w)
    assert list(w.iloc[0]) == [0.5, 0.5, 0, 0]
    assert list(w.iloc[1]) == [0.5, 0.5, 0, 0]        # held, ignores day-2 scores
    assert list(w.iloc[2]) == [0.5, 0.5, 0, 0]        # ties broken by ticker order
    assert np.allclose(w.sum(axis=1), 1.0)
    ls = line_b_weights(scores, top_k=1, bottom_k=1, rebalance_every=1)
    print("long/short, top_k=1, bottom_k=1:\n", ls)
    assert np.allclose(ls.sum(axis=1), 0.0) and np.allclose(ls.abs().sum(axis=1), 1.0)
    assert ls.loc[dates[0], "A"] == 0.5 and ls.loc[dates[0], "C"] == -0.5


# ----------------------------------------------------------------------------- backtest
def test_backtest_hand_calculation():
    banner("BACKTEST: decide at close t, trade at open t+1, pay costs on turnover")
    prices = _prices_from_opens({"X": [100, 110, 99, 99, 108.9, 100]})
    dates = prices.index.get_level_values("date").unique()
    target = pd.DataFrame({"X": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=dates[:5])
    res = run_backtest(target, prices, cost_bps=10)
    table = pd.concat([target["X"].rename("target"), res.held_weights["X"].rename("held"),
                       res.gross_returns, res.turnover, res.costs, res.returns], axis=1)
    print(table)
    #  day0: held 0                      -> 0
    #  day1: held 1, open 110 -> 99      -> -10%, buy at open: turnover 1 -> cost 0.1%
    #  day2: held 1, open 99 -> 99       -> 0
    #  day3: held 0, sold at open        -> 0, turnover 1 -> cost 0.1%
    #  day4: held 0                      -> 0
    #  day5: no next open -> dropped
    assert list(res.held_weights["X"]) == [0, 1, 1, 0, 0]
    assert np.allclose(res.gross_returns.values, [0, -0.1, 0, 0, 0])
    assert np.allclose(res.costs.values, [0, 0.001, 0, 0.001, 0])
    assert np.allclose(res.returns.values, [0, -0.101, 0, -0.001, 0])
    assert res.equity.iloc[-1] == pytest.approx(100_000 * 0.899 * 0.999)
    print(f"  final equity {res.equity.iloc[-1]:,.2f} == 100,000 x 0.899 x 0.999 -> OK")


def test_buy_and_hold_matches_price_ratio():
    banner("BACKTEST: buy & hold without costs earns exactly open[last]/open[first trade] - 1")
    p = generate_synthetic_prices(n_tickers=1, n_days=300, seed=11)
    dates = p.index.get_level_values("date").unique()
    w = equal_weight(dates[:-1], ["SYN00"])
    res = run_backtest(w, p, cost_bps=0)
    opens = to_wide(p, "open")["SYN00"]
    expected = opens.iloc[-1] / opens.iloc[1] - 1
    print(f"  backtest total return {res.metrics['total_return']:+.6f}   price ratio {expected:+.6f}")
    assert res.metrics["total_return"] == pytest.approx(expected, rel=1e-9)


def test_no_same_bar_cheating():
    banner("BACKTEST: knowing today's open-to-open move gives NO edge (it is traded one day later)")
    p = generate_synthetic_prices(n_tickers=3, n_days=1000, momentum_strength=0.0, seed=5)
    opens = to_wide(p, "open")
    r_oo_today = opens.shift(-1) / opens - 1
    cheat = (r_oo_today > 0).astype(float) / 3      # "perfect" signal for the CURRENT bar
    res = run_backtest(cheat.iloc[:-1], p, cost_bps=0)
    same_bar = (cheat.iloc[:-1] * r_oo_today.iloc[:-1]).sum(axis=1)   # what a buggy engine would report
    s_bug = same_bar.mean() / same_bar.std() * np.sqrt(252)
    print(f"  engine Sharpe (correct timing): {res.metrics['sharpe']:+.2f}")
    print(f"  Sharpe if traded on the same bar (bug): {s_bug:+.2f}")
    assert abs(res.metrics["sharpe"]) < 1.0
    assert s_bug > 5


def test_costs_scale_with_turnover():
    banner("BACKTEST: higher costs reduce returns exactly by turnover x cost")
    p = generate_synthetic_prices(n_tickers=2, n_days=200, seed=9)
    dates = p.index.get_level_values("date").unique()[:-1]
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.integers(0, 2, (len(dates), 2)) / 2, index=dates, columns=["SYN00", "SYN01"])
    r0, r10 = run_backtest(w, p, cost_bps=0), run_backtest(w, p, cost_bps=10)
    diff = (r0.returns - r10.returns)
    print(f"  total turnover {r10.turnover.sum():.1f}   sum of return differences {diff.sum():.5f}  "
          f"(= turnover x 0.001 = {r10.turnover.sum() * 0.001:.5f})")
    assert np.allclose(diff, r10.turnover * 0.001)
    with pytest.raises(ValueError):
        run_backtest(w.rename(columns={"SYN00": "NOPE"}), p)


def test_metrics_known_values():
    banner("METRICS: Sharpe, drawdown and hit rate on a hand-made return series")
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    m = performance_metrics(r)
    exp_sharpe = r.mean() / r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    exp_dd = (eq / eq.cummax() - 1).min()
    for k in ["total_return", "sharpe", "max_drawdown", "hit_rate"]:
        print(f"  {k:13s} {m[k]:+.6f}")
    assert m["sharpe"] == pytest.approx(exp_sharpe)
    assert m["max_drawdown"] == pytest.approx(exp_dd) == pytest.approx(max_drawdown(r))
    assert m["total_return"] == pytest.approx(eq.iloc[-1] - 1)
    assert m["hit_rate"] == pytest.approx(2 / 4)       # zero-return day ignored
    assert performance_metrics(pd.Series([0.0, 0.0]))["sharpe"] == 0.0
