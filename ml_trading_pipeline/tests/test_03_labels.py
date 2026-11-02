"""Step 3 - labels: forward returns are aligned correctly and the last h rows have no label."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.labels import forward_return, make_labels


def _toy():
    dates = pd.bdate_range("2024-01-01", periods=8, name="date")
    close = {"AAA": [10, 11, 12, 11, 10, 12, 13, 12], "BBB": [20, 19, 18, 19, 21, 20, 22, 23]}
    rows = []
    for t, cs in close.items():
        for d, c in zip(dates, cs):
            rows.append({"date": d, "ticker": t, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index().astype(float)


def test_forward_return_alignment():
    banner("LABELS: forward_return(t) = close(t+h)/close(t) - 1, per ticker")
    p = _toy()
    fwd = forward_return(p, horizon=2)
    table = pd.concat([p["close"], fwd], axis=1).unstack("ticker")
    print(table.round(4))
    aaa = fwd.xs("AAA", level="ticker")
    assert aaa.iloc[0] == pytest.approx(12 / 10 - 1)       # day0 -> day2
    assert aaa.iloc[3] == pytest.approx(12 / 11 - 1)       # day3 -> day5
    assert aaa.iloc[-2:].isna().all()                      # no future -> NaN
    bbb = fwd.xs("BBB", level="ticker")
    assert bbb.iloc[0] == pytest.approx(18 / 20 - 1)       # must not mix tickers
    print("  checked: values, NaN tail, no mixing between tickers -> OK")


def test_direction_and_three_class():
    banner("LABELS: direction (0/1) and three_class (-1/0/1 with dead zone)")
    p = _toy()
    d = make_labels(p, 1, "direction")
    tc = make_labels(p, 1, "three_class", dead_zone=0.085)
    fwd = forward_return(p, 1)
    out = pd.concat([fwd.rename("fwd_1"), d.rename("direction"), tc.rename("three_class")], axis=1)
    print(out.xs("AAA", level="ticker").round(3))
    valid = fwd.notna()
    assert set(d[valid].unique()) <= {0.0, 1.0}
    assert (d[valid] == (fwd[valid] > 0).astype(float)).all()
    assert d[~valid].isna().all()
    assert (tc[fwd > 0.085] == 1).all() and (tc[fwd < -0.085] == -1).all()
    assert (tc[(fwd.abs() <= 0.085)] == 0).all()


def test_cs_outperform_is_balanced(prices):
    banner("LABELS: cs_outperform marks the better half of the universe each day")
    y = make_labels(prices, 5, "cs_outperform").dropna()
    share = y.groupby(level="date").mean()
    print(f"  share of 1s per date: mean={share.mean():.3f} min={share.min():.3f} max={share.max():.3f}")
    print(f"  (with 5 tickers, 2 of 5 are above the median -> 0.4 each day)")
    assert np.allclose(share, 0.4)


def test_label_tail_is_nan(prices):
    banner("LABELS: exactly `horizon` unlabeled rows at the end of every ticker")
    h = 5
    y = make_labels(prices, h, "direction")
    tail_nans = y.isna().groupby(level="ticker").sum()
    print(tail_nans.to_string())
    assert (tail_nans == h).all()
    with pytest.raises(ValueError):
        make_labels(prices, h, "nonsense")
