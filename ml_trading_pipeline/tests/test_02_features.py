"""Step 2 - features: correct values on hand-made data and NO look-ahead (leakage) tests."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.features import add_cross_sectional_ranks, build_features


@pytest.fixture(scope="module")
def feats(prices, cfg):
    return build_features(prices, cfg["features"])


def test_features_overview(feats, prices, cfg):
    banner("FEATURES: one row per (date, ticker), list of columns and warm-up NaNs")
    assert feats.index.equals(prices.index)
    nan_counts = feats.isna().groupby(level="ticker").sum().iloc[0]
    print(f"  {feats.shape[1]} features for {len(feats)} rows")
    print("  warm-up NaN rows per ticker (comes from the longest look-back window):")
    print(nan_counts.to_string())
    print("\n  describe() after warm-up:")
    print(feats.dropna().describe().T[["mean", "std", "min", "max"]].round(3))
    longest = max(cfg['features']['return_windows'] + cfg['features']['vol_windows'])
    assert nan_counts.max() <= longest
    assert not feats.dropna().empty


def test_feature_values_on_handmade_series(cfg):
    banner("FEATURES: values match hand calculations on a tiny series")
    dates = pd.bdate_range("2024-01-01", periods=80, name="date")
    close = pd.Series(np.linspace(100, 179, 80), index=dates)  # rises by exactly 1 per day
    df = pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
                       "volume": 1000.0})
    df["ticker"] = "TEST"
    prices = df.reset_index().set_index(["date", "ticker"])
    f = build_features(prices, cfg["features"]).droplevel("ticker")
    t = dates[30]
    exp_ret5 = np.log(close[t] / close[dates[25]])
    exp_sma10 = close[t] / close[dates[21]:t].mean() - 1
    print(f"  ret_5 on {t.date()}:        got {f.loc[t, 'ret_5']:.6f}  expected {exp_ret5:.6f}")
    print(f"  sma_ratio_10 on {t.date()}: got {f.loc[t, 'sma_ratio_10']:.6f}  expected {exp_sma10:.6f}")
    print(f"  rsi_14 of a series that only rises: {f.loc[t, 'rsi_14']:.3f} (expected 1.0 = RSI 100)")
    print(f"  hl_range: got {f.loc[t, 'hl_range']:.6f} expected {2 / close[t]:.6f}")
    assert f.loc[t, "ret_5"] == pytest.approx(exp_ret5)
    assert f.loc[t, "sma_ratio_10"] == pytest.approx(exp_sma10)
    assert f.loc[t, "rsi_14"] == pytest.approx(1.0)
    assert f.loc[t, "hl_range"] == pytest.approx(2 / close[t])
    assert f.loc[t, "day_of_week"] == t.dayofweek


def _cut_date(prices):
    dates = prices.index.get_level_values("date").unique()
    return dates[len(dates) // 2]


def test_no_lookahead_truncation(prices, cfg):
    banner("LEAKAGE: features up to T are identical with or without data after T")
    T = _cut_date(prices)
    full = add_cross_sectional_ranks(build_features(prices, cfg["features"]))
    trunc_prices = prices[prices.index.get_level_values("date") <= T]
    trunc = add_cross_sectional_ranks(build_features(trunc_prices, cfg["features"]))
    a = full[full.index.get_level_values("date") <= T]
    diff = (a - trunc).abs().max().max()
    print(f"  cut date T = {T.date()}, compared {a.shape[0]} rows x {a.shape[1]} columns")
    print(f"  max |difference| = {diff}")
    pd.testing.assert_frame_equal(a, trunc, check_exact=False, rtol=1e-10, atol=1e-12)


def test_no_lookahead_future_poison(prices, cfg):
    banner("LEAKAGE: multiplying all prices after T by 10 does not change features up to T")
    T = _cut_date(prices)
    poisoned = prices.copy()
    after = poisoned.index.get_level_values("date") > T
    poisoned.loc[after, ["open", "high", "low", "close"]] *= 10
    poisoned.loc[after, "volume"] *= 50
    f1 = build_features(prices, cfg["features"])
    f2 = build_features(poisoned, cfg["features"])
    before = f1.index.get_level_values("date") <= T
    pd.testing.assert_frame_equal(f1[before], f2[before])
    changed_after = (f1[~before] - f2[~before]).abs().sum().sum()
    print(f"  features <= T identical; total change after T = {changed_after:.1f} (must be > 0)")
    assert changed_after > 0


def test_cross_sectional_ranks(feats):
    banner("FEATURES: cross-sectional ranks are in (0, 1] and computed per date")
    ranked = add_cross_sectional_ranks(feats, ["ret_5", "vol_21"])
    day = ranked.dropna().index.get_level_values("date")[0]
    print(ranked.loc[day, ["ret_5", "cs_rank_ret_5", "vol_21", "cs_rank_vol_21"]].round(4))
    r = ranked["cs_rank_ret_5"].dropna()
    assert r.between(0, 1, inclusive="right").all()
    one_day = ranked.loc[day]
    assert one_day["ret_5"].idxmax() == one_day["cs_rank_ret_5"].idxmax()
    assert one_day["cs_rank_ret_5"].max() == 1.0
