"""Step 1 - data: synthetic generator, schema validation, yfinance/CSV conversion, cache."""
import numpy as np
import pandas as pd
import pytest

from conftest import banner
from trading_ml.config import load_config
from trading_ml.data import (DataValidationError, generate_synthetic_prices, load_csv, load_prices,
                             summarize, to_wide, validate_ohlcv, yfinance_to_long)


def test_synthetic_prices_shape_and_schema(prices):
    banner("DATA: synthetic generator produces a valid long OHLCV frame")
    print(prices.head(6))
    print("\nper-ticker summary:")
    print(summarize(prices))
    assert list(prices.index.names) == ["date", "ticker"]
    assert list(prices.columns) == ["open", "high", "low", "close", "volume"]
    assert prices.index.get_level_values("ticker").nunique() == 5
    assert prices.index.get_level_values("date").nunique() == 700
    validate_ohlcv(prices)  # does not raise


def test_synthetic_is_deterministic():
    banner("DATA: same seed -> identical data; different seed -> different data")
    a = generate_synthetic_prices(n_tickers=2, n_days=50, seed=1)
    b = generate_synthetic_prices(n_tickers=2, n_days=50, seed=1)
    c = generate_synthetic_prices(n_tickers=2, n_days=50, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"].values, c["close"].values)
    print("  seed=1 twice: identical  |  seed=2: different  -> OK")


def test_synthetic_momentum_is_detectable():
    banner("DATA: the planted momentum makes 5d past return predict next-day return")
    strong = generate_synthetic_prices(n_tickers=4, n_days=1500, momentum_strength=0.05, seed=3)
    none_ = generate_synthetic_prices(n_tickers=4, n_days=1500, momentum_strength=0.0, seed=3)

    def corr(df):
        c = to_wide(df, "close")
        r = np.log(c).diff()
        past5 = r.rolling(5).sum()
        nxt = r.shift(-1)
        return float(pd.concat([past5.stack(), nxt.stack()], axis=1).dropna().corr().iloc[0, 1])

    c_strong, c_none = corr(strong), corr(none_)
    print(f"  corr(past 5d return, next-day return): momentum=0.05 -> {c_strong:+.3f}   "
          f"momentum=0 -> {c_none:+.3f}")
    assert c_strong > c_none + 0.03


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d.assign(high=d["low"] * 0.5), "inconsistent bars"),
        (lambda d: d.assign(close=d["close"].where(d.index != d.index[3], np.nan)), "NaN close"),
        (lambda d: d.assign(volume=-1.0), "volume"),
        (lambda d: d.iloc[::-1], "sorted"),
        (lambda d: pd.concat([d, d.iloc[:1]]), "duplicated"),
        (lambda d: d.drop(columns=["volume"]), "missing columns"),
        (lambda d: d.reset_index(), "MultiIndex"),
    ],
    ids=["high<low", "nan_close", "neg_volume", "unsorted", "duplicate", "missing_col", "flat_index"],
)
def test_validation_catches_bad_data(prices, mutation, message):
    bad = mutation(prices.copy())
    with pytest.raises(DataValidationError) as exc:
        validate_ohlcv(bad)
    print(f"  caught as expected -> {exc.value}")
    assert message in str(exc.value)


def _fake_yf_frame(layout: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=3), name="Date")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    base = {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Volume": 1000.0}
    if layout == "single":
        return pd.DataFrame({f: [base[f]] * 3 for f in fields}, index=idx)
    tickers = ["AAA", "BBB"]
    if layout == "ticker_first":
        cols = pd.MultiIndex.from_product([tickers, fields])
    else:  # field_first
        cols = pd.MultiIndex.from_product([fields, tickers])
    data = np.array([[base[c[1] if layout == "ticker_first" else c[0]] for c in cols]] * 3)
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.mark.parametrize("layout", ["single", "ticker_first", "field_first"])
def test_yfinance_conversion_all_layouts(layout):
    banner(f"DATA: yfinance result layout '{layout}' converts to long format (offline test)")
    raw = _fake_yf_frame(layout)
    tickers = ["AAA"] if layout == "single" else ["AAA", "BBB"]
    out = validate_ohlcv(yfinance_to_long(raw, tickers))
    print(out)
    assert sorted(out.index.get_level_values("ticker").unique()) == tickers
    assert len(out) == 3 * len(tickers)
    assert (out["close"] == 10.5).all()


def test_csv_roundtrip(tmp_path, prices):
    banner("DATA: CSV loader reads the documented long format")
    p = tmp_path / "prices.csv"
    prices.head(40).reset_index().to_csv(p, index=False)
    print(f"  wrote {p.name}:\n{p.read_text().splitlines()[0]}\n{p.read_text().splitlines()[1]}")
    back = load_csv(str(p))
    pd.testing.assert_frame_equal(back, prices.head(40), check_freq=False)
    print("  read back identical -> OK")


def test_load_prices_uses_cache(tmp_path):
    banner("DATA: load_prices() writes a cache on first call and reads it on the second")
    cfg = load_config(overrides={"data": {"cache_dir": str(tmp_path),
                                          "synthetic": {"n_tickers": 2, "n_days": 60}}})
    first = load_prices(cfg)
    files = list(tmp_path.iterdir())
    print(f"  cache files: {[f.name for f in files]}")
    assert len(files) == 1
    second = load_prices(cfg)
    pd.testing.assert_frame_equal(first, second, check_freq=False)
