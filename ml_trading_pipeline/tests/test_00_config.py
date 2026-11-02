"""Step 0 - configuration: defaults load, overrides merge, derived values are filled."""
from pathlib import Path

import yaml

from conftest import banner
from trading_ml.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, deep_merge, load_config


def test_default_config_loads():
    banner("CONFIG: default.yaml loads with every section")
    cfg = load_config()
    for section in ["project", "data", "features", "labels", "validation", "models",
                    "line_a", "line_b", "backtest", "tracking", "paper"]:
        assert section in cfg, section
        print(f"  section '{section}': {len(cfg[section])} keys")
    print(f"  models configured: {list(cfg['models'])}")
    assert cfg["validation"]["purge_days"] == cfg["labels"]["horizon"], "purge defaults to horizon"
    print(f"  purge_days derived from horizon = {cfg['validation']['purge_days']}")


def test_deep_merge_keeps_untouched_keys():
    banner("CONFIG: deep merge only replaces what the override names")
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 20}})
    print(f"  base={base}\n  override={{'a': {{'y': 20}}}}\n  merged={out}")
    assert out == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base["a"]["y"] == 2, "base must not be mutated"


def test_real_data_config_merges_on_defaults():
    banner("CONFIG: real_data.yaml inherits everything it does not override")
    cfg = load_config(PROJECT_ROOT / "configs" / "real_data.yaml")
    print(f"  source={cfg['data']['source']}  tickers={len(cfg['data']['tickers'])}  "
          f"horizon(inherited)={cfg['labels']['horizon']}  test_days={cfg['validation']['test_days']}")
    assert cfg["data"]["source"] == "yfinance"
    assert cfg["labels"]["horizon"] == 5            # inherited
    assert cfg["validation"]["test_days"] == 252     # overridden
    assert cfg["validation"]["min_train_days"] == 756


def test_models_section_is_replaced_not_merged():
    banner("CONFIG: listing `models:` in an override means exactly those models")
    cfg = load_config(overrides={"models": {"logistic": {"C": 1.0}}})
    print(f"  models after override: {list(cfg['models'])}")
    assert list(cfg["models"]) == ["logistic"]


def test_override_file(tmp_path: Path):
    banner("CONFIG: a user YAML file overrides defaults")
    p = tmp_path / "mine.yaml"
    p.write_text(yaml.safe_dump({"labels": {"horizon": 10}}))
    cfg = load_config(p)
    print(f"  horizon={cfg['labels']['horizon']}  purge_days={cfg['validation']['purge_days']}")
    assert cfg["labels"]["horizon"] == 10
    assert cfg["validation"]["purge_days"] == 10
    assert DEFAULT_CONFIG_PATH.exists()
