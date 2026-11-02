"""Step 9 - the command-line workflow exactly as you will run it: train -> paper trade -> compare."""
import json

import pandas as pd
import pytest
import yaml

from conftest import banner
from trading_ml.cli import compare_main, paper_main, train_main


@pytest.fixture(scope="module")
def cfg_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    cfg = {
        "project": {"output_dir": str(tmp / "outputs")},
        "data": {"cache_dir": str(tmp / "data"), "synthetic": {"n_tickers": 5, "n_days": 650}},
        "validation": {"n_splits": 2, "test_days": 63, "min_train_days": 200, "holdout_days": 15},
        "models": {"always_long": {}, "logistic": {"C": 0.1}, "hist_gb": {"max_iter": 50}},
        "line_b": {"top_k": 2},
        "tracking": {"backend": "local", "local_dir": str(tmp / "runs")},
        "paper": {"state_dir": str(tmp / "paper_state")},
    }
    p = tmp / "test_config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, tmp


def test_cli_train_paper_compare(cfg_file):
    banner("CLI: train.py -> paper_trade_daily.py --simulate -> compare_models.py")
    p, tmp = cfg_file
    session = train_main(["--config", str(p), "--session", "CLI"])
    assert session == "CLI"
    lb = paper_main(["--config", str(p), "--simulate"])
    tl = compare_main(["--config", str(p)])
    out = sorted(f.name for f in (tmp / "outputs").iterdir())
    ledgers = sorted(f.name for f in (tmp / "paper_state").iterdir())
    print(f"\n  outputs: {out}\n  ledgers: {ledgers}")
    assert {"training_summary_CLI.csv", "paper_leaderboard.csv", "paper_equity.png",
            "backtest_leaderboard_CLI.csv"} <= set(out)
    assert len(lb) == 3 * 3 + 3 and (lb["days"] == 15).all()
    assert len(tl) == 3 * 3 and set(tl["beats_bench"]) <= {"True", "False"}
    assert (tl["stage"] == "champion").sum() == 3


def test_cli_daily_rerun_is_noop(cfg_file, capsys):
    banner("CLI: running the daily job again on the same data does nothing (safe to run twice)")
    p, tmp = cfg_file
    before = json.loads((tmp / "paper_state" / "B__universe__logistic.json").read_text())
    paper_main(["--config", str(p)])
    after = json.loads((tmp / "paper_state" / "B__universe__logistic.json").read_text())
    out = capsys.readouterr().out
    assert "nothing new to process" in out
    assert before["history"] == after["history"]
    print("  ledger unchanged, message 'nothing new to process' printed -> OK")


def test_cli_fresh_live_start(cfg_file):
    banner("CLI: fresh live start - decide on the latest day only, first fills at the next open")
    p, tmp = cfg_file
    lb = paper_main(["--config", str(p), "--reset"])
    rep = pd.read_csv(tmp / "paper_state" / "daily_report.csv")
    print(rep[["date", "strategy", "equity", "action_for_tomorrow", "target_weights"]].head(6).to_string(index=False))
    assert rep["date"].nunique() == 1
    assert (lb["equity"] == 100_000).all(), "nothing filled yet on day one"
    pending = json.loads((tmp / "paper_state" / "B__universe__equal_weight.json").read_text())["pending"]
    assert pending is not None and len(pending["weights"]) == 5
