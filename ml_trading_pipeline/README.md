# ML Trading Pipeline

A complete, tested backbone for machine-learning trading research in Python:

```
data ─► features ─► labels ─► walk-forward training ─► signals ─► backtest (with costs)
                                        │                                │
                                        └──────── MLflow tracking ◄──────┘
                                                        │
                                  daily paper trading of every model + benchmarks ─► comparison
```

Two lines share all of the code:

| Line | Question | Label | Signal → portfolio |
|---|---|---|---|
| **A – single stock** | Will this stock go up over the next 5 days? | `direction` (or `three_class`) | score > 0.55 buy, < 0.45 sell, otherwise hold |
| **B – set of stocks** | Which stocks will beat the others? | `cs_outperform` (beat the daily median) | long the top-k scores, rebalance every 5 days |

Everything runs **offline out of the box** on synthetic market data (with a weak, known
momentum effect planted in it), so you can see the whole pipeline work before using real data.

---

## 1. Install

```bash
cd ~/Quantitative_Finance/ml_trading_pipeline
conda create -n trading python=3.11 -y && conda activate trading     # or any Python >= 3.9
pip install -r requirements.txt
```

`requirements.txt` includes MLflow, LightGBM, yfinance and pyarrow. The core (numpy, pandas,
scikit-learn, matplotlib, pyyaml, pytest) is enough to run every test and the demo. Without
MLflow, runs go to a local `runs/` folder, and without LightGBM that model is skipped.

## 2. Run the tests (with all print-outs)

```bash
./run_tests.sh                       # whole suite, output also saved to reports/test_report.txt
./run_tests.sh tests/test_05_signals_backtest.py    # one step only
python -m pytest tests -q            # quiet, just pass/fail
```

The tests run in order of the pipeline and every one prints what it checks.
`reports/test_report_reference.txt` is the output of the run where this backbone was built
(71 passed, 2 skipped: the LightGBM and real-MLflow tests were skipped there and run on your
machine once those libraries are installed).

| File | What is validated |
|---|---|
| `test_00_config` | defaults load; your YAML overrides merge; `models:` is replaced, not merged |
| `test_01_data` | synthetic data valid and reproducible; 7 kinds of bad data rejected; yfinance result in all 3 layouts converted offline; CSV loader; cache |
| `test_02_features` | values equal hand calculations; **look-ahead tests**: features up to T are identical with/without data after T, and when all later prices are ×10 |
| `test_03_labels` | forward returns aligned per ticker; exactly `horizon` unlabelled rows at the end; label kinds |
| `test_04_splits_models` | walk-forward folds never overlap; purge removes label leakage (shown with purge 0 vs 5); hold-out never used; every model fits/predicts/pickles; models beat a coin flip on a strong planted signal |
| `test_05_signals_backtest` | signal rules; Line A/B weights; **backtest equals a hand calculation to the cent**; buy & hold equals the price ratio; costs = turnover × bps; a "perfect" same-bar signal gets no edge (Sharpe +0.4 instead of the +15 a buggy engine would show) |
| `test_06_tracking` | local run store; MLflow code path (strict test double); real MLflow when installed; runs can be resumed |
| `test_07_training` | random labels → ~50% accuracy (no hidden leakage); **dose-response** (more planted signal → higher AUC/IC); the final model never sees the hold-out; full session logged with champions and artifacts |
| `test_08_paper` | broker fills next open, reserves costs, no leverage, ledger saved atomically; **paper broker equals the backtest to 1e-10** with the same weights; multi-model daily replay; idempotent re-runs; in-sample warning |
| `test_09_cli_end_to_end` | the exact command-line workflow below, plus safe re-runs and a fresh live start |

## 3. Use it

### Offline demo (synthetic data, ~1 minute)

```bash
python scripts/run_demo.py
```

This trains every model on both lines, replays the 60-day hold-out as daily paper trading and
prints all leaderboards. `reports/demo_output_reference.txt` and the two example PNGs in `reports/`
show what you should see.

### Step by step

```bash
# 1. walk-forward training + backtest of all models, everything logged to MLflow
python scripts/train.py                          # --lines A  --models logistic hist_gb  --session my-name

# 2a. sanity check: replay the hold-out period day by day (fake trading on unseen data)
python scripts/paper_trade_daily.py --simulate

# 2b. live fake trading: run once per trading day AFTER the market close
python scripts/paper_trade_daily.py --reset      # first time only: decide today, fills tomorrow
python scripts/paper_trade_daily.py              # every following day (catches up missed days)

# 3. compare models: backtest leaderboard, paper leaderboard, paper vs backtest
python scripts/compare_models.py
```

### Real data (yfinance)

```bash
python scripts/train.py             --config configs/real_data.yaml
python scripts/paper_trade_daily.py --config configs/real_data.yaml --simulate   # check on the hold-out
python scripts/paper_trade_daily.py --config configs/real_data.yaml --reset      # start live paper trading
```

Edit `configs/real_data.yaml` for your tickers (for example `SAP.DE`, `^GDAXI`, `SPY`). You only
need the keys that differ from `configs/default.yaml`.

### Run the daily job automatically (Linux cron, 22:30 Berlin time, Mon–Fri)

```bash
crontab -e
30 22 * * 1-5  cd ~/Quantitative_Finance/ml_trading_pipeline && ~/anaconda3/envs/trading/bin/python scripts/paper_trade_daily.py --config configs/real_data.yaml >> paper_state/cron.log 2>&1
```

Retrain monthly with `scripts/train.py`. The daily job always trades the **latest** session.
Each strategy keeps its ledger, and the switch to the new model is recorded in the ledger.

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db      # from the project folder, then open http://127.0.0.1:5000
```

| Experiment | One run = | What is logged |
|---|---|---|
| `trading-lineA` | model × ticker | params, ML metrics (`ml_*`), backtest (`bt_*`), benchmark (`bench_*`), per-fold metrics as steps, equity plot, predictions, feature importance, config, model |
| `trading-lineB` | model × universe | same |
| `trading-paper` | paper strategy | `equity`, `daily_return`, `total_return`, `exposure` with one step per trading day |

Useful tags for filtering are `session_id`, `stage = champion` (best ML model per target by
backtest Sharpe) and `beats_benchmark`.

## 4. Project structure

```
ml_trading_pipeline/
├── configs/default.yaml          every setting, with comments (synthetic, offline)
├── configs/real_data.yaml        only the overrides for yfinance data
├── scripts/                      train.py, paper_trade_daily.py, compare_models.py, run_demo.py
├── src/trading_ml/
│   ├── config.py                 load + merge YAML
│   ├── data/                     schema + validation, synthetic generator, yfinance / CSV loaders, cache
│   ├── features/technical.py     returns, trend, RSI, Bollinger, volatility, ATR, MACD, volume, relative, cross-sectional ranks
│   ├── labels/targets.py         forward returns, direction / three_class / cs_outperform
│   ├── validation/splits.py      walk-forward splitter with purge + hold-out
│   ├── models/                   common interface, baselines, logistic, random forest, hist GB, LightGBM, registry
│   ├── signals/rules.py          scores → target weights (Line A thresholds, Line B top-k)
│   ├── backtest/                 vectorised engine (next-open execution, costs) + metrics
│   ├── tracking/tracker.py       MLflow tracker + local fallback with the same interface
│   ├── pipeline/training.py      walk-forward → backtest → log, for every model and line
│   ├── paper/                    paper broker (JSON ledger) + daily multi-strategy runner
│   ├── reporting.py              leaderboards, paper vs backtest, equity plots
│   └── cli.py                    command-line entry points
├── tests/                        test_00 … test_09 (+ fake_mlflow.py test double)
└── reports/                      reference test output, demo output, example plots
```

Generated folders (ignored by git): `data/` cache, `mlflow.db` + `mlartifacts/` or `runs/`,
`outputs/` (summaries, plots), `paper_state/` (ledgers + `daily_report.csv`).

## 5. Rules the code enforces

1. **Decide at the close of t, trade at the open of t+1**, in both the backtest and the paper broker.
2. Features at t use only data up to t (truncation and future-poison tests).
3. Purge `horizon` days between train and test, and before the hold-out, because labels look forward.
4. Scalers and all other fitting happen inside the training fold (sklearn pipelines).
5. Costs are paid on every trade (default 10 bps of traded value).
6. Every model is compared with a benchmark on exactly the same dates.

## 6. Extending it

* **Add a model:** subclass `SklearnModel` (or `BaseModel`) in `models/ml_models.py`, register
  the name in `models/registry.py`, and add `name: {params}` under `models:` in your config.
  Training, backtest, MLflow and paper trading pick it up automatically.
* **Add a feature:** add a column in `features/technical.py`. The look-ahead tests cover it automatically.
* **Next steps from the plan:** Optuna tuning (nested MLflow runs), LSTM / TFT models behind the
  same interface, regime features (HMM), sentiment, Alpaca paper API, a Streamlit dashboard.

## 7. Known limitations (on purpose, for v1)

* The default data is synthetic. Results on it say whether the pipeline works, not whether a strategy makes money.
* Universe from today's tickers means survivorship bias on real data.
* The backtest rebalances daily to target weights, while the paper broker holds shares (weights drift).
  With daily rebalancing and no costs they are identical (tested).
* Fractional shares, no market-impact model, no shorting costs, and the US-market calendar comes from the data.
* Run the daily job after the close. During market hours yfinance returns an incomplete bar for today.

This is a research and learning project, not financial advice.
