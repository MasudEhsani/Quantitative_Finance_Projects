# Quantitative Finance Projects

A collection of projects in quantitative finance: interest-rate and credit models on
binomial trees, portfolio optimisation and risk, PCA- and inverse-reinforcement-learning
trading strategies, a tested machine-learning trading pipeline, credit-default modelling,
and a SQL data-warehouse ETL.

Repository: <https://github.com/MasudEhsani/Quantitative_Finance_Projects>

## Projects

| Folder | Topic | Main tools | Status |
|---|---|---|---|
| [`ml_trading_pipeline/`](ml_trading_pipeline/README.md) | End-to-end ML trading research: features → labels → walk-forward training → signals → backtest with costs → daily paper trading, with MLflow tracking | Python package, scikit-learn, LightGBM, MLflow, pytest | Tested package with CLI scripts and a full test suite |
| [`IRL-market-portfolio/`](IRL-market-portfolio/README.md) | Econometric estimation of the inverse-reinforcement-learning market model (Halperin & Feldshteyn 2018) and a trading strategy built on it | numpy, pandas, scikit-learn | Tested package, notebook, results |
| [`pca-trading-strategy/`](pca-trading-strategy/README.md) | Eigen-portfolios and the Absorption Ratio (Kritzman et al. 2011) driving an equity / fixed-income switching strategy | numpy, pandas, scikit-learn | Tested package, notebook, results |
| [`portfolio_optimization/`](portfolio_optimization/README.md) | Markowitz, minimum-variance, Sharpe-optimal and two-fund portfolios; shrinkage estimates; VaR / CVaR | numpy, scipy, cvxpy | Stand-alone scripts |
| [`kredit-default-projekt/`](kredit-default-projekt/README.md) | Credit-default prediction on LendingClub loans: imbalance handling (class weights, SMOTE, undersampling), model comparison, feature importance, written report (German) | pandas, scikit-learn, seaborn | Scripts, saved models, figures, report |
| [`Binomial models/`](Binomial%20models/README.md) | Binomial short-rate trees: defaultable bond, Black-Derman-Toy calibration, swap and swaption; 15-period option-pricing spreadsheet | numpy, scipy, spreadsheet | Stand-alone scripts |
| [`Option Pricing/`](Option%20Pricing/README.md) | Black-Derman-Toy swaption pricer (same script as in `Binomial models/`) | numpy, scipy | Stand-alone script |
| [`finance-etl-project/`](finance-etl-project/README.md) | MySQL ETL: messy raw bank extracts → staging → validated star schema, with quarantine table, run log, data-quality checks | MySQL 8, Python | Runnable pipeline, with a pandas mirror for checking without a database |

## Highlights

- **ML trading pipeline**: look-ahead tests, purged walk-forward splits, a backtest verified
  against hand calculation, and a paper broker that matches the backtest to 1e-10.
- **PCA strategy**: the Absorption-Ratio strategy reaches a Sharpe ratio of 0.92 against
  0.62 for a static 50/50 portfolio (`pca-trading-strategy/results/performance_summary.csv`).
- **IRL market model**: parameter recovery on simulated data; the strategy's raw
  out-of-sample Sharpe (~4.5) is flagged as unrealistic and shown to vanish at 20–50 bps
  trading cost.
- **Credit default**: six model variants on a 500-loan sample; ROC-AUC 0.55–0.68.

## Getting started

Each project is self-contained; install its own requirements and follow its README.

```bash
git clone https://github.com/MasudEhsani/Quantitative_Finance_Projects.git
cd Quantitative_Finance_Projects/pca-trading-strategy
pip install -r requirements.txt
python main.py
```

| Project | Requirements | Run |
|---|---|---|
| `ml_trading_pipeline` | `requirements.txt` (core: numpy, pandas, scikit-learn, matplotlib, pyyaml, pytest) | `python scripts/run_demo.py`, `./run_tests.sh` |
| `IRL-market-portfolio` | `requirements.txt` | `python main.py` |
| `pca-trading-strategy` | `requirements.txt` | `python main.py` |
| `portfolio_optimization` | numpy, scipy, pandas, matplotlib, cvxpy, openpyxl | see its README (run from `data/`) |
| `kredit-default-projekt` | pandas, numpy, scikit-learn, matplotlib, seaborn, joblib | `python src/01_eda.py` … `05_feature_importance.py` |
| `Binomial models`, `Option Pricing` | numpy, scipy | `python <script>.py` |
| `finance-etl-project` | MySQL 8, Python with a MySQL driver | `python3 scripts/run_etl.py --host … --user … --password …` |

## Housekeeping notes

Found while writing these READMEs, not changed:

- `Binomial models/Credit_Default_Swap_pricing.py` and `Option Pricing/Credit_structure.py`
  are identical, and neither prices a credit default swap: both calibrate a
  Black-Derman-Toy tree and price a swaption. The file names don't match the content.
- `finance-etl-project/finance-etl-project/` is an empty folder.
- `kredit-default-projekt/.~lock.Kreditrisikomodellierung_Projektbericht.docx#` is a
  LibreOffice lock file and can be deleted.
- `IRL-market-portfolio/src/pca_strategy/` is a copy of the package in
  `pca-trading-strategy/src/pca_strategy/`.
- `portfolio_optimization/src/Opt_by_PCA_Linear_Encoder.py` is a raw course-notebook export
  that cannot run as is; `pca-trading-strategy/` is its cleaned-up version.
- `ml_trading_pipeline/README.md` refers to `~/Quantitative_Finance/ml_trading_pipeline`;
  in this repository the path is `Quantitative_Finance_Projects/ml_trading_pipeline`.
