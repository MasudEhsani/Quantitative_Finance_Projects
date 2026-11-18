# Portfolio optimisation and risk

Stand-alone scripts on mean-variance portfolio theory, return estimation and tail risk,
using a universe of 23 US-listed stocks (IBM, GE, LLY, F, BAC, …). Each script prints its
results; some also show plots.

## Setup

```bash
pip install numpy scipy pandas matplotlib cvxpy openpyxl yfinance
```

The scripts that load data read `stocks.xlsx` **from the folder you run them in**, and the
file is in `data/`, so run them from there:

```bash
cd portfolio_optimization/data
python ../src/Portfolio_Risk_analysis.py
python ../src/cVar_risk.py
```

The other scripts contain their data and can be run from anywhere.

## Scripts (`src/`)

| Script | What it does |
|---|---|
| `Sharp_Optimal_portfolio.py` | Three-asset example: **global minimum-variance** portfolio, analytically (Σ⁻¹1 / 1ᵀΣ⁻¹1) and numerically (SLSQP), with the difference between the two; then the **tangency (Sharpe-optimal)** portfolio for a risk-free rate of 1 % |
| `Two_Funds_theorem.py` | 23 stocks with given daily means and covariance: two **minimum-variance portfolios** for daily target returns of 0.05 % and 0.08 % (fully invested, short selling allowed); the efficient frontier traced by mixing them (**two-fund theorem**), plotted; then the **tangency portfolio** for a daily risk-free rate of 0.01 % |
| `Portfolio_opt_Estimation.py` | Mean-variance optimisation from sample estimates of three return series: the maximum-return portfolio for 5 % volatility, realised versus estimated return, **tangency portfolio + risk-free asset scaled to a target volatility**; empirical VaR and CVaR; a binomial probability check |
| `Portfolio_Risk_analysis.py` | Loads daily returns from `data/stocks.xlsx`: sample means and covariance, **shrinkage estimator** of expected returns (towards the cross-sectional mean, α = 0.9), long-only **Markowitz** minimum-variance portfolio |
| `cVar_risk.py` | Same data and shrinkage estimate; **minimum-variance portfolios with cvxpy** for a daily return of at least 0.05 %: long-only, and long-short with the total short position limited to 10 %; per-stock variance, **VaR and CVaR** (which stock is riskiest by each measure), and the risk of the equal-weight portfolio |
| `Opt_by_PCA_Linear_Encoder.py` | Raw export of a course notebook on eigen-portfolios and the Absorption Ratio. It can't run as it is (it imports a `grading` module and `tensorflow.contrib`, and reads a `/home/jovyan/...` path). The cleaned-up, tested version is [`../pca-trading-strategy/`](../pca-trading-strategy/README.md). |

## Data

- `data/stocks.xlsx`: daily data for the 23 stocks; the first column is the date and IBM
  is the first asset.
- `Portfolio_Risk_analysis.py` and `cVar_risk.py` also contain a commented-out block that
  downloads the same prices from Yahoo Finance with `yfinance` (2016-01-03 to 2017-12-29).
