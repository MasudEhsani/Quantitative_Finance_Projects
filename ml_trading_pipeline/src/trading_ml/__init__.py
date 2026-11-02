"""trading_ml - a small, testable ML trading research pipeline.

Pipeline stages (each one is a sub-package):

    data -> features -> labels -> validation (walk-forward) -> models
         -> signals -> backtest -> tracking (MLflow) -> paper trading
"""

__version__ = "0.1.0"
