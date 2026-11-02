#!/usr/bin/env bash
# Run the whole test suite with all print-outs and save a copy to reports/test_report.txt
#   ./run_tests.sh                 # everything
#   ./run_tests.sh tests/test_05_signals_backtest.py     # one step
set -o pipefail
cd "$(dirname "$0")"
mkdir -p reports
python -m pytest "${@:-tests}" 2>&1 | tee reports/test_report.txt
status=$?
echo
echo "full report saved to reports/test_report.txt"
exit $status
