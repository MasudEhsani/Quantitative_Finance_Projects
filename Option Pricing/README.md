# Option Pricing

## `Credit_structure.py` — swaption on a Black-Derman-Toy tree

Despite its name, this script prices an **interest-rate swaption**. It is identical to
`../Binomial models/Credit_Default_Swap_pricing.py`; the full description is in
[`../Binomial models/README.md`](../Binomial%20models/README.md).

In short:
1. **Calibration:** a Black-Derman-Toy short-rate tree, r(i, j) = a_i · e^(0.1·j), is
   calibrated to a 10-year spot curve (3.00 % to 3.70 %) by bootstrapping.
2. **Swap:** a payer swap (fixed 3.9 %, notional 1,000,000, years 3 to 10) is valued by
   backward induction.
3. **Swaption:** the option to enter that swap at t = 3 is priced by discounting
   max(swap value, 0) back to t = 0.

```bash
pip install numpy scipy
python Credit_structure.py      # prints the calibrated a_i and the swaption price (≈ 8,096.57)
```
