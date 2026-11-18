# Binomial models

Interest-rate and credit models on binomial lattices, plus a binomial option-pricing
spreadsheet. Stand-alone scripts; each prints its result.

```bash
pip install numpy scipy
python Credit_default_Swap_binomial.py
python Credit_Default_Swap_pricing.py
```

## Files

### `Credit_default_Swap_binomial.py` — defaultable zero-coupon bond

Despite the file name, this prices a **defaultable zero-coupon bond**, not a CDS.

1. **Short-rate tree** (10 periods): r(i, j) = r0 · u^j · d^(i−j), with r0 = 5 %, u = 1.1,
   d = 0.9. The tree is printed.
2. **Hazard rate** at each node: h(i, j) = a · b^(j − i/2), with a = 0.01, b = 1.01.
3. **Backward induction** with risk-neutral probability 0.5. At each node the bond either
   survives (prob. 1 − h) and continues, or defaults (prob. h) and pays the recovery
   R · F (R = 20 %, F = 100):

   P(i, j) = [ (1 − h) · ½ (P(i+1, j) + P(i+1, j+1)) + h · R · F ] / (1 + r(i, j))

**Result:** defaultable zero-coupon bond price ≈ **57.22**.

The short-rate tree is built twice. The first block stops one period early and is only
printed; the second block is the one used for pricing.

### `Credit_Default_Swap_pricing.py` — BDT calibration, swap and swaption

Despite the file name, this prices an **interest-rate swaption**, not a CDS. The same file
is in `../Option Pricing/Credit_structure.py`.

1. **Market data:** a 10-year spot-rate curve, 3.00 % to 3.70 %, and the zero-coupon bond
   prices it implies.
2. **Black-Derman-Toy tree:** r(i, j) = a_i · e^(b·j) with volatility parameter b = 0.1.
3. **Calibration:** the drift parameters a_0 … a_9 are found one period at a time
   (bootstrapping with Brent's root finder), so that the tree reprices every market
   zero-coupon bond exactly.
4. **Payer swap** (pay fixed 3.9 %, receive floating, notional 1,000,000) from year 3 to
   year 10, with payments in arrears, valued by backward induction.
5. **Swaption**: the right to enter that swap at t = 3 (one exercise date, so effectively
   European). The payoff max(swap value, 0) at t = 3 is discounted back to t = 0.

**Result:** swaption price ≈ **8,096.57**.

### `15period-Binomial.ods` — option-pricing spreadsheet

A 15-period Cox-Ross-Rubinstein binomial model (LibreOffice Calc):

- **Inputs:** T, N, Δt, r, σ, dividend yield c, S0, K, and from them u = e^(σ√Δt),
  d = 1/u and q = (e^((r−c)Δt) − d) / (u − d);
- **Calculations:** stock-price evolution; payoffs and prices of a European call, an
  American call (with intrinsic values) and a put; futures prices on the tree.

Two cells show `#VALUE!` errors; check those before relying on the futures part.
