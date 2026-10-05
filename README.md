# math-numerics-lab

Numerical experiments in mathematics. **This is an experiments repo, not a proof
repo.** Every number here is produced by actually running the code; each module
states the finite range over which its results were verified.

## Modules

### `zeta_zeros.py` -- nontrivial zeros of the Riemann zeta function

Computes the Hardy Z function

    theta(t) = arg Gamma(1/4 + i t/2) - (t/2) log(pi)
    Z(t)     = exp(i theta(t)) * zeta(1/2 + i t)      (real for real t)

with `mpmath` at 40-digit precision, finds the first 30 zeros `1/2 + i*gamma_n`
by scanning `Z(t)` for sign changes (step 0.05) and refining each with
`mp.findroot`.

**Verified range:** the first 30 zeros, i.e. `0 < gamma < 101.319`.

**Reference check:** the first five zeros are asserted against published values
to `< 1e-6` (observed max error `1.8e-15`). Values cross-checked against
`mpmath.zetazero` (an independent algorithm) to ~20 digits.

Reference values used:

| n | gamma_n |
|---|---------|
| 1 | 14.134725141734693 |
| 2 | 21.022039638771555 |
| 3 | 25.010857580145688 |
| 4 | 30.424876125859513 |
| 5 | 32.935061587739189 |

It also reports the zero-count against the Riemann-von Mangoldt main term
`N(T) ~ (T/2pi)log(T/2pi) - T/2pi + 7/8` and observed gaps against the average
spacing `2*pi/log(T/2pi)`.

**Scope / caveats.** Every zero reported lies on the critical line *by
construction* of the search (we only scan `Re(s) = 1/2`). This says **nothing**
about the Riemann Hypothesis and does not address whether off-line zeros exist.

#### Run

```
python zeta_zeros.py
```

Requires `mpmath`. Default run takes ~3 seconds.

## Reproduce

```
E:/anaconda3/python.exe zeta_zeros.py     # or `python zeta_zeros.py`
```

## License

MIT
