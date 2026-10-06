# math-numerics-lab

Four independent **numerical experiments** in mathematics: computing zeros of
the Riemann zeta function on the critical line, enumerating Goldbach partitions
of even numbers, collecting Collatz (3n+1) trajectory statistics, and measuring
the convergence order of a pseudo-spectral solver for the 2D incompressible
Navier-Stokes equations. Every number quoted below was produced by actually
running the code in this repository on the interpreter and library versions
pinned in `requirements.txt`; none is transcribed from memory. Each module
reports the finite range over which its results hold and is checked against
hardcoded published reference values.

This is an **experiments repo, not a proof repo.** Four of the deepest open
problems in mathematics sit behind these four experiments: the Riemann
Hypothesis, the Goldbach conjecture, the Collatz conjecture, and the
existence-and-smoothness problem for the 3D Navier-Stokes equations. Nothing
here resolves any of them, and none of the code is intended to. What it does is
demonstrate, at a scale you can run on a laptop, the difference between a finite
numerical check and a proof: how a computation is validated against known
values, how the verified range is stated honestly, and where the evidence stops.

## What this is not

**Finite Goldbach verification is not a proof.** `goldbach.py` checks that every
even `n` up to 2,000,000 has at least one representation as a sum of two primes.
That is a finite statement about a finite set. The Goldbach conjecture asserts it
for *all* even `n > 2`, infinitely many of them; no amount of extending the bound
proves the general case. The counts here are an existence check, not a structural
proof: they tell you a partition exists, not why.

**Computing zeros on the critical line says nothing about zeros off it.**
`zeta_zeros.py` scans `Re(s) = 1/2` and reports the zeros it finds there. Every
zero it prints lies on the critical line *by construction of the search*, because
we only ever evaluate `zeta` at `1/2 + i t`. A zero with real part different from
1/2 would be invisible to this method, so the output carries no information about
whether such zeros exist. The Riemann Hypothesis remains open.

**Checking Collatz up to N is not a proof.** `collatz.py` iterates every starting
value in `[1, 1,000,000]` and confirms each reaches 1. This is the Collatz
conjecture verified for a finite initial segment. It cannot be extended to a proof
by pushing the bound higher; a proof needs a structural argument about why *every*
trajectory must descend, which this computation does not supply.

**Convergence on the Taylor-Green vortex says nothing about global regularity.**
`ns2d_convergence.py` measures the discretisation order of a solver on a smooth,
exactly-known solution. In two dimensions the smooth solution of the Navier-Stokes
equations is already *known* to exist for all time, so no existence question is at
stake. The open problem, smoothness in three dimensions, is not touched: a
spectral method converging at the expected order on one analytic initial datum is
a statement about the solver accuracy, not about the behaviour of the equations.

## The four experiments

| Module | Computes | Checked against (published value) | Range actually covered |
|---|---|---|---|
| `zeta_zeros.py` | First 30 nontrivial zeros `1/2 + i*gamma_n`, via sign changes of the Hardy `Z` function at 40-digit precision | `gamma_1..gamma_5` = 14.134725141734693, 21.022039638771555, 25.010857580145688, 30.424876125859513, 32.935061587739189 | `0 < gamma < 101.319` (30 zeros); max observed error `1.78e-15` |
| `goldbach.py` | Unordered Goldbach partition count `U(n)` for every even `n`, by FFT autocorrelation of the prime indicator | `U(4)=1`, `U(10)=2`, `U(100)=6`, `U(1000)=28`; also `pi(1000)=168` | every even `n` in `[4, 2,000,000]` |
| `collatz.py` | Total stopping time `tau(n)` and trajectory maximum for every `n` | `tau(1)=0, tau(2)=1, tau(3)=7, tau(6)=8, tau(7)=16, tau(9)=19, tau(18)=20, tau(25)=23, tau(27)=111`; max along `27` = 9232 | every `n` in `[1, 1,000,000]` |
| `ns2d_convergence.py` | Temporal and spatial convergence order of a Fourier pseudo-spectral + RK4 solver for the vorticity-streamfunction form | Taylor-Green exact solution; analytic enstrophy decay rate `4*nu` | `N` in `{32, 64, 128, 256}`, `dt` in `{0.2, 0.1, 0.05, 0.025}`, `nu = 0.1` |

## Results actually obtained

All values below are transcribed from the modules' own run output (see the
`results/` directory).

### Riemann zeta zeros (`zeta_zeros.py`)

First five zeros `1/2 + i*gamma_n` (30 computed in total, `0 < gamma < 101.319`):

| n | gamma_n |
|---|---------|
| 1 | 14.134725141734694 |
| 2 | 21.022039638771555 |
| 3 | 25.010857580145689 |
| 4 | 30.424876125859513 |
| 5 | 32.935061587739190 |

Max absolute difference from the published references over the first five:
`1.78e-15`. Zero count over `0 < gamma < 101.3189`: **30** computed vs
Riemann-von Mangoldt main term `N(T) = 29.5846` (difference `+0.415`). Observed
gap / average-spacing extremes: `0.5276` and `1.5703`.

### Goldbach partitions (`goldbach.py`)

Bound: every even `n` in `[4, 2,000,000]` (999,999 even values).

| Quantity | Value |
|---|---|
| Evens with no representation | none |
| Minimum partition count | 1 (at `n = 4`) |
| Largest `n` with exactly one partition | 12 |
| Maximum partition count | 27,988 (at `n = 1,981,980`) |
| Average partition count | 5,987.2143 |

Measured count vs Hardy-Littlewood heuristic (order-of-magnitude only):

| n | measured | HL score | ratio |
|---|---|---|---|
| 1,000 | 28 | 36.9 | 0.759 |
| 10,000 | 127 | 207.5 | 0.612 |
| 100,000 | 810 | 1,328.2 | 0.610 |
| 1,000,000 | 5,402 | 9,223.3 | 0.586 |
| 2,000,000 | 9,720 | 16,726.1 | 0.581 |

### Collatz statistics (`collatz.py`)

Bound: every `n` in `[1, 1,000,000]` reaches 1 (0 unresolved).

| Quantity | Value |
|---|---|
| `tau(27)` | 111 |
| max along trajectory from 27 | 9,232 |
| Largest peak from any `n <= 10^6` | 56,991,483,520 (from `n = 704,511`) |
| Last record holder for `tau` below `10^6` | `n = 837,799` with `tau = 524` |
| Mean / median / max of `tau` | 131.4344 / 125.0 / 524 |
| Measured mean `tau` / mean `log2 n` | 7.1088 |

### 2D Navier-Stokes convergence (`ns2d_convergence.py`)

Solver: Fourier pseudo-spectral (collocation) in vorticity-streamfunction form,
with the **2/3-rule dealiasing** applied to the nonlinear term, and classical
explicit RK4 time stepping. `nu = 0.1`, `T = 1.0`. The explicit-RK4 diffusive
stability limit is `nu * K2max * dt < ~2.785`, which sets the usable step size at
each resolution; steps above that bound are unstable and are not reported.

**The Taylor-Green case is degenerate for spatial convergence.** Its advection
term `(u . grad) w` vanishes identically (measured `1.127e-14`, i.e. roundoff)
and its vorticity `2 cos x cos y` is a single exactly-representable Fourier mode.
So the scheme reproduces it to machine precision at every `N`, and any "order"
read off that table is roundoff noise, not the scheme's accuracy. It is kept as a
correctness check:

| N | L2 error vs Taylor-Green |
|---|---|
| 32 | 6.3264e-16 |
| 64 | 7.0672e-16 |
| 128 | 7.5514e-16 |
| 256 | 6.6628e-16 |

The sign and scale conventions are verified at `N = 64`: recovered `psi`, `u`, `v`
match the closed forms to `7.8e-16`, `6.7e-16`, `6.7e-16`.

**A non-degenerate test — method of manufactured solutions** with
`w_ex = e^{-2 nu t} P_r(x) P_r(y)`, `r = 0.7` (full spectrum, genuine
nonlinearity, plus its source term) — measures the real spatial order:

| N | L2 error | observed order p |
|---|---|---|
| 16 | 8.0960e-01 | |
| 32 | 1.0323e-01 | 2.97 |
| 64 | 1.3306e-03 | 6.28 |
| 128 | 4.6609e-07 | 11.48 |
| 256 | 5.8372e-14 | 22.93 |

The error falls by a large factor per doubling and the observed order **grows**
with `N`: the signature of spectral (exponential) convergence on an analytic
solution, not a fixed algebraic order.

**Temporal convergence** at fixed `N = 64` (spatially exact for Taylor-Green, so
time integration is the only error):

| nsteps | dt | L2 error | observed order p |
|---|---|---|---|
| 80 | 1.250e-02 | 5.3633e-14 | |
| 160 | 6.250e-03 | 3.8868e-15 | 3.79 |
| 320 | 3.125e-03 | 3.1959e-16 | 3.60 |
| 640 | 1.563e-03 | 4.9523e-16 | -0.63 |

RK4 is 4th order, so `p -> 4` while truncation dominates (3.79, 3.60); the
finest refinement reaches the roundoff floor near `1e-16`, where the measured
order degrades into noise. The sweep starts at `nsteps = 80` rather than 20
because `dt = 0.05` and `0.025` at `N = 64` give `nu * K2max * dt = 10.24` and
`5.12`, both **above** the `~2.785` stability limit — they are diffusive
unstable, not merely inaccurate, so they are excluded.

**Enstrophy decay** at `N = 64`, `dt = 5e-4`: the discrete `Z(t) = <w^2>` tracks
the analytic `Z(0) e^{-4 nu t}` with a maximum relative drift of `3.074e-15` over
`0 <= t <= 1`. (This is measured, not assumed: the module integrates with
`track_enstrophy=True` and takes the max deviation from the closed form.)

## Why this is interesting

**Riemann zeta.** The Riemann Hypothesis, that every nontrivial zero has real
part exactly 1/2, controls the error term in the prime counting function and
remains open. A real attack replaces the numerical scan with the analytic theory
of the zeta function (explicit formulas, the functional equation) and, for the
function-field analogue over finite fields, the Weil conjectures proved by
Deligne, where the analogue of RH is a theorem.

**Goldbach.** The conjecture that every even `n > 2` is a sum of two primes is
open; Helfgott (2013) proved the *weak* form (every odd `n > 5` is a sum of three
primes). A real approach uses the circle method and the Hardy-Littlewood
asymptotic `n / (log n)^2`, the heuristic this module compares against, to bound
the count from below.

**Collatz.** The conjecture that every trajectory reaches 1 is open and famously
resistant; Tao (2019) proved that almost all trajectories (in a logarithmic
density sense) eventually descend below any unbounded function. A proof would
need a structural argument controlling the (3n+1) map, not a larger verification
bound.

**Navier-Stokes.** Smoothness in 3D is one of the Clay Millennium Problems and is
open; in 2D it is a theorem. Beale-Kato-Majda (1984) gives a conditional
regularity criterion (blow-up is equivalent to blow-up of the vorticity `L^inf`
norm), and Buckmaster-Vicol (2019) showed non-uniqueness of weak solutions. A
real approach works on such criteria, not on a single discretisation.

## How to run

Requires the pinned environment (numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.6,
mpmath 1.3.0, pytest 8.4.2). This environment is Anaconda Python 3.13.9 at
`E:/anaconda3/python.exe`.

Run any module standalone (each prints its own summary):

```
E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/zeta_zeros.py
E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/goldbach.py
E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/collatz.py
E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/ns2d_convergence.py
```

Run the test suite. The `tests/` directory holds 20 tests; running from the repo
root also collects the standalone `test_goldbach.py` (8 more) for 28 total. The
small bounds keep the whole suite well under two minutes:

```
E:/anaconda3/python.exe -m pytest -q D:/gh-opt/math-numerics-lab/tests   # 20
E:/anaconda3/python.exe -m pytest -q D:/gh-opt/math-numerics-lab         # 28
```

Note for Git Bash users: `python` is not on `PATH` here and the interpreter
resolves `/d/...` paths as `C:\d\...`, so always invoke `E:/anaconda3/python.exe`
explicitly with a drive-letter path (as above), not `/d/...`.

### A harmless startup message

On this Anaconda-on-Windows setup, importing `mpmath` (used by `zeta_zeros.py`)
first tries to load the optional `gmpy2` accelerator, whose DLL fails to load
with `Windows fatal exception: code 0xc0000139`. This is a Windows DLL search
issue in the `gmpy2` build, not a fault in this code: `mpmath` catches it and
falls back to its pure-Python integer backend, and every result and assertion is
unchanged. You may see that message printed before the test output; it does not
mean a test failed. All 20 tests still pass, and the zeta zeros are computed to
the same 40-digit precision either way.

## References

- B. Riemann, *Uber die Anzahl der Primzahlen unter einer gegebenen Grosse*, 1859.
- G. H. Hardy and J. E. Littlewood, *Some problems of Partitio Numerorum V*, 1923
  (Goldbach and the singular series).
- H. A. Helfgott, *The ternary Goldbach conjecture is true*, 2013.
- T. Tao, *Almost all Collatz orbits attain almost bounded values*, 2019.
- J. T. Beale, T. Kato, A. Majda, *Remarks on the breakdown of smooth solutions
  for the 3-D Euler equations*, Comm. Math. Phys. 94 (1984).
- G. I. Taylor and A. E. Green, *Mechanism of the production of small eddies from
  large ones*, Proc. R. Soc. A 158 (1937).
- T. Buckmaster and V. Vicol, *Nonuniqueness of weak solutions to the
  Navier-Stokes equation*, Ann. of Math. 189 (2019).

## License

MIT, see `LICENSE`.
