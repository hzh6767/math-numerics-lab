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

Temporal convergence at fixed `N = 32`, `T = 1.0` (`nu = 0.1`, Taylor-Green
exact solution):

| dt | max abs error |
|---|---|
| 0.2 | 1.397945e-05 |
| 0.1 | 8.517524e-07 |
| 0.05 | 2.751954e-11 |
| 0.025 | 1.712852e-12 |

Measured orders: `4.0367`, `14.9177` (an artifact of the error reaching the
round-off floor), `4.0060`; minimum order `4.0060` (RK4 implies about 4).

Spatial convergence (smooth blob, `nu = 0.05`, `T = 0.2`, `dt = 1e-3`), error vs
the next-finer grid: `32 -> 64: 1.264768e-07`, `64 -> 128: 3.269095e-08`,
`128 -> 256: 8.070901e-09`, roughly a factor of 4 per doubling of `N`.

Enstrophy decay at `N = 32`: measured rate `0.400029` vs analytic `4*nu = 0.4`
(relative error `7.14e-05`).

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
