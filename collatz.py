#!/usr/bin/env python3
"""
collatz.py -- numerical statistics of Collatz (3n+1) trajectories.

WHAT THIS IS
------------
A numerical EXPERIMENT. We iterate the Collatz map and record, for every
starting value n in [1, N]:

  * total stopping time  = number of steps to first reach 1, and
  * maximum value reached along the way.

WHAT THIS IS NOT
----------------
This is NOT a proof, and it neither proves nor disproves the Collatz
conjecture. It is a FINITE verification: every statement below is
"verified for all n <= N" for the fixed bound N this script computes.
Extending the bound, however far, would still not be a proof. The
conjecture (that every n eventually reaches 1) remains open.

METHOD
------
The Collatz map is
    n -> n/2        if n even
    n -> 3n + 1     if n odd
Total stopping time tau(n) is the number of steps until the value 1 is
reached (tau(1) = 0).

We memoise: a numpy array STEP[n] caches tau(n) once known, and MAXV[n]
caches the largest value on the trajectory from n. When we walk a fresh
trajectory we stop as soon as we hit a value whose tau is already known,
then propagate the counts backward along the stored path. Because every
later value on the path is already resolved, the work is roughly linear
in N rather than exponential.

Reference values asserted below are the published/classic ones:
    tau: 1->0, 2->1, 3->7, 6->8, 7->16, 9->19, 18->20, 25->23, 27->111
    max value on the path from 27 is 9232.

DETERMINISM: no RNG is used anywhere; results are identical every run.
Only numpy is required.
"""

import sys
import time

import numpy as np

# ----------------------------------------------------------------------
# Bound. Raise it if you have time to spare; the report always states it.
# ----------------------------------------------------------------------
N = 1_000_000


def build_tables(N):
    """Return (STEP, MAXV) arrays indexed 1..N.

    STEP[n] = total stopping time of n (0 for n = 1)
    MAXV[n] = maximum value reached on the trajectory starting at n
    """
    STEP = np.full(N + 1, -1, dtype=np.int64)   # -1 == "not yet known"
    MAXV = np.zeros(N + 1, dtype=np.int64)
    # Values above N that we happen to visit get cached in dicts, since the
    # trajectory can overshoot the table (e.g. the path from 27 reaches 9232).
    step_over = {}
    maxv_over = {}

    STEP[1] = 0
    MAXV[1] = 1

    for n in range(1, N + 1):
        if STEP[n] != -1:
            continue  # already filled in by an earlier trajectory

        # Walk forward, recording the path, until we reach a known value.
        path = []
        v = n
        while v > N or STEP[v] == -1:
            path.append(v)
            if v & 1:
                v = 3 * v + 1
            else:
                v >>= 1

        # v is now resolved: tau(v) known.
        if v <= N:
            st = int(STEP[v])
            mx = int(MAXV[v])
        else:
            st = step_over[v]
            mx = maxv_over[v]

        # Propagate backward along the recorded path.
        for w in reversed(path):
            st += 1                       # one step w -> (next on path)
            mx = mx if mx >= w else w     # w itself is on the path
            if w <= N:
                STEP[w] = st
                MAXV[w] = mx
            else:
                step_over[w] = st
                maxv_over[w] = mx

    return STEP, MAXV


def total_stopping_time(n, STEP):
    """tau(n): steps to reach 1. Requires n <= the built bound."""
    return int(STEP[n])


def max_value_reached(n, MAXV):
    """Largest value on the trajectory from n. Requires n <= the built bound."""
    return int(MAXV[n])


def main():
    t0 = time.perf_counter()

    print("=" * 68)
    print("Collatz (3n+1) trajectory statistics -- numerical experiment")
    print("=" * 68)
    print(f"Bound N = {N:,}  (all starting values n in [1, N] are iterated)")
    print()

    STEP, MAXV = build_tables(N)
    build_t = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Reference checks -- must match exactly (published values).
    # ------------------------------------------------------------------
    ref_tau = {1: 0, 2: 1, 3: 7, 6: 8, 7: 16, 9: 19, 18: 20, 25: 23, 27: 111}
    print("Reference checks (total stopping time):")
    ok = True
    for n, expected in ref_tau.items():
        got = total_stopping_time(n, STEP)
        flag = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  tau({n:>3}) = {got:>4}   expected {expected:>4}   {flag}")

    ref_max = {27: 9232}
    print("Reference checks (maximum value on trajectory):")
    for n, expected in ref_max.items():
        got = max_value_reached(n, MAXV)
        flag = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  max_value({n}) = {got:>6}   expected {expected:>6}   {flag}")

    if not ok:
        print("\nREFERENCE CHECK FAILED -- aborting. The computation is wrong.")
        sys.exit(1)
    print("All reference checks passed.")
    print()

    # ------------------------------------------------------------------
    # (a) Every n in [1, N] reaches 1.
    # ------------------------------------------------------------------
    unreached = int(np.count_nonzero(STEP[1:] < 0))
    print("(a) Reachability")
    print(f"    Every n in [1, {N:,}] reaches 1: "
          f"{'yes' if unreached == 0 else 'NO'} "
          f"({unreached} unresolved).")
    print(f"    This is a finite check over [1, {N:,}]; it is NOT a proof.")
    print()

    # ------------------------------------------------------------------
    # (b) Record holders for total stopping time.
    # ------------------------------------------------------------------
    print("(b) Record holders for total stopping time (new maxima of tau)")
    print("    n          tau(n)    previous record   gain")
    records = []
    best = -1
    for n in range(1, N + 1):
        t = int(STEP[n])
        if t > best:
            prev = best
            gain = "" if best < 0 else f"+{t - best}"
            records.append((n, t))
            print(f"    {n:>9,}  {t:>8,}     {prev:>8}        {gain}")
            best = t
    print(f"    ({len(records)} records; last record at n = {records[-1][0]:,})")
    print()

    # ------------------------------------------------------------------
    # (c) Record holder for maximum value reached.
    # ------------------------------------------------------------------
    idx = int(np.argmax(MAXV[1:])) + 1
    peak = int(MAXV[idx])
    print("(c) Largest value reached by any trajectory starting at n <= "
          f"{N:,}")
    print(f"    n = {idx:,}  reaches a maximum of {peak:,}")
    print()

    # ------------------------------------------------------------------
    # (d) Histogram summary of stopping times.
    # ------------------------------------------------------------------
    tau = STEP[1:].astype(np.float64)  # tau for every n in [1, N]
    tmin = int(tau.min())
    tmax = int(tau.max())
    tmean = float(tau.mean())
    tmedian = float(np.median(tau))
    tstd = float(tau.std())
    print("(d) Summary of total stopping times over n in [1, N]")
    print(f"    count  = {N:,}")
    print(f"    min    = {tmin}")
    print(f"    median = {tmedian:.1f}")
    print(f"    mean   = {tmean:.4f}")
    print(f"    std    = {tstd:.4f}")
    print(f"    max    = {tmax}")
    print()
    print("    Coarse histogram (tau buckets):")
    edges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 400]
    hist, _ = np.histogram(tau, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        frac = c / N
        bar = "#" * int(round(frac * 60))
        print(f"      [{lo:>3}, {hi:>3}): {c:>8,}  {frac*100:6.2f}%  {bar}")
    print()

    # ------------------------------------------------------------------
    # (e) Empirical average vs. the naive heuristic. We REPORT what we
    #     measured; we do not assert a closed-form law.
    # ------------------------------------------------------------------
    # Heuristic: along a "typical" trajectory about log2(n) steps are even
    # halvings and the rest are odd steps each multiplying by ~3. A naive
    # 2^K vs 3^K balancing argument says the map should shrink by a factor
    # (3/4)^(#odd steps) on average, so trajectories plausibly grow only
    # logarithmically and tau(n) grows roughly like log n. The textbook-
    # looking constant 1/(1 - log2 3) is NEGATIVE (because log2 3 > 1), so
    # it is not a usable average; we print it only to show that, and report
    # the measured mean and the measured mean / mean(log2 n) instead of
    # asserting any formula.
    log2n = np.log2(np.arange(1, N + 1, dtype=np.float64))
    ratio = tmean / float(log2n.mean())
    naive = 1.0 / (1.0 - np.log2(3.0))
    print("(e) Average stopping time vs. heuristic")
    print(f"    measured mean tau over [1, N]        = {tmean:.4f}")
    print(f"    mean of log2(n) over [1, N]          = {float(log2n.mean()):.4f}")
    print(f"    measured mean tau / mean log2(n)     = {ratio:.4f}")
    print(f"    naive 1/(1 - log2 3)                 = {naive:.4f}  (negative -> not a valid average)")
    print("    Naive 2^K vs 3^K argument: each odd step multiplies by ~3 and")
    print("    each even step divides by 2; on average the map shrinks, so the")
    print("    average stopping time is expected to grow only logarithmically")
    print("    in n. We report the MEASURED ratio above rather than assert it.")
    print()

    wall = time.perf_counter() - t0
    print("-" * 68)
    print(f"Build time: {build_t:.2f}s   Total: {wall:.2f}s")
    print(f"All results are finite numerical evidence for n <= {N:,}.")
    print("The Collatz conjecture is neither proved nor disproved here.")
    print("-" * 68)


if __name__ == "__main__":
    main()
