"""
Goldbach partitions of even numbers -- a numerical experiment.

This module SIEVES the primes up to a bound N, counts for every even n in
[4, N] the number of unordered representations n = p + q with p <= q and both
p and q prime ("Goldbach partitions"), and reports statistics about those
counts.  It also compares the measured counts against the Hardy-Littlewood
heuristic for their order of magnitude.

WHAT THIS IS AND IS NOT
-----------------------
Goldbach's conjecture (every even n > 2 is a sum of two primes) is OPEN.
Nothing here proves it.  Everything below is a FINITE verification: the counts
are correct for the stated range and say nothing about larger n.  Every number
printed by run() is computed at run time from the sieve; none is hardcoded
except the published reference values used for assertion.

METHOD
------
- Primes by the sieve of Eratosthenes (numpy boolean sieve up to N).
- Counts from a linear autocorrelation of the primality indicator vector,
  evaluated with an FFT (numpy.fft.rfft).  The convolution coefficients are
      p_n = #{ordered (p, q) with p + q = n, both p and q prime}.
  The unordered count (p <= q) is then
      U(n) = (p_n + [n/2 is prime]) / 2 ,
  where the extra 1 is present exactly when n/2 is prime, i.e. when the pair
  (n/2, n/2) is counted once as ordered but is a single unordered partition.
- The FFT is exact for these magnitudes.  At N = 2,000,000 the largest counts
  are order 1e4; float64 relative round-off ~1e-16 gives absolute error
  ~1e-12, well below 0.5, so rint() recovers exact integers.  This is checked
  against a direct prime-index scan for the reference values and for a sample
  of larger even n.
- A naive double loop over all pairs is O(N^2) and is not used.

Deterministic: no RNG, no wall-clock value enters any result.
Standalone:  E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/goldbach.py
"""

import numpy as np
from math import log

# Twin-prime constant C2 = prod_{p>2} (1 - 1/(p-1)^2), a published constant.
TWIN_PRIME_C2 = 0.6601618158468696

DEFAULT_N = 2_000_000

# Even n used to cross-check the FFT counts against a direct scan (exact).
FFT_VALIDATION_NS = (4, 10, 100, 1000, 123456, 500000, 999998, 1999998, 2000000)

# Published reference values asserted exactly.
REFERENCE_COUNTS = {
    4: 1,      # 4 = 2 + 2
    10: 2,     # 10 = 3 + 7 = 5 + 5
    100: 6,    # 3+97, 11+89, 17+83, 29+71, 41+59, 47+53
    1000: 28,
}


def sieve_primes(n):
    """Boolean array is_prime of length n+1; is_prime[k] is True iff k is prime."""
    if n < 0:
        raise ValueError("n must be non-negative")
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    i = 2
    while i * i <= n:
        if is_prime[i]:
            is_prime[i * i:: i] = False
        i += 1
    return is_prime


def goldbach_counts(n_max, is_prime=None):
    """
    Return (evens, counts):
      evens  = array [4, 6, 8, ..., n_max] (n_max forced even),
      counts = unordered Goldbach partition count U(n) for each even n.

    Uses the FFT autocorrelation of the primality indicator; the result is an
    exact integer array (validated by rint and by direct scan in run()).
    """
    if n_max % 2:
        n_max -= 1
    if n_max < 4:
        raise ValueError("n_max must be at least 4")
    if is_prime is None:
        is_prime = sieve_primes(n_max)

    size = 1
    while size < 2 * n_max + 1:
        size <<= 1
    a = np.zeros(size, dtype=np.float64)
    a[: n_max + 1] = is_prime.astype(np.float64)
    fa = np.fft.rfft(a)
    conv = np.fft.irfft(fa * fa, n=size)[: n_max + 1]

    ordered = np.rint(conv).astype(np.int64)
    assert ordered.min() >= 0, "negative ordered count -- FFT precision failure"

    evens = np.arange(4, n_max + 1, 2)
    half_is_prime = is_prime[evens // 2].astype(np.int64)  # 1 iff n/2 prime
    counts = (ordered[evens] + half_is_prime) // 2
    return evens, counts


def direct_unordered_count(is_prime, n):
    """Exact unordered Goldbach count of n by a prime-index scan (validation only)."""
    total = 0
    for p in range(2, n // 2 + 1):
        if is_prime[p] and is_prime[n - p]:
            total += 1
    return total


def hardy_littlewood_estimate(n, c2=TWIN_PRIME_C2):
    """
    Hardy-Littlewood heuristic count of Goldbach representations:
        2 * C2 * n / (log n)^2 * prod_{p | n, p > 2} (p - 1)/(p - 2).
    This is an ASYMPTOTIC HEURISTIC, not an exact formula and not a proof of
    anything; it is used only to show measured counts agree in magnitude.
    Exposed for tests and external use (uses natural log).
    """
    if n < 6:
        return float("nan")
    prod = 1.0
    m = n
    # Strip all factors of 2 first: the product runs only over odd primes p | n.
    while m % 2 == 0:
        m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            prod *= (p - 1) / (p - 2)
        p += 2
    if m > 2:
        prod *= (m - 1) / (m - 2)
    return 2.0 * c2 * n / (log(n) ** 2) * prod


def check_references(is_prime, evens, counts):
    """
    Assert the published reference values exactly, against BOTH the FFT counts
    and a direct prime-index scan.  Returns a list of (n, expected) that passed.
    """
    passed = []
    index_of = {int(n): k for k, n in enumerate(evens)}
    for n, expected in REFERENCE_COUNTS.items():
        direct = direct_unordered_count(is_prime, n)
        fft_val = int(counts[index_of[n]])
        assert direct == expected, f"direct scan for {n}: {direct} != {expected}"
        assert fft_val == expected, f"FFT scan for {n}: {fft_val} != {expected}"
        passed.append((n, expected))
    return passed


def run(n_max=DEFAULT_N, verbose=True):
    """Run the experiment up to even n_max, print a summary, return results dict."""
    if n_max % 2:
        n_max -= 1

    is_prime = sieve_primes(n_max)
    evens, counts = goldbach_counts(n_max, is_prime=is_prime)

    # (a) is there any even n in [4, n_max] with no representation?
    zero_mask = counts == 0
    none_in_range = not bool(zero_mask.any())
    smallest_fail = None if none_in_range else int(evens[zero_mask][0])

    # (b) minimum count, its n, and the largest n with exactly one partition.
    min_count = int(counts.min())
    min_n = int(evens[int(np.argmin(counts))])
    one_ns = evens[counts == 1]
    first_one_n = int(one_ns[0]) if one_ns.size else None
    largest_one_n = int(one_ns[-1]) if one_ns.size else None

    # (c) maximum count and its n.
    max_count = int(counts.max())
    max_n = int(evens[int(np.argmax(counts))])

    # (d) average count.
    avg_count = float(counts.mean())

    # (e) record holders: n where the running maximum strictly increases.
    running_max = np.maximum.accumulate(counts)
    is_record = np.empty(counts.shape, dtype=bool)
    is_record[0] = True
    is_record[1:] = running_max[1:] > running_max[:-1]
    record_ns = evens[is_record]
    record_counts = counts[is_record]

    # (f) FFT-vs-direct validation on a fixed set of even n.
    index_of = {int(n): k for k, n in enumerate(evens)}
    fft_validation = []
    for n in FFT_VALIDATION_NS:
        if n > n_max:
            continue
        direct = direct_unordered_count(is_prime, n)
        fft_val = int(counts[index_of[n]])
        assert fft_val == direct, f"FFT {fft_val} != direct {direct} at n={n}"
        fft_validation.append((n, direct))

    # Reference-value assertions.
    reference_checks = check_references(is_prime, evens, counts)

    # Hardy-Littlewood comparison.
    hl_ns = [n for n in (1000, 10_000, 100_000, 1_000_000, 2_000_000) if n <= n_max]
    hl_rows = []
    for n in hl_ns:
        measured = int(counts[index_of[n]])
        est = hardy_littlewood_estimate(n)
        hl_rows.append((n, measured, est, measured / est))

    results = {
        "n_max": n_max,
        "every_even_covered": none_in_range,
        "smallest_fail": smallest_fail,
        "min_count": min_count,
        "min_n": min_n,
        "first_one_n": first_one_n,
        "largest_one_n": largest_one_n,
        "n_with_count_one": int(one_ns.size),
        "max_count": max_count,
        "max_n": max_n,
        "avg_count": avg_count,
        "n_even_values": int(evens.size),
        "record_ns": record_ns,
        "record_counts": record_counts,
        "hl_rows": hl_rows,
        "reference_checks": reference_checks,
        "fft_validation": fft_validation,
    }

    if verbose:
        print("=" * 72)
        print("Goldbach partitions of even numbers -- numerical experiment")
        print("=" * 72)
        print(f"Bound (largest even n tested) : {n_max:,}  (primes sieved to {n_max:,})")
        print(f"Even values examined          : {results['n_even_values']:,} "
              f"(n = 4, 6, ..., {n_max:,})")
        print()
        print("[a] Any even n in range with NO representation found?")
        if none_in_range:
            print(f"    NONE. Every even n in [4, {n_max:,}] has at least one partition.")
        else:
            print(f"    Found: {smallest_fail}  (unexpected)")
        print()
        print("[b] Minimum partition count over the range")
        print(f"    min count            = {min_count}  (attained at n = {min_n})")
        print(f"    first n with 1       = {first_one_n}")
        print(f"    largest n with 1     = {largest_one_n}")
        print(f"    #(even n with count 1) = {results['n_with_count_one']:,}")
        print()
        print("[c] Maximum partition count over the range")
        print(f"    max count = {max_count}  (attained at n = {max_n})")
        print()
        print("[d] Average partition count over all even n in range")
        print(f"    average = {avg_count:.4f}")
        print()
        print("[e] Record-setting n (Goldbach comet band structure)")
        print(f"    distinct record holders up to {n_max:,}: {record_ns.size}")
        for lo, hi in [(0, 1000), (900_000, 1_000_001), (1_500_000, 2_000_001)]:
            sel = (record_ns >= lo) & (record_ns <= hi)
            if sel.any():
                print(f"    -- record holders with n in [{lo:,}, {hi:,}]:")
                for n, c in zip(record_ns[sel], record_counts[sel]):
                    print(f"         n = {int(n):>9,}   count = {int(c)}")
        print()
        print("[f] Measured vs Hardy-Littlewood heuristic (order-of-magnitude only)")
        print("    score = 2 * C2 * n / (log n)^2 * prod_{p|n, p>2} (p-1)/(p-2)")
        print(f"    {'n':>10}  {'measured':>10}  {'HL score':>10}  {'ratio':>7}")
        for n, measured, est, ratio in hl_rows:
            print(f"    {n:>10,}  {measured:>10,}  {est:>10.1f}  {ratio:>7.3f}")
        print("    (heuristic asymptotic, not an exact formula; magnitude agreement only)")
        print()
        print("Reference values asserted exactly (FFT and direct scan):")
        for n, expected in reference_checks:
            print(f"    n = {n:>4}: {expected}  OK")
        print("FFT-vs-direct cross-checks (exact):")
        for n, direct in fft_validation:
            print(f"    n = {n:>9,}: {direct}  OK")
        print()
        print("FINITE RANGE ONLY: the above verifies the counts for every even")
        print(f"n in [4, {n_max:,}]. A finite verification is NOT a proof of the")
        print("Goldbach conjecture, which remains open.")
        print("=" * 72)

    return results


if __name__ == "__main__":
    run()
