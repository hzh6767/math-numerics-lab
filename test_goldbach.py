"""
pytest checks for goldbach.py.

Run with:  E:/anaconda3/python.exe -m pytest D:/gh-opt/math-numerics-lab/test_goldbach.py

These test a FINITE numerical experiment; they do not (and cannot) prove the
Goldbach conjecture.
"""

import numpy as np
import pytest

from goldbach import (
    REFERENCE_COUNTS,
    TWIN_PRIME_C2,
    check_references,
    direct_unordered_count,
    goldbach_counts,
    hardy_littlewood_estimate,
    run,
    sieve_primes,
)


def test_sieve_small():
    is_prime = sieve_primes(30)
    primes = [k for k in range(31) if is_prime[k]]
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_sieve_prime_count_1000():
    is_prime = sieve_primes(1000)
    assert int(is_prime.sum()) == 168  # pi(1000) = 168 (published)


def test_reference_partition_counts():
    """The four published reference values must match exactly."""
    is_prime = sieve_primes(2000)
    for n, expected in REFERENCE_COUNTS.items():
        assert direct_unordered_count(is_prime, n) == expected


def test_fft_matches_direct_scan():
    """FFT-based counts must equal a direct prime-index scan at sample n."""
    n_max = 200_000
    is_prime = sieve_primes(n_max)
    evens, counts = goldbach_counts(n_max, is_prime=is_prime)
    index_of = {int(n): k for k, n in enumerate(evens)}
    for n in (4, 10, 100, 1000, 12_346, 99_998, 200_000):
        assert int(counts[index_of[n]]) == direct_unordered_count(is_prime, n)


def test_no_zeros_in_small_range():
    """Every even n in [4, 100000] has at least one partition (finite check)."""
    n_max = 100_000
    evens, counts = goldbach_counts(n_max)
    assert counts.min() >= 1


def test_run_result_consistency():
    res = run(n_max=20_000, verbose=False)
    assert res["every_even_covered"] is True
    assert res["smallest_fail"] is None
    assert res["min_count"] >= 1
    assert res["max_count"] >= res["min_count"]
    assert res["n_even_values"] == (20_000 - 4) // 2 + 1


def test_check_references_returns_all():
    is_prime = sieve_primes(1000)
    evens, counts = goldbach_counts(1000, is_prime=is_prime)
    passed = check_references(is_prime, evens, counts)
    assert {n for n, _ in passed} == set(REFERENCE_COUNTS)


def test_hardy_littlewood_order_of_magnitude():
    """
    Heuristic is a magnitude guide only: the measured counts for several n sit
    within a factor of ~2 of the Hardy-Littlewood score.
    """
    n_max = 1_000_000
    evens, counts = goldbach_counts(n_max)
    index_of = {int(n): k for k, n in enumerate(evens)}
    assert abs(TWIN_PRIME_C2 - 0.6601618) < 1e-6
    for n in (1000, 10_000, 100_000, 1_000_000):
        measured = int(counts[index_of[n]])
        est = hardy_littlewood_estimate(n)
        ratio = measured / est
        assert 0.3 < ratio < 3.0
