"""
pytest suite over the four numerical-experiment modules in the repo root:

    zeta_zeros.py        -- nontrivial zeros of zeta on the critical line
    goldbach.py          -- Goldbach partitions of even numbers
    collatz.py           -- Collatz trajectory statistics
    ns2d_convergence.py  -- 2D Navier-Stokes convergence study

Every assertion below pins a PUBLISHED reference value (or the exact property
the module reports); nothing here is a mere smoke test. All numbers were
recomputed by running the modules -- see the project README.

These test FINITE numerical experiments. They do not, and cannot, prove the
Riemann Hypothesis, Goldbach's conjecture, the Collatz conjecture, or the
regularity of the 2D/3D Navier-Stokes equations.

Bounds are deliberately SMALL so the whole suite runs in well under two
minutes; each module owns its own (much larger) default run in its __main__.
"""

import numpy as np
import pytest


# ===========================================================================
# zeta_zeros.py -- first five imaginary parts of the nontrivial zeros
# ===========================================================================

PUBLISHED_GAMMAS = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
]


@pytest.fixture(scope="module")
def zeta_module():
    import zeta_zeros
    return zeta_zeros


@pytest.fixture(scope="module")
def first_five_zeros(zeta_module):
    """Locate the first five zeros once and share them across the zeta tests."""
    zeros = zeta_module.find_zeros(5)
    assert len(zeros) == 5, f"expected 5 zeros, found {len(zeros)}"
    return zeros


def test_zeta_reference_table_is_published(zeta_module):
    """The module's own hardcoded table must equal the published values."""
    assert [float(v) for v in zeta_module.REFERENCE_GAMMAS] == pytest.approx(
        PUBLISHED_GAMMAS, abs=1e-15
    )


def test_zeta_first_five_zeros_match_references_to_6dp(first_five_zeros):
    """gamma_1..gamma_5 agree with published values to 6 decimal places."""
    for i, (computed, ref) in enumerate(zip(first_five_zeros, PUBLISHED_GAMMAS), 1):
        assert abs(float(computed) - ref) < 5e-7, (
            f"gamma_{i}: computed {float(computed)!r} vs reference {ref!r}"
        )


def test_zeta_max_reference_error(zeta_module, first_five_zeros):
    """max |computed - reference| over the first five zeros is < 1e-6."""
    assert zeta_module.max_reference_error(first_five_zeros) < 1e-6


def test_zeta_hardy_Z_is_real_and_matches_zeta_form(zeta_module):
    """Z(t) = exp(i theta) zeta(1/2 + i t) equals mpmath.siegelz and is real."""
    import mpmath as mp
    for t in ("5.0", "17.0", "50.0"):
        z_siegelz, z_from_zeta = zeta_module.verify_Z_identity(mp.mpf(t))
        assert abs(mp.im(z_from_zeta)) < mp.mpf("1e-30")
        assert abs(mp.re(z_from_zeta) - z_siegelz) < mp.mpf("1e-30")


def test_zeta_Z_sign_flips_at_each_zero_and_alternates(zeta_module, first_five_zeros):
    """
    Each reported gamma is bracketed by a genuine sign change of Z, the sign is
    constant between consecutive zeros (no missed root), and the sign of Z
    just to the RIGHT of a zero alternates (a local max/min structure): each
    simple zero flips Z, so the value in the interval to its right differs from
    the previous interval's value.
    """
    for i, g in enumerate(first_five_zeros, 1):
        eps = 0.5  # far smaller than any zero gap here (min gap ~ 2.9)
        z_lo = zeta_module.hardy_Z(g - eps)
        z_hi = zeta_module.hardy_Z(g + eps)
        assert (z_lo > 0) != (z_hi > 0), f"Z does not flip sign across gamma_{i}"

    # Z keeps one sign on each open interval between consecutive zeros ...
    for i in range(len(first_five_zeros) - 1):
        mid = (first_five_zeros[i] + first_five_zeros[i + 1]) / 2
        assert zeta_module.hardy_Z(mid) != 0
        a = zeta_module.hardy_Z(first_five_zeros[i] + 0.5)
        b = zeta_module.hardy_Z(mid)
        assert (a > 0) == (b > 0), f"Z changes sign inside interval {i + 1}"

    # ... and that sign alternates interval to interval (each zero is simple).
    interval_signs = [
        zeta_module.hardy_Z((first_five_zeros[i] + first_five_zeros[i + 1]) / 2) > 0
        for i in range(len(first_five_zeros) - 1)
    ]
    for i in range(len(interval_signs) - 1):
        assert interval_signs[i] != interval_signs[i + 1]


def test_zeta_zero_counting_main_term(zeta_module, first_five_zeros):
    """Riemann-von Mangoldt main term is in the right ballpark at the last zero."""
    T = float(first_five_zeros[-1]) + 0.001
    n_main = float(zeta_module.N_main_term(T))
    counted = sum(1 for g in first_five_zeros if 0 < float(g) < T)
    assert counted == 5
    assert abs(counted - n_main) < 1.0  # main term is a smooth approximation


# ===========================================================================
# goldbach.py -- partition counts and exact published reference values
# ===========================================================================

@pytest.fixture(scope="module")
def goldbach_module():
    import goldbach
    return goldbach


def test_goldbach_sieve_primes(goldbach_module):
    is_prime = goldbach_module.sieve_primes(30)
    assert [k for k in range(31) if is_prime[k]] == [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    ]


def test_goldbach_pi_1000(goldbach_module):
    """pi(1000) = 168 (published)."""
    assert int(goldbach_module.sieve_primes(1000).sum()) == 168


def test_goldbach_reference_counts_direct_and_fft(goldbach_module):
    """4->1, 10->2, 100->6, 1000->28 by BOTH the direct scan and the FFT path."""
    is_prime = goldbach_module.sieve_primes(2000)
    evens, counts = goldbach_module.goldbach_counts(2000, is_prime=is_prime)
    index_of = {int(n): k for k, n in enumerate(evens)}
    expected = {4: 1, 10: 2, 100: 6, 1000: 28}
    assert set(goldbach_module.REFERENCE_COUNTS) == set(expected)
    for n, exp in expected.items():
        assert int(counts[index_of[n]]) == exp, f"FFT count at {n}"
        assert goldbach_module.direct_unordered_count(is_prime, n) == exp, f"direct {n}"


def test_goldbach_every_even_has_a_partition_modest_bound(goldbach_module):
    """Every even n in [4, 20000] has at least one Goldbach partition (finite)."""
    evens, counts = goldbach_module.goldbach_counts(20_000)
    assert counts.min() >= 1
    assert int(counts[0]) == 1  # n = 4 = 2 + 2


def test_goldbach_largest_single_partition_n_matches_reported(goldbach_module):
    """
    The largest n in the tested range with exactly one partition is whatever
    the module reports -- assert that property against the counts array itself
    (verified value over [4, 20000] is 12 = 5 + 7).
    """
    bound = 20_000
    res = goldbach_module.run(n_max=bound, verbose=False)
    evens, counts = goldbach_module.goldbach_counts(bound)
    one_ns = evens[counts == 1]
    assert res["largest_one_n"] == int(one_ns[-1])
    assert res["first_one_n"] == int(one_ns[0])
    assert res["largest_one_n"] == 12  # the actual computed value
    assert res["every_even_covered"] is True
    assert res["smallest_fail"] is None


def test_goldbach_hardy_littlewood_constant_and_order(goldbach_module):
    """Twin-prime constant C2 pinned; HL heuristic is the right order of magnitude."""
    assert abs(goldbach_module.TWIN_PRIME_C2 - 0.6601618158468696) < 1e-15
    evens, counts = goldbach_module.goldbach_counts(1_000_000)
    index_of = {int(n): k for k, n in enumerate(evens)}
    for n in (1000, 10_000, 100_000, 1_000_000):
        ratio = int(counts[index_of[n]]) / goldbach_module.hardy_littlewood_estimate(n)
        assert 0.3 < ratio < 3.0, f"n={n} ratio={ratio}"


# ===========================================================================
# collatz.py -- total stopping times and peak values
# ===========================================================================

@pytest.fixture(scope="module")
def collatz_tables():
    import collatz
    STEP, MAXV = collatz.build_tables(100_000)
    return collatz, STEP, MAXV


def test_collatz_reference_total_stopping_times(collatz_tables):
    collatz, STEP, _ = collatz_tables
    expected = {1: 0, 2: 1, 3: 7, 6: 8, 7: 16, 9: 19, 18: 20, 25: 23, 27: 111}
    for n, tau in expected.items():
        assert collatz.total_stopping_time(n, STEP) == tau, f"tau({n})"


def test_collatz_max_value_of_27(collatz_tables):
    collatz, _, MAXV = collatz_tables
    assert collatz.max_value_reached(27, MAXV) == 9232


def test_collatz_every_n_reaches_one_modest_bound(collatz_tables):
    """Every n in [1, 100000] reaches 1 (finite check; NOT a proof)."""
    _, STEP, _ = collatz_tables
    assert np.count_nonzero(STEP[1:100001] < 0) == 0


def test_collatz_known_record_holder(collatz_tables):
    """n = 77031 is the classic record holder with tau = 350 below 100000."""
    collatz, STEP, _ = collatz_tables
    tau = STEP[1:100001].astype(np.int64)
    assert int(STEP[77031]) == 350
    assert int(tau.max()) == 350
    assert int(np.argmax(tau)) + 1 == 77031


# ===========================================================================
# ns2d_convergence.py -- 2D Navier-Stokes convergence study
# ===========================================================================

@pytest.fixture(scope="module")
def ns2d_module():
    import ns2d_convergence
    return ns2d_convergence


def test_ns2d_exact_solution_satisfies_equations(ns2d_module):
    """
    The Taylor-Green reference really solves the vorticity equation:
    omega_t = nu * Laplacian(omega) with Laplacian(omega) = -2 omega.
    """
    N, nu = 32, 0.1
    _, X, Y = ns2d_module.make_grid(N)
    # numeric omega_t via central time difference
    dt = 1e-6
    d_omega_dt = (ns2d_module.exact_vorticity(X, Y, dt, nu)
                  - ns2d_module.exact_vorticity(X, Y, -dt, nu)) / (2 * dt)
    lap = -2.0 * ns2d_module.exact_vorticity(X, Y, 0.0, nu)
    assert np.max(np.abs(d_omega_dt - nu * lap)) < 1e-9


def test_ns2d_temporal_order_at_least_3_5(ns2d_module):
    """RK4 temporal order at fixed N is ~4 (allow slack: assert >= 3.5)."""
    errs, orders = ns2d_module.temporal_convergence(
        N=16, nu=0.1, T=0.5, dts=(0.1, 0.05, 0.025)
    )
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), errs
    assert min(orders) >= 3.5, f"temporal orders {orders}"


def test_ns2d_spatial_error_decreases_with_resolution(ns2d_module):
    """Refining N must reduce the (self-convergence) error, monotonically."""
    errs = ns2d_module.spatial_convergence(Ns=(16, 32, 64), nu=0.05, T=0.1, dt=5e-3)
    ordered = [errs[(16, 32)], errs[(32, 64)]]
    assert ordered[0] > ordered[1], f"error did not decrease: {errs}"
    assert errs[(32, 64)] < 1e-5


def test_ns2d_enstrophy_decays_at_analytic_rate(ns2d_module):
    """Discrete enstrophy decays at rate 4*nu (here 0.4) to within 1%."""
    ts, zs, rate = ns2d_module.enstrophy_decay(N=16, nu=0.1, T=0.5, dt=1e-3)
    assert rate == pytest.approx(0.4, rel=0.01)
    assert zs[0] == pytest.approx(1.0, abs=1e-12)
    for t, z in zip(ts, zs):
        assert z == pytest.approx(ns2d_module.exact_enstrophy(t, 0.1), rel=2e-3)
