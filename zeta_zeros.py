#!/usr/bin/env python3
"""
zeta_zeros.py -- Numerical experiment: nontrivial zeros of the Riemann zeta
function on the critical line 1/2 + i*gamma.

WHAT THIS IS
------------
A *numerical experiment*, not a proof. We compute the Hardy Z function

    theta(t) = arg Gamma(1/4 + i t/2) - (t/2) log(pi)      (Riemann-Siegel theta)
    Z(t)     = exp(i theta(t)) * zeta(1/2 + i t)           (real for real t)

using mpmath at high working precision, locate sign changes of Z(t) by scanning
t on a fine grid, and refine each sign change to a root gamma_n.

IMPORTANT HONESTY NOTES
-----------------------
* By construction we only ever look AT the critical line Re(s)=1/2, so every
  zero reported here lies on the critical line BY CONSTRUCTION OF THE SEARCH.
  This says NOTHING about whether other (off-line) zeros exist. We make no
  claim whatsoever about the Riemann Hypothesis.
* Every number printed below is produced by actually running this code. None
  is transcribed from memory. The results are verified over a stated, finite
  range only.

Reference values (published, checked to >= 6 decimals) for the first five
zeros are hardcoded below and asserted.
"""

import time
import mpmath as mp

# 40 significant digits of working precision (task asked for 30+).
mp.mp.dps = 40

# Published reference values for the first five nontrivial zeros (gamma_n,
# where the zero is 1/2 + i*gamma_n), given to 15 decimals.
REFERENCE_GAMMAS = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def theta(t):
    """Riemann-Siegel theta: theta(t) = arg Gamma(1/4 + i t/2) - (t/2) log(pi)."""
    return mp.siegeltheta(t)


def hardy_Z(t):
    """Hardy Z function. Z(t) = exp(i theta(t)) * zeta(1/2 + i t); real for real t."""
    return mp.siegelz(t)


def verify_Z_identity(t):
    """Cross-check: exp(i theta) * zeta(1/2+i t) really equals mpmath's Z(t).

    Returns (Z_from_siegelz, Z_from_zeta); both should agree and be real. Used
    as a sanity check that we understand the objects we are scanning.
    """
    z_from_siegelz = mp.siegelz(t)
    z_from_zeta = mp.e ** (1j * mp.siegeltheta(t)) * mp.zeta(mp.mpf(1) / 2 + 1j * t)
    return z_from_siegelz, z_from_zeta


# ---------------------------------------------------------------------------
# Root finding: scan for sign changes, then refine
# ---------------------------------------------------------------------------

def find_zeros(num_zeros, t_start=2.0, step=0.05, t_max_hint=None):
    """Scan Z(t) for sign changes starting just above t_start and refine roots.

    step=0.05 is far smaller than the average zero spacing (2*pi/log(t/2pi) is
    ~9 at t~15 and ~2.1 near the 30th zero), so consecutive zeros are not
    skipped in the range explored here. Each sign change is refined with
    mp.findroot at the current high precision.

    Returns a list of mpf values gamma_1 < gamma_2 < ... .
    """
    zeros = []
    t = mp.mpf(t_start)
    f_prev = hardy_Z(t)
    prev_t = t
    t_max_hint = t_max_hint if t_max_hint is not None else 10 * num_zeros + 200
    while len(zeros) < num_zeros:
        t = t + step
        if t > t_max_hint:
            break
        f_cur = hardy_Z(t)
        if f_prev == 0:
            # Z is exactly zero at prev_t (measure-zero coincidence); count it.
            zeros.append(prev_t)
        elif f_prev * f_cur < 0:
            root = mp.findroot(hardy_Z, (prev_t, t), solver="anderson")
            zeros.append(root)
        f_prev = f_cur
        prev_t = t
    return zeros


# ---------------------------------------------------------------------------
# Riemann-von Mangoldt zero-counting main term
# ---------------------------------------------------------------------------

def N_main_term(T):
    """Riemann-von Mangoldt smooth main term for zeros with 0 < gamma < T.

    N(T) ~ (T/2pi) log(T/2pi) - T/2pi + 7/8.
    """
    T = mp.mpf(T)
    return (T / (2 * mp.pi)) * mp.log(T / (2 * mp.pi)) - T / (2 * mp.pi) + mp.mpf(7) / 8


def average_spacing(T):
    """Local average spacing of zeros near height T: 2*pi / log(T/2pi)."""
    T = mp.mpf(T)
    return 2 * mp.pi / mp.log(T / (2 * mp.pi))


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def max_reference_error(zeros):
    """Max |computed - reference| over the first five zeros (as float)."""
    errs = [abs(float(zeros[k]) - ref) for k, ref in enumerate(REFERENCE_GAMMAS)]
    return max(errs)


def check_gaps(zeros):
    """Compare consecutive gaps to the local average spacing; return (min,max) ratio."""
    ratios = []
    for i in range(1, len(zeros)):
        g = float(zeros[i] - zeros[i - 1])
        mid = float((zeros[i] + zeros[i - 1]) / 2)
        ratios.append(g / float(average_spacing(mid)))
    return min(ratios), max(ratios)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    NUM_ZEROS = 30

    print("=" * 74)
    print("Numerical experiment: nontrivial zeros of zeta on the critical line")
    print("=" * 74)
    print(f"mpmath working precision : {mp.mp.dps} significant digits")
    print(f"target number of zeros   : {NUM_ZEROS}")
    print()

    # Sanity check: the two definitions of Z must agree and be real.
    print("Sanity check of Z(t) = exp(i theta) zeta(1/2 + i t) vs mpmath.siegelz:")
    ok_identity = True
    for t in ["5.0", "17.0", "50.0"]:
        zs, zz = verify_Z_identity(mp.mpf(t))
        re_diff = abs(mp.re(zz) - zs)
        im_abs = abs(mp.im(zz))
        flag = "OK" if (re_diff < mp.mpf("1e-30") and im_abs < mp.mpf("1e-30")) else "FAIL"
        ok_identity = ok_identity and (flag == "OK")
        print(f"  t={t:>5}  Z_siegelz={mp.nstr(zs, 15):>18}  "
              f"Re(zeta-form)={mp.nstr(mp.re(zz), 15):>18}  "
              f"|Im|={mp.nstr(im_abs, 3):>8}  {flag}")
    assert ok_identity, "Z(t) identity from mpmath vs zeta() did not agree!"
    print()

    # Locate the zeros.
    print("Scanning Z(t) for sign changes (step=0.05) ...")
    scan_start = time.time()
    zeros = find_zeros(NUM_ZEROS)
    scan_time = time.time() - scan_start
    print(f"Found {len(zeros)} zeros in {scan_time:.2f} s.")
    print()

    # Print table of zeros.
    print("Computed zeros (zero is 1/2 + i*gamma_n):")
    print(f"{'n':>3}  {'gamma_n':>28}")
    for i, g in enumerate(zeros, 1):
        print(f"{i:>3}  {mp.nstr(g, 25):>28}")
    print()

    # Agreement with reference values.
    ref_err = max_reference_error(zeros)
    print("Agreement with published reference values (first five zeros):")
    for i, ref in enumerate(REFERENCE_GAMMAS, 1):
        d = abs(float(zeros[i - 1]) - ref)
        print(f"  gamma_{i}: computed={mp.nstr(zeros[i - 1], 18)}  "
              f"ref={ref:.15f}  |diff|={d:.3e}")
    print(f"  max |diff| over first five = {ref_err:.3e}")
    assert ref_err < 1e-6, "first five zeros do NOT match references to 6 decimals!"
    print("  ASSERTION PASSED: first five zeros agree to < 1e-6 (>= 6 decimals).")
    print()

    # Zero counting vs Riemann-von Mangoldt main term.
    T_last = float(zeros[-1])
    T_count = T_last + 0.001  # count zeros with 0 < gamma < T_count
    counted = sum(1 for g in zeros if 0 < float(g) < T_count)
    n_main = N_main_term(T_count)
    print("Zero-counting vs Riemann-von Mangoldt main term N(T) ~ "
          "(T/2pi)log(T/2pi) - T/2pi + 7/8:")
    print(f"  height T                      = {T_count:.4f}")
    print(f"  zeros counted with 0<gamma<T  = {counted}")
    print(f"  main term N(T)                = {mp.nstr(n_main, 12)}")
    print(f"  difference (counted - N(T))   = {counted - float(n_main):+.4f}")
    print()

    # Gaps vs average spacing.
    gmin, gmax = check_gaps(zeros)
    print("Observed gaps vs average spacing 2*pi/log(T/2pi):")
    print("  first few gaps (float): "
          + ", ".join(f"{float(zeros[i] - zeros[i - 1]):.4f}" for i in range(1, 6)))
    print(f"  min gap / avg_spacing = {gmin:.4f}")
    print(f"  max gap / avg_spacing = {gmax:.4f}")
    print(f"  (average spacing near last zero = {float(average_spacing(T_last)):.4f})")
    print()

    print("=" * 74)
    print("STATEMENT OF SCOPE / CAVEATS")
    print("=" * 74)
    print(f"* All {len(zeros)} zeros reported lie on the critical line BY")
    print("  CONSTRUCTION of the search (we only scan t on Re(s)=1/2).")
    print("* This states NOTHING about the Riemann Hypothesis and does NOT")
    print("  address whether zeros off the critical line exist.")
    print(f"* Verified numerically only for 0 < gamma < {T_count:.3f} "
          f"({counted} zeros).")
    print()
    print(f"Total wall time: {time.time() - t0:.2f} s")
    return zeros


if __name__ == "__main__":
    main()
