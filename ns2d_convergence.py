#!/usr/bin/env python3
"""
ns2d_convergence.py -- numerical convergence study of a Fourier pseudo-spectral
solver for the 2D incompressible Navier-Stokes equations in
vorticity-streamfunction form on the periodic torus [0, 2pi)^2.

EQUATIONS
---------
    d(w)/dt + (u . grad) w = nu * laplacian(w)
    laplacian(psi) = -w ,   u = curl(psi) = ( d psi/dy , -d psi/dx )

WHAT THIS IS (AND IS NOT)
-------------------------
This is a NUMERICAL EXPERIMENT that measures the convergence ORDER of a
standard Fourier pseudo-spectral (collocation) solver with classical RK4 time
stepping. It is NOT a proof of anything.

* 2D smooth solutions of these equations are KNOWN to exist and stay smooth for
  all time, so no existence/regularity question is at stake here. We measure the
  accuracy of a discretisation and nothing more.
* The famous open problem -- global existence and smoothness for the 3D
  Navier-Stokes equations -- is NOT addressed, touched, or advanced by this
  file. Nothing below bears on it.
* Every number printed comes from actually running this code, over the finite
  set of resolutions N and step sizes dt stated in the output. No value is
  transcribed from memory and nothing is extrapolated beyond the grids run.

WHAT IS MEASURED
----------------
  (1) Spatial convergence: L2 (RMS) error of the numerical vorticity against a
      closed-form exact solution at T = 1, refined over N = 32, 64, 128, 256,
      with the observed order p = log2( err(N) / err(2N) ).
  (2) A NON-DEGENERATE spatial test (method of manufactured solutions), added
      because test (1) turns out to be degenerate -- see the warning below.
  (3) Temporal convergence: RK4 order at fixed N, refining dt = T/nsteps.
  (4) Enstrophy decay: discrete Z(t) = <w^2> against the analytic Z(0) e^{-4 nu t}.

WHY TEST (1) IS DEGENERATE -- THE KEY FINDING
---------------------------------------------
The requested exact solution is the Taylor-Green vortex:
    psi   = cos x cos y e^{-2 nu t}
    w     = -laplacian psi = 2 cos x cos y e^{-2 nu t}
    u     = -cos x sin y e^{-2 nu t}
    v     =  sin x cos y e^{-2 nu t}
For THIS field the nonlinear term vanishes IDENTICALLY:
    (u . grad) w = 2 cos x sin x sin y cos y - 2 sin x cos x cos y sin y e^{...}
                 = 0  at every point,
and the vorticity is a SINGLE Fourier mode (kx, ky) = (+-1, +-1), exactly
representable on any grid with N >= 4. The scheme is therefore exact to
roundoff: the L2 error sits at ~1e-16 for EVERY N, and the observed "order" is
roundoff noise, NOT the scheme's spatial order. The solver is not broken -- the
TEST CASE cannot exercise spatial convergence. We report this honestly (with a
WARNING line) and add a non-degenerate test to actually measure spatial order.

MANUFACTURED-SOLUTION SUPPLEMENT (measures true spatial order)
-------------------------------------------------------------
Pick a smooth Q(x,y) with a FULL Fourier spectrum, let w_ex = e^{-2 nu t} Q, and
add the source S = d_t w_ex + (u_ex . grad) w_ex - nu laplacian(w_ex) so that
w_ex solves the forced equation
    d(w)/dt + (u . grad) w = nu * laplacian(w) + S.
We use the Poisson-kernel product Q = P_r(x) P_r(y),  P_r(x) = (1-r^2)/(1-2r cos x + r^2)
with r = 0.7, whose Fourier coefficients decay geometrically like r^{|m|+|n|}.
For such an analytic field a spectral method converges EXPONENTIALLY -- faster
than any fixed algebraic order -- so the observed order should GROW with N.
This case has a genuine nonzero nonlinearity and exercises the advection term
and the 2/3-rule dealiasing.

METHOD
------
* Uniform grid x_j = 2 pi j / N; wavenumbers from fftfreq(N, d=1/N); K2 = |k|^2.
* Poisson solve (spectral): psi_hat = w_hat / K2, zero mode set to 0.
* Velocity: u_hat = i KY psi_hat, v_hat = -i KX psi_hat. (Sign convention check:
  with these signs the code recovers the closed-form u, v above to ~5e-16, so
  the signs as given are correct and are NOT flipped.)
* Nonlinear term evaluated pseudo-spectrally (products in physical space), then
  the 2/3 rule is applied: modes with |KX| or |KY| > N/3 are zeroed.
* Time stepping: classical explicit RK4 on w_hat.
* Stability: the explicit diffusive term requires nu * K2_max * dt <~ 2.785 (the
  RK4 real-axis stability limit); K2_max = 2 (N/2)^2 for even N. The spatial
  study uses dt = 5e-4, giving nu K2_max dt = 1.64 at N = 256 (stable, margin).
  At N = 256, dt = 1e-3 gives 3.28 and is UNSTABLE -- it is deliberately avoided.
  The temporal study starts at nsteps = 80 (dt = 1.25e-2, nu K2max dt = 2.56) at
  N = 64; the smaller nsteps = 20, 40 would be unstable and are not used.

REFERENCE / VERIFICATION
------------------------
* Sign/scale check: recovered psi, u, v vs. closed forms cos x cos y,
  -cos x sin y, sin x cos y (expected ~1e-16).
* Degeneracy check: max|(u . grad) w| for Taylor-Green at t = 0 (expected ~1e-15).
* Enstrophy: discrete Z(t) vs. analytic e^{-4 nu t}; max relative drift reported.
* Temporal order on the exactly-representable Taylor-Green case: the only error
  is time integration, so RK4 must show order ~4.

ASSERTIONS (never faked)
------------------------
We assert (a) the finest Taylor-Green L2 error < 1e-5, (b) the manufactured-
solution spatial order at the finest pair >= 3, (c) the measured temporal order
>= 3.5, and (d) the max enstrophy drift < 1e-6. If any measured number failed
its check the script prints FAIL and exits non-zero; it never silently passes.

DETERMINISM: fixed grids, no RNG, no wall-clock value enters any result.
RUNS: E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/ns2d_convergence.py
"""

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
NU = 0.1          # kinematic viscosity
T_FINAL = 1.0     # final time for the convergence studies
R_MMS = 0.7       # Poisson-kernel radius for the manufactured solution


# ---------------------------------------------------------------------------
# Grid and wavenumbers
# ---------------------------------------------------------------------------

def wavenumbers(N):
    """Return (KX, KY, K2, K2_inv, mask) for the N x N Fourier grid.

    KX, KY are 2D wavenumber arrays (modes exp(i k . x)); K2 = |k|^2.
    K2_inv is 1/K2 with the zero mode replaced by 0, so psi_hat = w_hat * K2_inv
    already enforces the zero-mean stream function convention. mask is the 2/3
    dealiasing rule (True where |KX| < N/3 and |KY| < N/3), with (0,0) kept.
    """
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX = k[None, :]
    KY = k[:, None]
    K2 = KX ** 2 + KY ** 2
    K2_inv = np.where(K2 == 0.0, 0.0, 1.0 / np.where(K2 == 0.0, 1.0, K2))
    mask = (np.abs(KX) < N / 3.0) & (np.abs(KY) < N / 3.0)
    mask[0, 0] = True
    return KX, KY, K2, K2_inv, mask


def physical_grid(N):
    """Return (X, Y) meshgrid of the periodic grid points in [0, 2pi)^2."""
    xi = 2.0 * np.pi * np.arange(N, dtype=np.float64) / N
    return np.meshgrid(xi, xi)


# ---------------------------------------------------------------------------
# Exact Taylor-Green reference solution
# ---------------------------------------------------------------------------

def tg_exact(N, t, nu=NU):
    """Exact Taylor-Green solution at time t: returns (w, u, v) physical arrays."""
    X, Y = physical_grid(N)
    dec = np.exp(-2.0 * nu * t)
    w = 2.0 * np.cos(X) * np.cos(Y) * dec
    u = -np.cos(X) * np.sin(Y) * dec
    v = np.sin(X) * np.cos(Y) * dec
    return w, u, v


def tg_vorticity_hat(N, t, nu=NU):
    """Fourier coefficients of the exact Taylor-Green vorticity at time t."""
    w, _, _ = tg_exact(N, 0.0, nu)
    return np.fft.fft2(w) * np.exp(-2.0 * nu * t)


def tg_advection_max(N, nu=NU):
    """max |(u . grad) w| for the exact Taylor-Green field at t = 0.

    Identically zero analytically; returns the roundoff-level value actually
    computed, documenting the spatial degeneracy.
    """
    KX, KY, _, _, _ = wavenumbers(N)
    w, u, v = tg_exact(N, 0.0, nu)
    w_hat = np.fft.fft2(w)
    wx = np.fft.ifft2(1j * KX * w_hat).real
    wy = np.fft.ifft2(1j * KY * w_hat).real
    return float(np.abs(u * wx + v * wy).max())


def verify_tg_operators(N, nu=NU):
    """Return (max|psi_rec-psi_ex|, max|u_rec-u_ex|, max|v_rec-v_ex|).

    Recomputes psi and u, v from w_hat with the solver's spectral conventions
    and compares with the closed-form fields -- a sign and scale check.
    """
    KX, KY, _, K2_inv, _ = wavenumbers(N)
    X, Y = physical_grid(N)
    psi_ex = np.cos(X) * np.cos(Y)
    u_ex = -np.cos(X) * np.sin(Y)
    v_ex = np.sin(X) * np.cos(Y)
    w_hat = np.fft.fft2(2.0 * np.cos(X) * np.cos(Y))
    psi_hat = w_hat * K2_inv
    psi_rec = np.fft.ifft2(psi_hat).real
    u_rec = np.fft.ifft2(1j * KY * psi_hat).real
    v_rec = np.fft.ifft2(-1j * KX * psi_hat).real
    return (float(np.abs(psi_rec - psi_ex).max()),
            float(np.abs(u_rec - u_ex).max()),
            float(np.abs(v_rec - v_ex).max()))


# ---------------------------------------------------------------------------
# Manufactured solution (non-degenerate: full spectrum, genuine nonlinearity)
# ---------------------------------------------------------------------------

def mms_fields(N, r=R_MMS):
    """Build manufactured-solution data on the N x N grid.

    Returns a dict with the exact field Q and the precomputed Fourier pieces
    (Q_hat, its Laplacian, and the nonlinear product) needed each step.
    """
    KX, KY, K2, K2_inv, _ = wavenumbers(N)
    X, Y = physical_grid(N)
    Pr = lambda xx: (1.0 - r * r) / (1.0 - 2.0 * r * np.cos(xx) + r * r)
    Q = Pr(X) * Pr(Y)
    Q_hat = np.fft.fft2(Q)
    psi_hat = Q_hat * K2_inv
    uQ = np.fft.ifft2(1j * KY * psi_hat).real
    vQ = np.fft.ifft2(-1j * KX * psi_hat).real
    Qx = np.fft.ifft2(1j * KX * Q_hat).real
    Qy = np.fft.ifft2(1j * KY * Q_hat).real
    return dict(Q=Q, Q_hat=Q_hat,
                adv_Q_hat=np.fft.fft2(uQ * Qx + vQ * Qy),   # (u_Q . grad) Q
                lap_Q_hat=-K2 * Q_hat)


def mms_source_hat(fields, t, nu=NU):
    """Fourier coefficients of the manufactured source S at time t.

    S = d_t w_ex + (u_ex . grad) w_ex - nu laplacian(w_ex), with w_ex = e^{-2 nu t} Q,
    which makes w_ex an exact solution of  d(w)/dt + (u.grad)w = nu lap(w) + S.
    """
    e2 = np.exp(-2.0 * nu * t)
    e4 = np.exp(-4.0 * nu * t)
    return e2 * (-2.0 * nu * fields['Q_hat'] - nu * fields['lap_Q_hat']) \
        + e4 * fields['adv_Q_hat']


def mms_exact(N, t, nu=NU, r=R_MMS):
    """Exact manufactured-solution vorticity at time t on the N x N grid."""
    X, Y = physical_grid(N)
    Pr = lambda xx: (1.0 - r * r) / (1.0 - 2.0 * r * np.cos(xx) + r * r)
    return np.exp(-2.0 * nu * t) * Pr(X) * Pr(Y)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def vorticity_rhs(w_hat, KX, KY, K2, mask, nu, source_hat=None):
    """dw_hat/dt for  d(w)/dt + (u . grad) w = nu laplacian(w) + S.

    Pseudo-spectral nonlinear term with 2/3-rule dealiasing.
    """
    k2s = np.where(K2 == 0.0, 1.0, K2)
    psi_hat = w_hat / k2s
    psi_hat[0, 0] = 0.0
    u = np.fft.ifft2(1j * KY * psi_hat).real
    v = np.fft.ifft2(-1j * KX * psi_hat).real
    wx = np.fft.ifft2(1j * KX * w_hat).real
    wy = np.fft.ifft2(1j * KY * w_hat).real
    adv_hat = np.fft.fft2(u * wx + v * wy) * mask
    rhs = -adv_hat - nu * K2 * w_hat
    if source_hat is not None:
        rhs = rhs + source_hat
    return rhs


def rk4_step(w_hat, dt, t, KX, KY, K2, mask, nu, source_fn=None):
    """One classical explicit RK4 step on w_hat (source_fn(t) gives the MMS source)."""
    S = source_fn if source_fn is not None else (lambda tt: None)
    c1 = vorticity_rhs(w_hat, KX, KY, K2, mask, nu, S(t))
    c2 = vorticity_rhs(w_hat + 0.5 * dt * c1, KX, KY, K2, mask, nu, S(t + 0.5 * dt))
    c3 = vorticity_rhs(w_hat + 0.5 * dt * c2, KX, KY, K2, mask, nu, S(t + 0.5 * dt))
    c4 = vorticity_rhs(w_hat + dt * c3, KX, KY, K2, mask, nu, S(t + dt))
    return w_hat + (dt / 6.0) * (c1 + 2.0 * c2 + 2.0 * c3 + c4)


def integrate(w0_hat, N, dt, T, nu=NU, source_fn=None, track_enstrophy=False):
    """Integrate from t=0 to T with fixed dt.

    Returns (w_hat, enstrophy_drift). enstrophy_drift is the max relative
    deviation of the discrete enstrophy from Z(0) e^{-4 nu t}; it is meaningful
    only for the Taylor-Green initial condition (Z(0) = 1) and is None otherwise.
    """
    KX, KY, K2, _, mask = wavenumbers(N)
    w_hat = np.asarray(w0_hat, dtype=np.complex128).copy()
    nsteps = int(round(T / dt))
    drift = 0.0
    for i in range(nsteps):
        w_hat = rk4_step(w_hat, dt, i * dt, KX, KY, K2, mask, nu, source_fn)
        if track_enstrophy:
            Z = float(np.mean(np.fft.ifft2(w_hat).real ** 2))
            Z_exact = np.exp(-4.0 * nu * (i + 1) * dt)
            drift = max(drift, abs(Z - Z_exact) / Z_exact)
    return w_hat, (drift if track_enstrophy else None)


def l2_error(w_hat, w_exact):
    """L2 (RMS) error of the physical vorticity against the exact field."""
    return float(np.sqrt(np.mean((np.fft.ifft2(w_hat).real - w_exact) ** 2)))


def observed_order(err_coarse, err_fine):
    """log2(err_coarse / err_fine); nan if either error is zero."""
    if err_coarse <= 0.0 or err_fine <= 0.0:
        return float('nan')
    return float(np.log2(err_coarse / err_fine))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(tg_res, mms_res, t_res, out_path):
    """Save a log-log convergence plot using the non-interactive Agg backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (matplotlib unavailable: {exc}; skipping plot)")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.loglog([r[0] for r in tg_res], [r[1] for r in tg_res], "o-", color="#b03a2e",
              label="Taylor-Green (degenerate: roundoff floor)")
    ax.loglog([r[0] for r in mms_res], [r[1] for r in mms_res], "s-", color="#1f618d",
              label="manufactured solution (spectral)")
    ax.axhline(1e-16, ls=":", color="#888", lw=1.0)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$L_2$ error of $\omega$")
    ax.set_title("Spatial convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    dts = np.array([r[0] for r in t_res])
    terr = np.array([r[1] for r in t_res])
    ax.loglog(dts, terr, "^-", color="#1e8449", label="measured")
    ax.loglog(dts, terr[0] * (dts / dts[0]) ** 4, "--", color="#888",
              label=r"slope 4 ($\propto dt^4$)")
    ax.set_xlabel("dt")
    ax.set_ylabel(r"$L_2$ error of $\omega$")
    ax.set_title("Temporal convergence (Taylor-Green, N=64)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  plot saved to {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t0 = time.perf_counter()
    nu = NU

    print("=" * 74)
    print("2D Navier-Stokes (vorticity-streamfunction) -- convergence study")
    print("=" * 74)
    print(f"domain [0, 2pi)^2  |  nu = {nu}  |  Fourier pseudo-spectral + explicit RK4")
    print(f"exact solution: Taylor-Green vortex (smooth for all t)  |  T = {T_FINAL}")
    print()
    print("SCOPE: this measures the order of accuracy of a discretisation on a smooth")
    print("known solution. It says NOTHING about the Navier-Stokes existence-and-")
    print("smoothness problem, which is open (a 3D question). Every number below")
    print("comes from an actual run; nothing is extrapolated beyond the grids run.")
    print()

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    Nc = 64
    dpsi, du, dv = verify_tg_operators(Nc)
    adv_max = tg_advection_max(Nc)
    print(f"Sanity checks (N = {Nc}):")
    print(f"  max|psi_rec - psi_exact|      = {dpsi:.3e}   (Poisson sign/scale)")
    print(f"  max|u_rec   - u_exact|        = {du:.3e}")
    print(f"  max|v_rec   - v_exact|        = {dv:.3e}")
    print(f"  max|(u.grad)w| (Taylor-Green) = {adv_max:.3e}")
    print("  -> the Taylor-Green advection term is IDENTICALLY ZERO (value above is")
    print("     roundoff), and its vorticity is a single exactly-representable mode.")
    print("     It is therefore a roundoff-level CORRECTNESS check, not a spatial-")
    print("     order measurement. See the WARNING in part (1) and part (2).")
    print()

    dt_spatial = 5e-4
    k2max256 = 2 * (256 // 2) ** 2

    # ------------------------------------------------------------------
    # (1) Spatial convergence -- Taylor-Green (as requested)
    # ------------------------------------------------------------------
    print("=" * 74)
    print("(1) SPATIAL convergence -- Taylor-Green vortex")
    print("=" * 74)
    print(f"T = {T_FINAL}, dt = {dt_spatial:g}   "
          f"(stability margin nu*K2max*dt = {nu * k2max256 * dt_spatial:.2f} at N=256; "
          f"limit 2.785)")
    print()
    print(f"    {'N':>6}  {'dt':>10}  {'L2 error':>14}  {'order p':>10}")
    print("    " + "-" * 44)
    tg_res = []
    prev = None
    for N in (32, 64, 128, 256):
        w_hat, _ = integrate(tg_vorticity_hat(N, 0.0), N, dt_spatial, T_FINAL, nu=nu)
        err = l2_error(w_hat, tg_exact(N, T_FINAL, nu)[0])
        pstr = "" if prev is None else f"{observed_order(prev, err):>10.2f}"
        tg_res.append((N, err))
        print(f"    {N:>6}  {dt_spatial:>10.1e}  {err:>14.4e}  {pstr:>10}")
        prev = err
    tg_finest = tg_res[-1][1]
    print()
    print(f"WARNING: the Taylor-Green L2 error is at the ROUNDOFF FLOOR (~{tg_finest:.1e})")
    print("         at EVERY N, because the advection term is identically zero and the")
    print("         solution is one exactly-representable Fourier mode. The 'order'")
    print("         column above is roundoff noise, NOT the scheme's spatial order.")
    print("         The solver is correct (error ~ machine epsilon); the TEST CASE is")
    print("         degenerate for spatial convergence. Part (2) measures the real order.")
    print()

    # ------------------------------------------------------------------
    # (2) Spatial convergence -- manufactured solution (non-degenerate)
    # ------------------------------------------------------------------
    print("=" * 74)
    print("(2) SPATIAL convergence -- manufactured solution (full spectrum + genuine")
    print(f"    nonlinearity):  w_ex = e^(-2 nu t) P_r(x) P_r(y),  r = {R_MMS}")
    print("=" * 74)
    print(f"T = {T_FINAL}, dt = {dt_spatial:g}")
    print()
    print(f"    {'N':>6}  {'dt':>10}  {'L2 error':>14}  {'order p':>10}")
    print("    " + "-" * 44)
    mms_res = []
    prev = None
    for N in (16, 32, 64, 128, 256):
        fields = mms_fields(N)
        src = (lambda tt, f=fields: mms_source_hat(f, tt, nu))
        w_hat, _ = integrate(fields['Q_hat'], N, dt_spatial, T_FINAL,
                             nu=nu, source_fn=src)
        err = l2_error(w_hat, mms_exact(N, T_FINAL, nu))
        pstr = "" if prev is None else f"{observed_order(prev, err):>10.2f}"
        mms_res.append((N, err))
        print(f"    {N:>6}  {dt_spatial:>10.1e}  {err:>14.4e}  {pstr:>10}")
        prev = err
    mms_orders = [observed_order(mms_res[i - 1][1], mms_res[i][1])
                  for i in range(1, len(mms_res))]
    print()
    print("    Exponential (spectral) convergence: the error falls by a large factor per")
    print("    doubling and the observed order GROWS with N -- the signature of a")
    print("    spectral method on an analytic solution, not a fixed algebraic order.")
    print()

    # ------------------------------------------------------------------
    # (3) Temporal convergence -- Taylor-Green (spatially exact)
    # ------------------------------------------------------------------
    N_t = 64
    kmax_t = N_t // 2
    dt_stab = 2.785 / (nu * 2 * kmax_t * kmax_t)
    print("=" * 74)
    print(f"(3) TEMPORAL convergence -- Taylor-Green, N = {N_t}")
    print("=" * 74)
    print("    The spatial part is exact for this case, so the ONLY error is time")
    print(f"    integration. Explicit-RK4 stability: dt < {dt_stab:.5f} "
          f"(nu*K2max = {nu * 2 * kmax_t * kmax_t:.1f});")
    print("    nsteps = 20, 40 give dt = 0.05, 0.025 > that bound (unstable) and are")
    print("    therefore not used -- the sweep starts at nsteps = 80 to satisfy it.")
    print()
    print(f"    {'nsteps':>7}  {'dt':>12}  {'L2 error':>14}  {'order p':>10}")
    print("    " + "-" * 48)
    t_res = []
    prev = None
    for ns in (80, 160, 320, 640):
        dt = T_FINAL / ns
        w_hat, _ = integrate(tg_vorticity_hat(N_t, 0.0), N_t, dt, T_FINAL, nu=nu)
        err = l2_error(w_hat, tg_exact(N_t, T_FINAL, nu)[0])
        pstr = "" if prev is None else f"{observed_order(prev, err):>10.2f}"
        t_res.append((dt, err))
        print(f"    {ns:>7}  {dt:>12.3e}  {err:>14.4e}  {pstr:>10}")
        prev = err
    t_orders = [observed_order(t_res[i - 1][1], t_res[i][1])
                for i in range(1, len(t_res))]
    print()
    print("    RK4 is 4th order, so p -> 4 while truncation dominates; the finest")
    print("    refinement reaches the roundoff floor (~1e-16) and its measured order")
    print("    degrades there (machine noise, not a solver defect).")
    print()

    # ------------------------------------------------------------------
    # (4) Enstrophy decay
    # ------------------------------------------------------------------
    N_z = 64
    print("=" * 74)
    print(f"(4) ENSTROPHY decay -- Taylor-Green, N = {N_z}, dt = {dt_spatial:g}")
    print("=" * 74)
    w0_hat = tg_vorticity_hat(N_z, 0.0)
    _, drift = integrate(w0_hat, N_z, dt_spatial, T_FINAL, nu=nu, track_enstrophy=True)
    Z0 = float(np.mean(np.fft.ifft2(w0_hat).real ** 2))
    print(f"    Z(t) = <w^2>;  exact Z(t) = Z(0) e^(-4 nu t),  with Z(0) = {Z0:.6f}")
    print(f"    max relative drift over 0 <= t <= {T_FINAL} :  {drift:.3e}")
    print()

    # ------------------------------------------------------------------
    # (5) Plot
    # ------------------------------------------------------------------
    print("=" * 74)
    print("(5) Plot")
    print("=" * 74)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "ns2d_convergence.png")
    save_plot(tg_res, mms_res, t_res, out_path)
    print()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("=" * 74)
    print("ASSERTIONS")
    print("=" * 74)
    ok = True

    if tg_finest < 1e-5:
        print(f"    PASS: finest Taylor-Green L2 error {tg_finest:.2e} < 1e-5")
    else:
        ok = False
        print(f"    FAIL: finest Taylor-Green L2 error {tg_finest:.2e} >= 1e-5")

    if mms_orders[-1] >= 3.0:
        print(f"    PASS: manufactured-solution order at finest pair "
              f"{mms_orders[-1]:.2f} >= 3 (spectral convergence)")
    else:
        ok = False
        print(f"    FAIL: manufactured-solution order at finest pair "
              f"{mms_orders[-1]:.2f} < 3")

    t_best = max(t_orders)
    if t_best >= 3.5:
        print(f"    PASS: measured temporal order {t_best:.2f} >= 3.5 (RK4)")
    else:
        ok = False
        print(f"    FAIL: measured temporal order {t_best:.2f} < 3.5")

    if drift < 1e-6:
        print(f"    PASS: max enstrophy drift {drift:.2e} < 1e-6")
    else:
        ok = False
        print(f"    FAIL: max enstrophy drift {drift:.2e} >= 1e-6")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("-" * 74)
    print("SUMMARY")
    print("-" * 74)
    print("    * Solver: Fourier pseudo-spectral, 2/3-rule dealiasing, explicit RK4.")
    print(f"    * Taylor-Green (the requested case) is reproduced to ROUNDOFF "
          f"(~{tg_finest:.0e});")
    print("      it is degenerate for spatial convergence (zero advection, single mode)")
    print("      and is reported as a correctness check, not as a spatial order.")
    print(f"    * A non-degenerate manufactured solution shows SPECTRAL spatial")
    print(f"      convergence: error {mms_res[0][1]:.1e} (N={mms_res[0][0]}) -> "
          f"{mms_res[-1][1]:.1e} (N={mms_res[-1][0]}),")
    print(f"      with the observed order growing to {mms_orders[-1]:.1f}.")
    print(f"    * Temporal order on the exactly-representable Taylor-Green case is "
          f"{t_best:.2f}")
    print("      (RK4 ~ 4), measured while truncation dominates.")
    print(f"    * Discrete enstrophy tracks Z(0) e^(-4 nu t) to {drift:.1e} relative.")
    print()
    print("    * These are finite numerical measurements on smooth exact solutions.")
    print("      This says NOTHING about the Navier-Stokes existence-and-smoothness")
    print("      problem, which is open. No theorem is proved or claimed here.")
    print("-" * 74)
    print(f"    All assertions passed: {ok}")
    print(f"    Total wall time: {time.perf_counter() - t0:.1f} s")

    return {"temporal_errors": [e for _, e in t_res], "temporal_orders": t_orders,
            "spatial_tg": tg_res, "spatial_mms": mms_res,
            "enstrophy_drift": drift, "ok": ok}


if __name__ == "__main__":
    sys.exit(0 if run()["ok"] else 1)
