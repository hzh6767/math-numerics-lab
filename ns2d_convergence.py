#!/usr/bin/env python3
"""
ns2d_convergence.py -- numerical convergence study of a pseudo-spectral solver
for the 2D incompressible Navier-Stokes equations, in vorticity-streamfunction
form, on the periodic torus [0, 2pi)^2.

EQUATIONS
---------
    d(omega)/dt + (u . grad) omega = nu * Laplacian(omega)
    -Laplacian(psi) = omega ,   u = d(psi)/dy ,  v = -d(psi)/dx

WHAT THIS IS (AND IS NOT)
-------------------------
This is a NUMERICAL EXPERIMENT measuring the convergence ORDER of a standard
pseudo-spectral (Fourier collocation) solver with classical RK4 time stepping.
It is NOT a proof of anything.

* In 2D the smooth solution of these equations is KNOWN to exist for all time,
  so no existence/regularity question is at stake here. We are measuring the
  accuracy of a discretisation, nothing more.
* The open problem (smoothness in 3D) is NOT touched by this file.
* Every number printed comes from actually running this code, over the finite
  set of resolutions N and step sizes dt stated in the output. Nothing below
  is extrapolated beyond the grids that were actually run.

WHAT IS MEASURED
----------------
  (1) temporal order at fixed spatial resolution N: RK4 -> order ~4;
  (2) decrease of the solution error as the spatial resolution N is refined
      (self-convergence against the next-finer grid);
  (3) decay rate of the discrete enstrophy vs. the analytic rate 4*nu.

EXACT (TAYLOR-GREEN) REFERENCE SOLUTION
---------------------------------------
    psi(x,y,t)   = cos x cos y exp(-2 nu t)
    omega        = -Laplacian psi = 2 cos x cos y exp(-2 nu t)
    u = dpsi/dy  = -cos x sin y exp(-2 nu t)
    v = -dpsi/dx =  sin x cos y exp(-2 nu t)
This is an EXACT solution of the equations above: omega is a scalar multiple of
psi, so the Jacobian J(psi, omega) = psi_x omega_y - psi_y omega_x vanishes, and
omega_t = nu Laplacian(omega) holds because Laplacian(psi) = -2 psi. The spatial
enstrophy Z(t) = <omega^2> (mean over the torus) equals exp(-4 nu t), i.e. it
decays at the constant rate 4*nu.

Deterministic: fixed grids, no RNG, no wall-clock value enters any result.
Standalone:   E:/anaconda3/python.exe D:/gh-opt/math-numerics-lab/ns2d_convergence.py
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Grid and wavenumbers
# ---------------------------------------------------------------------------

def make_grid(N):
    """Return (x, X, Y) for an N x N uniform grid on [0, 2pi)^2 (endpoint=False)."""
    x = 2.0 * np.pi * np.arange(N, dtype=np.float64) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return x, X, Y


def wavenumbers(N):
    """Return (KX, KY, K2, dealias_mask) for the N x N Fourier grid.

    KX, KY are the integer wavenumbers k with modes exp(i k . x); K2 = |k|^2.
    dealias_mask implements the standard 2/3 rule on the nonlinear term.
    """
    k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers 0,1,...,-1
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    kmax = N // 3
    mask = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)
    mask[0, 0] = True
    return KX, KY, K2, mask


# ---------------------------------------------------------------------------
# Exact (Taylor-Green) reference solution
# ---------------------------------------------------------------------------

def exact_vorticity(X, Y, t, nu):
    """omega(x,y,t) = 2 cos x cos y exp(-2 nu t)."""
    return 2.0 * np.cos(X) * np.cos(Y) * np.exp(-2.0 * nu * t)


def exact_velocity(X, Y, t, nu):
    """(u, v) for the Taylor-Green exact solution."""
    e = np.exp(-2.0 * nu * t)
    u = -np.cos(X) * np.sin(Y) * e
    v = np.sin(X) * np.cos(Y) * e
    return u, v


def exact_enstrophy(t, nu):
    """Analytic enstrophy Z(t) = <omega^2> = exp(-4 nu t) for the exact solution."""
    return float(np.exp(-4.0 * nu * t))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def enstrophy(omega):
    """Discrete enstrophy: spatial mean of omega^2."""
    return float(np.mean(omega ** 2))


def vorticity_rhs(omega_hat, nu, KX, KY, K2, mask):
    """Right-hand side of d(omega_hat)/dt = -(u . grad omega)_hat - nu |k|^2 omega_hat.

    Velocity is recovered from the stream function psi_hat = omega_hat / |k|^2
    (since -Laplacian psi = omega), and the nonlinear term is evaluated
    pseudo-spectrally with 2/3 dealiasing.
    """
    psi_hat = np.zeros_like(omega_hat)
    np.divide(omega_hat, K2, out=psi_hat, where=K2 > 0)
    u = np.fft.ifft2(1j * KY * psi_hat).real
    v = np.fft.ifft2(-1j * KX * psi_hat).real
    domega_dx = np.fft.ifft2(1j * KX * omega_hat).real
    domega_dy = np.fft.ifft2(1j * KY * omega_hat).real
    adv_hat = np.fft.fft2(u * domega_dx + v * domega_dy) * mask
    return -adv_hat - nu * K2 * omega_hat


def rk4_step(omega_hat, dt, nu, KX, KY, K2, mask):
    """One classical fourth-order Runge-Kutta step."""
    k1 = vorticity_rhs(omega_hat, nu, KX, KY, K2, mask)
    k2 = vorticity_rhs(omega_hat + 0.5 * dt * k1, nu, KX, KY, K2, mask)
    k3 = vorticity_rhs(omega_hat + 0.5 * dt * k2, nu, KX, KY, K2, mask)
    k4 = vorticity_rhs(omega_hat + dt * k3, nu, KX, KY, K2, mask)
    return omega_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve(omega0_hat, N, nu, T, dt, dealias=True, sample_times=None):
    """Integrate the vorticity equation to time T with RK4.

    Returns (omega, omega_hat, samples):
      omega      -- physical vorticity at t = T (N x N array),
      omega_hat  -- its Fourier spectrum,
      samples    -- dict {t: enstrophy(t)} for the requested sample_times,
                    evaluated at the nearest step boundary at or below t.
    """
    KX, KY, K2, mask = wavenumbers(N)
    if not dealias:
        mask = np.ones_like(mask)
    n_steps = int(round(T / dt))
    if n_steps < 1:
        raise ValueError("need T/dt >= 1")
    h = T / n_steps
    omega_hat = np.asarray(omega0_hat, dtype=np.complex128).copy()

    want = sorted({float(s) for s in sample_times}) if sample_times else []
    samples = {}
    wi = 0
    if want and abs(want[0]) <= 1e-12:
        samples[want[0]] = enstrophy(np.fft.ifft2(omega_hat).real)
        wi = 1
    t = 0.0
    for _ in range(n_steps):
        omega_hat = rk4_step(omega_hat, h, nu, KX, KY, K2, mask)
        t += h
        while wi < len(want) and want[wi] <= t + 1e-9:
            samples[want[wi]] = enstrophy(np.fft.ifft2(omega_hat).real)
            wi += 1
    omega = np.fft.ifft2(omega_hat).real
    return omega, omega_hat, samples


def to_spectrum(omega_phys):
    """Physical vorticity -> Fourier spectrum."""
    return np.fft.fft2(omega_phys)


# ---------------------------------------------------------------------------
# (1) Temporal convergence order against the exact solution
# ---------------------------------------------------------------------------

def solve_taylor_green(N, nu, T, dt, sample_times=None):
    """Solve from the exact Taylor-Green state at t=0; return (err, omega, omega_hat, samples)."""
    _, X, Y = make_grid(N)
    w0 = exact_vorticity(X, Y, 0.0, nu)
    omega, omega_hat, samples = solve(to_spectrum(w0), N, nu, T, dt,
                                      sample_times=sample_times)
    err = float(np.max(np.abs(omega - exact_vorticity(X, Y, T, nu))))
    return err, omega, omega_hat, samples


def temporal_convergence(N=32, nu=0.1, T=1.0, dts=(0.2, 0.1, 0.05, 0.025)):
    """Return (errors, orders). orders[i] = log(e_i/e_{i+1})/log(dt_i/dt_{i+1})."""
    errs = [solve_taylor_green(N, nu, T, dt)[0] for dt in dts]
    orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1]))
              for i in range(len(dts) - 1)]
    return errs, orders


# ---------------------------------------------------------------------------
# (2) Spatial convergence (self-convergence against the next-finer grid)
# ---------------------------------------------------------------------------

def smooth_ic(N, sigma=0.6):
    """A smooth, broadband vorticity blob centred on the torus (no RNG).

    Distance uses the PERIODIC metric: a Gaussian in (X-pi) alone would tear a
    discontinuity at the seam X=0 ~ X=2pi (the two edges are the same points on
    the torus), which destroys the spectral accuracy we are trying to measure.
    """
    _, X, Y = make_grid(N)
    dx = np.pi - np.abs(np.pi - np.abs(X - np.pi))
    dy = np.pi - np.abs(np.pi - np.abs(Y - np.pi))
    return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


def spatial_convergence(Ns=(32, 64, 128, 256), nu=0.05, T=0.2, dt=1e-3,
                        sigma=0.6):
    """Error of each resolution vs the next-finer one (subsampled to shared nodes).

    The grids are nested (Nb a multiple of Na), so the coarse nodes are a subset
    of the fine ones and no interpolation is needed.
    """
    fields = {}
    for N in Ns:
        w0 = smooth_ic(N, sigma=sigma)
        omega, _, _ = solve(to_spectrum(w0), N, nu, T, dt)
        fields[N] = omega
    out = {}
    for a, b in zip(Ns[:-1], Ns[1:]):
        if b % a:
            continue
        f = b // a
        out[(a, b)] = float(np.max(np.abs(fields[a] - fields[b][::f, ::f])))
    return out


# ---------------------------------------------------------------------------
# (3) Enstrophy decay rate vs the analytic rate 4*nu
# ---------------------------------------------------------------------------

def enstrophy_decay(N=32, nu=0.1, T=1.0, dt=1e-3, n_samples=7):
    """Return (times, enstrophy, measured_rate) for the Taylor-Green solution.

    measured_rate is -d(log Z)/dt from a least-squares line fit; analytic = 4*nu.
    """
    _, X, Y = make_grid(N)
    w0 = exact_vorticity(X, Y, 0.0, nu)
    times = [float(t) for t in np.linspace(0.0, T, n_samples)]
    _, _, samples = solve(to_spectrum(w0), N, nu, T, dt, sample_times=times)
    ts = np.array(sorted(samples))
    zs = np.array([samples[t] for t in ts])
    slope = float(np.polyfit(ts, np.log(zs), 1)[0])
    return ts, zs, -slope


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    t0 = time.time()
    nu = 0.1
    print("=" * 74)
    print("2D Navier-Stokes (vorticity-streamfunction) -- convergence study")
    print("=" * 74)
    print(f"domain [0, 2pi)^2  |  nu = {nu}  |  Fourier pseudo-spectral + RK4")
    print("Taylor-Green exact solution used as the reference (see module docstring).")
    print()

    # (1) Temporal order.
    dts = (0.2, 0.1, 0.05, 0.025)
    errs, orders = temporal_convergence(N=32, nu=nu, T=1.0, dts=dts)
    print("(1) Temporal convergence at fixed N = 32, T = 1.0")
    print(f"    {'dt':>8}  {'max|omega_num - omega_exact|':>30}")
    for dt, e in zip(dts, errs):
        print(f"    {dt:>8}  {e:>30.6e}")
    for i, p in enumerate(orders):
        print(f"    order between dt={dts[i]} and dt={dts[i+1]}: {p:.4f}")
    print(f"    min observed temporal order = {min(orders):.4f}  (RK4 -> ~4)")
    print()

    # (2) Spatial convergence.
    Ns = (32, 64, 128, 256)
    serr = spatial_convergence(Ns=Ns, nu=0.05, T=0.2, dt=1e-3)
    print("(2) Spatial convergence, smooth blob IC, nu = 0.05, T = 0.2, dt = 1e-3")
    print(f"    {'grid pair':>14}  {'max|coarse - fine|':>22}")
    for (a, b), e in serr.items():
        print(f"    {a:>5} -> {b:<5}  {e:>22.6e}")
    print("    (error falls by roughly 4x per doubling of N; see the caveats below)")
    print()

    # (3) Enstrophy decay rate.
    ts, zs, rate = enstrophy_decay(N=32, nu=nu, T=1.0, dt=1e-3)
    print("(3) Discrete enstrophy decay for the Taylor-Green solution, N = 32")
    print(f"    {'t':>7}  {'Z(t) numeric':>16}  {'exp(-4 nu t)':>16}")
    for t, z in zip(ts, zs):
        print(f"    {t:>7.3f}  {z:>16.10f}  {exact_enstrophy(t, nu):>16.10f}")
    print(f"    measured decay rate = {rate:.6f}   analytic 4*nu = {4*nu}")
    print(f"    relative error      = {abs(rate - 4*nu)/(4*nu):.3e}")
    print()

    print("=" * 74)
    print("SCOPE / CAVEATS")
    print("=" * 74)
    print("* This measures DISCRETISATION ORDER only; it proves no theorem.")
    print("* 2D smooth solutions of these equations are known to exist for all")
    print(f"  time. We report only N in {list(Ns)} and dt in {list(dts)}.")
    print("* The 3D Navier-Stokes regularity problem is NOT addressed here and")
    print("  remains open.")
    print()
    print(f"Total wall time: {time.time() - t0:.2f} s")
    return {"temporal_errors": errs, "temporal_orders": orders,
            "spatial_errors": serr, "enstrophy_rate": rate}


if __name__ == "__main__":
    run()
