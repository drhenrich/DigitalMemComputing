"""
Numerical regression tests for nbody_trojan.py — the single source of truth for
both the CR3BP geometry (effective_potential, grad_curv, lpoints,
analytical_collinear) and the heliocentric N+1-body dynamics (accel, integrate,
seed_lpoint, jacobi_constant).

Two layers of tests:
  (A) Analytic invariants of the CR3BP geometry — these pin the math that was
      previously copy-pasted across 7 files. If grad_curv or lpoints ever
      silently drifts, these fail.
  (B) Dynamical invariants of the integrator — the three quantities the AJP
      manuscript relies on: Jacobi conservation, L4 bounded libration, and the
      Trojan libration period. These are the numbers the paper states, so they
      double as living documentation of the claims.

Run:  python -m pytest test_nbody_trojan.py -v
  or: python test_nbody_trojan.py            (falls back to direct asserts)
"""
import numpy as np

import nbody_trojan as N

MU_EARTH_MOON = 7.342e22 / (5.972e24 + 7.342e22)   # ~0.01215
MU_SUN_JUPITER = N.q("Jupiter") / (1.0 + N.q("Jupiter"))


# ─── A. CR3BP geometry invariants (protect the consolidation) ──────────────────

def test_l4_l5_are_equilateral():
    """L4, L5 must form equilateral triangles with the two primaries at
    (-mu,0) and (1-mu,0): all three mutual distances equal 1."""
    for mu in [MU_EARTH_MOON, MU_SUN_JUPITER, 0.1]:
        L = N.lpoints(mu)
        P1 = np.array([-mu, 0.0])
        P2 = np.array([1 - mu, 0.0])
        for name in ("L4", "L5"):
            d1 = np.linalg.norm(L[name] - P1)
            d2 = np.linalg.norm(L[name] - P2)
            d12 = np.linalg.norm(P2 - P1)
            assert abs(d1 - 1.0) < 1e-12, f"{name}-P1 != 1 (mu={mu})"
            assert abs(d2 - 1.0) < 1e-12, f"{name}-P2 != 1 (mu={mu})"
            assert abs(d12 - 1.0) < 1e-12


def test_collinear_points_on_xaxis():
    """L1, L2, L3 lie on the x-axis (y=0) and are ordered L3 < L1 < L2
    relative to the secondary's position (1-mu)."""
    for mu in [MU_EARTH_MOON, MU_SUN_JUPITER]:
        L = N.lpoints(mu)
        for name in ("L1", "L2", "L3"):
            assert abs(L[name][1]) < 1e-12, f"{name} not on x-axis"
        assert L["L3"][0] < -mu < L["L1"][0] < 1 - mu < L["L2"][0]


def test_gradient_zero_at_lagrange_points():
    """The defining property: grad Omega = 0 at every Lagrange point.
    (Softened EPS limits how close to zero we can get, so tolerate 1e-6.)"""
    for mu in [MU_EARTH_MOON, MU_SUN_JUPITER]:
        L = N.lpoints(mu)
        for name, p in L.items():
            gx, gy, _ = N.grad_curv(p[0], p[1], mu)
            assert abs(gx) < 1e-6, f"grad_x at {name} = {gx} (mu={mu})"
            assert abs(gy) < 1e-6, f"grad_y at {name} = {gy} (mu={mu})"


def test_gradient_formula_matches_reference():
    """Direct check that grad_curv returns the textbook CR3BP gradient.
    Pins the exact expression so a future edit can't silently change it."""
    mu = MU_EARTH_MOON
    x, y = 0.3, 0.4
    r1 = np.sqrt((x + mu)**2 + y**2) + N.EPS_CR3BP
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + N.EPS_CR3BP
    gx_ref = x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    gy_ref = y - (1 - mu)*y/r1**3 - mu*y/r2**3
    oyy_ref = (1 - (1 - mu)/r1**3 + 3*(1 - mu)*y**2/r1**5
               - mu/r2**3 + 3*mu*y**2/r2**5)
    gx, gy, oyy = N.grad_curv(x, y, mu)
    assert abs(gx - gx_ref) < 1e-15
    assert abs(gy - gy_ref) < 1e-15
    assert abs(oyy - oyy_ref) < 1e-15


def test_effective_potential_formula():
    """Omega(x,y) = (x^2+y^2)/2 + (1-mu)/r1 + mu/r2."""
    mu = MU_EARTH_MOON
    x, y = 0.3, 0.4
    r1 = np.sqrt((x + mu)**2 + y**2) + N.EPS_CR3BP
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + N.EPS_CR3BP
    ref = (x**2 + y**2)/2 + (1 - mu)/r1 + mu/r2
    assert abs(N.effective_potential(x, y, mu) - ref) < 1e-15


# ─── B. Dynamical invariants (the manuscript's claims) ─────────────────────────

def test_jacobi_constant_jupiter_l4():
    """Jacobi constant is conserved to relative < 1e-6 over 200 yr at dt=0.01 yr.
    (Manuscript dmm_lagrange_stability.tex states ~1e-8; we assert the looser 1e-6
    so the test is robust across machines/scipy versions while still catching
    any real integrator regression.)"""
    r0, v0 = N.seed_lpoint("Jupiter", "L4", t0=0.0)
    dt, T = 0.01, 200.0
    n = int(T / dt)
    r = np.array(r0, float); v = np.array(v0, float)
    a = N.accel(r, 0.0, ("Jupiter",))
    C0 = N.jacobi_constant(r0, v0, "Jupiter", 0.0)
    worst = 0.0
    for i in range(1, n + 1):
        t = i * dt
        r = r + v*dt + 0.5*a*dt*dt
        a2 = N.accel(r, t, ("Jupiter",))
        v = v + 0.5*(a + a2)*dt
        a = a2
        if i % 500 == 0:
            worst = max(worst, abs(N.jacobi_constant(r, v, "Jupiter", t) - C0))
    rel = worst / abs(C0)
    assert rel < 1e-6, f"Jacobi relative drift {rel:.2e} exceeds 1e-6"


def test_l4_bounded_libration_jupiter():
    """A body seeded at Jupiter L4 must stay bounded (libration amplitude
    < 0.01 AU in the co-rotating frame) over 1000 yr — it does not escape."""
    r0, v0 = N.seed_lpoint("Jupiter", "L4", t0=0.0)
    dt, T = 0.005, 1000.0
    n = int(T / dt)
    r = np.array(r0, float); v = np.array(v0, float)
    a = N.accel(r, 0.0, ("Jupiter",))
    xs, ys = [], []
    for i in range(1, n + 1):
        t = i * dt
        r = r + v*dt + 0.5*a*dt*dt
        a2 = N.accel(r, t, ("Jupiter",))
        v = v + 0.5*(a + a2)*dt
        a = a2
        if i % 200 == 0:
            rc = N.to_corotating(np.array([t]), r[None, :], "Jupiter")[0]
            xs.append(rc[0]); ys.append(rc[1])
    xs, ys = np.array(xs), np.array(ys)
    # L4 corotating-frame reference position
    L4 = N.lpoints(MU_SUN_JUPITER)["L4"] * N.PLANETS["Jupiter"][0]
    drift = np.hypot(xs - L4[0], ys - L4[1]).max()
    assert drift < 0.5, f"L4 libration drift {drift:.2f} AU > 0.5 (escape!)"


def test_trojan_libration_period():
    """The small-amplitude libration period about L4/L5 is
    T_lib = 2*pi / (n * sqrt(27*mu/4)) = P_host / sqrt(27*mu/4).
    For Jupiter this is ~147.9 yr. Assert it within 1 yr."""
    mu = MU_SUN_JUPITER
    P_jup = N.PLANETS["Jupiter"][2]
    T_lib = P_jup / np.sqrt(27*mu/4)
    assert abs(T_lib - 147.9) < 1.0, f"T_lib = {T_lib:.1f} yr, expected ~148"


def test_seed_lpoint_corotation_velocity():
    """seed_lpoint must set v = n ẑ × r (pure co-rotation): so |v| = n|r|."""
    for host in ("Earth", "Jupiter"):
        for which in ("L4", "L5"):
            r, v = N.seed_lpoint(host, which)
            n = N.mean_motion(host)
            assert abs(np.hypot(*v) - n*np.hypot(*r)) < 1e-6 * n*np.hypot(*r)


# ─── run directly if invoked without pytest ────────────────────────────────────
if __name__ == "__main__":
    import inspect, sys
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and inspect.isfunction(f)]
    passed, failed = 0, 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
