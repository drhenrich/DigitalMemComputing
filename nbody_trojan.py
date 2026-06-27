"""
Restricted (N+1)-body core for Trojan / Lagrange-point stability.
Massless test particles integrated in the time-dependent field of the Sun +
planets (planets on prescribed circular, coplanar orbits; they are NOT perturbed
by the test particles). Units: AU, years, solar masses; G = 4π².

This is the *dynamical* complement to the v3 DMM: v3 locates the equilibria
∇Ω=0; here we forward-integrate test particles to see which survive under the
real time-dependent many-planet field.
"""
import numpy as np

GM_SUN = 4.0 * np.pi**2            # AU^3 / yr^2  (G·M_sun with M_sun=1)
M_SUN_KG = 1.989e30
ROUTH = 0.03852

# name: (a_AU, mass_kg, period_yr, mean_longitude_J2000_deg, color)
PLANETS = {
    "Mercury": (0.38710, 3.285e23,   0.24085, 252.25, "#b5b5b5"),
    "Venus":   (0.72333, 4.867e24,   0.61520, 181.98, "#e8cda0"),
    "Earth":   (1.00000, 5.972e24,   1.00000, 100.46, "#4fc3f7"),
    "Mars":    (1.52368, 6.390e23,   1.88085, 355.43, "#c1440e"),
    "Jupiter": (5.20260, 1.898e27,  11.8618,   34.40, "#c88b3a"),
    "Saturn":  (9.55491, 5.683e26,  29.4571,   49.94, "#e4d191"),
    "Uranus":  (19.2184, 8.681e25,  84.0205,  313.23, "#80d8e8"),
    "Neptune": (30.1104, 1.024e26, 164.770,   304.88, "#3f54ba"),
}

def q(name):       # planet/Sun mass ratio
    return PLANETS[name][1] / M_SUN_KG

def mean_motion(name):
    a = PLANETS[name][0]
    # heliocentric two-body mean motion: n^2 = GM_sun(1+q)/a^3
    return 2*np.pi * np.sqrt(1.0 + q(name)) * a**(-1.5)   # rad/yr

def planet_angle(name, t):
    a, m, per, L0, c = PLANETS[name]
    return np.radians(L0) + mean_motion(name)*t

def planet_pos(name, t):
    a = PLANETS[name][0]; th = planet_angle(name, t)
    return np.array([a*np.cos(th), a*np.sin(th)])


# ── CR3BP geometry (normalized rotating frame) ─────────────────────────────────
# Single source of truth for the circular restricted 3-body effective potential
# Omega(x,y) and its equilibria. All DMM/figure/diagnostic scripts import from
# here rather than re-deriving these expressions (they had drifted across five
# independent copies). Softening EPS matches the historical value used in every
# script except ring_dmm_benchmark.py (which deliberately uses 1e-12).
#
# Convention: primaries at (-mu, 0) [mass 1-mu] and (1-mu, 0) [mass mu]; unit
# separation, unit angular velocity. Omega = (x^2+y^2)/2 + (1-mu)/r1 + mu/r2.
EPS_CR3BP = 1e-9


def effective_potential(x, y, mu):
    """CR3BP effective potential Omega(x,y) in the co-rotating frame.

    Vectorized over array x, y. Uses softening EPS_CR3BP to avoid the
    singularity at the primaries."""
    r1 = np.sqrt((x + mu)**2 + y**2) + EPS_CR3BP
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + EPS_CR3BP
    return (x**2 + y**2)/2 + (1 - mu)/r1 + mu/r2


def grad_curv(x, y, mu):
    """Gradient of Omega and the yy-curvature Omega_yy at (x, y).

    Returns (gx, gy, oyy). gx, gy are the components of -nabla Omega (the
    direction of the conservative force in the rotating frame); oyy = Omega_yy
    is the curvature used by the DMM to decide the sign of the y-axis memory
    current (sigma = sign(-oyy)). Vectorized over x, y."""
    r1 = np.sqrt((x + mu)**2 + y**2) + EPS_CR3BP
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + EPS_CR3BP
    gx = x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    gy = y - (1 - mu)*y/r1**3 - mu*y/r2**3
    oyy = (1 - (1 - mu)/r1**3 + 3*(1 - mu)*y**2/r1**5
           - mu/r2**3 + 3*mu*y**2/r2**5)
    return gx, gy, oyy


def analytical_collinear(mu):
    """x-positions of the three collinear Lagrange points L1, L2, L3.

    L1 lies between the primaries, L2 beyond the secondary (mass mu), L3 beyond
    the primary (mass 1-mu). Found by bracketed root-finding on gx(x,0,mu)=0."""
    from scipy.optimize import brentq
    def gx0(x):
        r1 = abs(x + mu) + EPS_CR3BP
        r2 = abs(x - 1 + mu) + EPS_CR3BP
        return x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    L1 = brentq(gx0, -mu + 1e-4, 1 - mu - 1e-4)
    L2 = brentq(gx0, 1 - mu + 1e-4, 2.5)
    L3 = brentq(gx0, -2.5, -mu - 1e-4)
    return L1, L2, L3


def lpoints(mu):
    """The five Lagrange points of the CR3BP as a dict {name: np.array([x,y])}.

    L1,L2,L3 are collinear (on the x-axis); L4,L5 are the equilateral points."""
    L1x, L2x, L3x = analytical_collinear(mu)
    return {"L1": np.array([L1x, 0.]), "L2": np.array([L2x, 0.]),
            "L3": np.array([L3x, 0.]),
            "L4": np.array([0.5 - mu,  np.sqrt(3)/2]),
            "L5": np.array([0.5 - mu, -np.sqrt(3)/2])}


def accel(r, t, perturbers):
    """Heliocentric acceleration on a test particle at r (AU) at time t (yr).
    Includes the indirect term (Sun's reflex acceleration from each planet),
    which is essential for the Lagrange points to be true equilibria."""
    rn = (r[0]*r[0] + r[1]*r[1])**1.5 + 1e-12
    a = -GM_SUN * r / rn                                   # Sun (direct)
    for name in perturbers:
        gm = GM_SUN*q(name)
        rp = planet_pos(name, t)
        dr = rp - r
        dn = (dr[0]*dr[0] + dr[1]*dr[1])**1.5 + 1e-12
        rpn = (rp[0]*rp[0] + rp[1]*rp[1])**1.5 + 1e-12
        a = a + gm*dr/dn - gm*rp/rpn                       # direct + indirect
    return a


def integrate(r0, v0, t_max, dt, perturbers, sample=20):
    """Velocity-Verlet (leapfrog) integration. Returns (times, traj Nx2)."""
    r = np.array(r0, float); v = np.array(v0, float)
    n = int(t_max/dt)
    times = [0.0]; traj = [r.copy()]
    a = accel(r, 0.0, perturbers)
    for i in range(1, n+1):
        t = i*dt
        r = r + v*dt + 0.5*a*dt*dt
        a_new = accel(r, t, perturbers)
        v = v + 0.5*(a + a_new)*dt
        a = a_new
        if i % sample == 0:
            times.append(t); traj.append(r.copy())
        if not np.isfinite(r).all() or np.hypot(*r) > 60:   # ejected
            times.append(t); traj.append(r.copy()); break
    return np.array(times), np.array(traj)


def seed_lpoint(host, which, t0=0.0, dr_au=0.0, dphi_deg=0.0):
    """Inertial (r,v) for a test particle at the host's L-point, co-rotating at n_host.
    dr_au / dphi_deg apply a small radial / angular perturbation."""
    a = PLANETS[host][0]; n = mean_motion(host); th = planet_angle(host, t0)
    if which in ("L4", "L5"):
        sign = +1 if which == "L4" else -1
        ang = th + sign*np.radians(60.0)
        rad = a
    else:
        # collinear: normalized root x_L (planet at 1-mu) -> physical radius a*x_L
        mu = q(host)/(1+q(host))
        from scipy.optimize import brentq
        def g0(x):
            r1 = abs(x+mu)+1e-12; r2 = abs(x-1+mu)+1e-12
            return x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        xL = {"L1": brentq(g0,-mu+1e-4,1-mu-1e-4),
              "L2": brentq(g0,1-mu+1e-4,2.5),
              "L3": brentq(g0,-2.5,-mu-1e-4)}[which]
        # Heliocentric coordinate relative to the Sun is x_helio = xL + mu
        x_helio = xL + mu
        ang = th if x_helio > 0 else th + np.pi
        rad = a*abs(x_helio)
    rad += dr_au
    ang += np.radians(dphi_deg)
    r = np.array([rad*np.cos(ang), rad*np.sin(ang)])
    # velocity of a point co-rotating at rate n about the Sun:  v = n ẑ × r
    v = n * np.array([-r[1], r[0]])
    return r, v


def to_corotating(times, traj, host):
    """Rotate inertial trajectory into the frame co-rotating with the host planet."""
    out = np.empty_like(traj)
    for i, t in enumerate(times):
        th = planet_angle(host, t)
        c, s = np.cos(-th), np.sin(-th)
        out[i,0] = c*traj[i,0] - s*traj[i,1]
        out[i,1] = s*traj[i,0] + c*traj[i,1]
    return out


def accel_cloud(R, t, perturbers):
    """Vectorized heliocentric acceleration for a cloud R of shape (P,2)."""
    rn = (R[:,0]**2 + R[:,1]**2)[:,None]**1.5 + 1e-12
    A = -GM_SUN * R / rn
    for name in perturbers:
        gm = GM_SUN*q(name)
        rp = planet_pos(name, t)                      # (2,)
        dr = rp[None,:] - R                            # (P,2)
        dn = (dr[:,0]**2 + dr[:,1]**2)[:,None]**1.5 + 1e-12
        rpn = (rp[0]**2 + rp[1]**2)**1.5 + 1e-12
        A = A + gm*dr/dn - gm*rp[None,:]/rpn
    return A

def integrate_cloud(R0, V0, t_max, dt, perturbers, host, sample=40):
    """Vectorized leapfrog for a cloud. Returns times, traj (S,P,2),
    and survived mask (P,) — survived = stayed co-orbital (radius within 25% of a_host)."""
    R = np.array(R0, float); V = np.array(V0, float)
    a_host = PLANETS[host][0]
    n = int(t_max/dt)
    times = [0.0]; traj = [R.copy()]
    survived = np.ones(R.shape[0], bool)
    alive_hist = [1.0]
    A = accel_cloud(R, 0.0, perturbers)
    for i in range(1, n+1):
        t = i*dt
        R = R + V*dt + 0.5*A*dt*dt
        A2 = accel_cloud(R, t, perturbers)
        V = V + 0.5*(A + A2)*dt
        A = A2
        rad = np.hypot(R[:,0], R[:,1])
        left = (np.abs(rad - a_host)/a_host > 0.25) | (rad > 60) | ~np.isfinite(rad)
        survived &= ~left
        if i % sample == 0:
            times.append(t); traj.append(R.copy()); alive_hist.append(survived.mean())
    return np.array(times), np.array(traj), survived, np.array(alive_hist)

def seed_cloud(host, which, nP, dr_spread=0.0, dphi_spread=3.0, t0=0.0, seed=0):
    """A cloud of nP test particles around the host's L-point."""
    rng = np.random.default_rng(seed)
    R0 = np.zeros((nP,2)); V0 = np.zeros((nP,2))
    for k in range(nP):
        dr = rng.normal(0, dr_spread) if dr_spread > 0 else 0.0
        dphi = rng.normal(0, dphi_spread) if dphi_spread > 0 else 0.0
        r, v = seed_lpoint(host, which, t0=t0, dr_au=dr, dphi_deg=dphi)
        R0[k] = r; V0[k] = v
    return R0, V0


def jacobi_constant(r, v, host, t):
    """Jacobi constant in the Sun+host rotating frame (only meaningful for
    perturbers=[host]). Should be ~conserved along a trajectory."""
    n = mean_motion(host)
    rp = planet_pos(host, t)
    r1 = np.hypot(*r) + 1e-12
    r2 = np.hypot(*(r-rp)) + 1e-12
    # rotating-frame speed: v_rot = v - n ẑ × r
    v_rot = v - n*np.array([-r[1], r[0]])
    Omega = 0.5*n*n*(r[0]**2+r[1]**2) + GM_SUN/r1 + GM_SUN*q(host)/r2
    return 2*Omega - (v_rot[0]**2 + v_rot[1]**2)
