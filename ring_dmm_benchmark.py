"""
Restricted (N+1)-body ring problem: a rugged benchmark for memory-as-dissipation
================================================================================
A central star (M0=1) plus N planets on a co-rotating ring create a rugged
effective potential with many equilibria, saddles, maxima, and singular wells —
a landscape where Newton's basins are fractal and a generic start often diverges
or collides. We use it to test whether the paper's method earns its keep against
classical root-finding.

IMPORTANT (faithfulness): the solver here is the SAME method as the paper,
generalized to 2D. The paper flips the transverse force where Omega_yy<0
(sigma = sign(-Omega_yy)); in a general landscape the unstable direction is any
negative-curvature Hessian eigendirection, so we flip the gradient along those.
We do NOT precondition by 1/|lambda| (that would be damped Newton and would make
the memory decorative). Convergence is supplied by memory-controlled damping
gamma_eff = gamma0 + kappa*m, m_dot = beta|grad|  — memory as dissipation.
"""
import numpy as np
from scipy.optimize import fsolve

# ── System ──────────────────────────────────────────────────────────────────────
M0 = 1.0
N  = 8
MU = 0.08          # total ring mass / star mass
m  = MU / N
R  = 1.0
G  = 1.0
EPS = 1e-12

# exact co-rotating angular velocity of the rigid ring
csc_sum = sum(1.0/np.sin(np.pi*j/N) for j in range(1, N))
omega2  = G*M0/R**3 + G*m/(4.0*R**3)*csc_sum
omega   = np.sqrt(omega2)

PLANETS = np.array([[R*np.cos(2*np.pi*j/N), R*np.sin(2*np.pi*j/N)] for j in range(N)])


def grad_hess(x, y):
    """Gradient and Hessian of the effective potential
    Omega = 0.5*omega^2 r^2 + G M0/r + sum_j G m/|r-r_j|."""
    rs = np.sqrt(x*x + y*y) + EPS
    gx = omega2*x - G*M0*x/rs**3
    gy = omega2*y - G*M0*y/rs**3
    hxx = omega2 - G*M0/rs**3 + 3*G*M0*x*x/rs**5
    hyy = omega2 - G*M0/rs**3 + 3*G*M0*y*y/rs**5
    hxy = 3*G*M0*x*y/rs**5
    for (px, py) in PLANETS:
        dx, dy = x-px, y-py
        d = np.sqrt(dx*dx+dy*dy) + EPS
        gx -= G*m*dx/d**3
        gy -= G*m*dy/d**3
        hxx += -G*m/d**3 + 3*G*m*dx*dx/d**5
        hyy += -G*m/d**3 + 3*G*m*dy*dy/d**5
        hxy += 3*G*m*dx*dy/d**5
    return np.array([gx, gy]), np.array([[hxx, hxy],[hxy, hyy]])

def grad_vec(p):  return grad_hess(p[0], p[1])[0]
def hess_mat(p):  return grad_hess(p[0], p[1])[1]

def hits_singularity(p, eps_sun=0.04, eps_planet=0.02):
    if np.hypot(p[0], p[1]) < eps_sun: return True
    return bool(np.any(np.hypot(PLANETS[:,0]-p[0], PLANETS[:,1]-p[1]) < eps_planet))


# ── Solvers ───────────────────────────────────────────────────────────────────
def solve_newton(start, max_iter=100, tol=1e-9):
    p = np.array(start, float)
    for _ in range(max_iter):
        g, h = grad_hess(p[0], p[1])
        if np.linalg.norm(g) < tol:
            return p, True
        if hits_singularity(p) or np.linalg.norm(p) > 8:
            return p, False
        try:
            p = p + np.linalg.solve(h, -g)
        except np.linalg.LinAlgError:
            return p, False
        if not np.isfinite(p).all():
            return p, False
    return p, False


def solve_dmm(start, beta=0.5, gamma0=0.2, kappa=1.0, m_cap=3.0,
              dt=0.01, max_steps=15000, conv_thr=1e-4, polish=True):
    """Faithful memory-as-dissipation solver: curvature sign-flip (no 1/|lambda|)
    + memory-controlled damping. gamma0=0 -> memory is the sole dissipation."""
    pos = np.array(start, float); vel = np.zeros(2); mem = 0.0
    for _ in range(max_steps):
        g, h = grad_hess(pos[0], pos[1])
        gn = np.linalg.norm(g)
        if gn < conv_thr:
            if polish:
                try:
                    sol, _, ier, _ = fsolve(grad_vec, pos, fprime=hess_mat,
                                            full_output=True, xtol=1e-12)
                    if ier == 1 and np.linalg.norm(grad_vec(sol)) < 1e-9 \
                       and not hits_singularity(sol):
                        return sol, True
                except Exception:
                    pass
            return pos, True
        if hits_singularity(pos) or np.linalg.norm(pos) > 8:
            return pos, False
        # curvature sign-flip force: F = -sum_k sign(lambda_k) (g·e_k) e_k
        lam, V = np.linalg.eigh(h)
        F = -(np.sign(lam) * (V.T @ g)) @ V.T
        mem = min(mem + beta*gn*dt, m_cap)
        g_eff = gamma0 + kappa*mem
        ax = 2*omega*vel[1] + F[0] - g_eff*vel[0]
        ay = -2*omega*vel[0] + F[1] - g_eff*vel[1]
        vel += np.array([ax, ay])*dt
        pos += vel*dt
    return pos, False


# ── Ground truth + clustering ────────────────────────────────────────────────
def cluster(points, tol=2e-3):
    uniq = []
    for p in points:
        if not any(np.hypot(p[0]-u[0], p[1]-u[1]) < tol for u in uniq):
            uniq.append(p)
    return uniq

def classify(p):
    lam, _ = np.linalg.eigh(hess_mat(p))
    if lam[0] > 0 and lam[1] > 0: return "min"
    if lam[0] < 0 and lam[1] < 0: return "max"
    return "saddle"

def ground_truth(n=140, seed=0):
    """All equilibria via dense multi-start Newton from random seeds + clustering."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n*n):
        s = rng.uniform(-1.6, 1.6, 2)
        if hits_singularity(s): continue
        p, ok = solve_newton(s)
        if ok and not hits_singularity(p) and np.hypot(*p) < 3:
            found.append(p)
    return cluster(found)


if __name__ == "__main__":
    print(f"Ring system: N={N} planets, ring mu={MU:.1e}, omega={omega:.5f}")
    print("Establishing ground-truth equilibria (dense multi-start Newton)…")
    truth = ground_truth()
    by = {"min":0, "saddle":0, "max":0}
    for p in truth: by[classify(p)] += 1
    print(f"  {len(truth)} equilibria: {by['min']} minima, {by['saddle']} saddles, {by['max']} maxima")

    # Comparative benchmark from a uniform grid (same starts for both solvers)
    G_ = 25
    xs = np.linspace(-1.5, 1.5, G_); ys = np.linspace(-1.5, 1.5, G_)
    starts = [(x, y) for x in xs for y in ys if not hits_singularity((x, y))]
    print(f"\nBenchmark from {len(starts)} generic grid starts:")

    def coverage(found_pts):
        """fraction of ground-truth equilibria located (within 1e-2)."""
        hit = sum(1 for t in truth
                  if any(np.hypot(t[0]-f[0], t[1]-f[1]) < 1e-2 for f in found_pts))
        return hit, len(truth)

    res = {}
    for name, solver in [("Newton", solve_newton), ("DMM (memory-as-dissipation)", solve_dmm)]:
        ok_pts, n_ok, n_div = [], 0, 0
        for s in starts:
            p, ok = solver(s)
            if ok and not hits_singularity(p) and np.hypot(*p) < 3:
                ok_pts.append(p); n_ok += 1
            else:
                n_div += 1
        uniq = cluster(ok_pts)
        hit, tot = coverage(uniq)
        res[name] = dict(n_ok=n_ok, n_div=n_div, uniq=uniq, hit=hit)
        print(f"  {name:30s}: converged {n_ok:3d}/{len(starts)} | "
              f"distinct eq found {len(uniq):2d} | "
              f"ground-truth coverage {hit}/{tot} ({100*hit/tot:.0f}%) | diverged {n_div}")

    # ── Figure (white background, publication style) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams.update({"font.family":"serif","axes.linewidth":1.3,
                         "xtick.direction":"in","ytick.direction":"in"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, name in zip(axes, ["Newton", "DMM (memory-as-dissipation)"]):
        ax.set_aspect("equal")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(th), R*np.sin(th), ":", color="0.7", lw=0.8)
        ax.plot(0, 0, "o", color="#FDB813", ms=12, zorder=8)
        ax.plot(PLANETS[:,0], PLANETS[:,1], "o", color="#4fc3f7", ms=7, zorder=7)
        # ground-truth equilibria (open circles)
        for t in truth:
            ax.plot(*t, "o", mfc="none", mec="0.55", ms=11, mew=1.0, zorder=5)
        # found equilibria (filled stars)
        for f in res[name]["uniq"]:
            ax.plot(*f, "*", color="#2ca02c", ms=11, mec="k", mew=0.4, zorder=9)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        r = res[name]
        ax.set_title(f"{name}\nconverged {r['n_ok']}/{len(starts)} starts, "
                     f"found {r['hit']}/{len(truth)} equilibria", fontsize=10)
    axes[0].set_ylabel(r"$y$"); axes[0].set_xlabel(r"$x$"); axes[1].set_xlabel(r"$x$")
    fig.suptitle(f"Equilibria of the star + {N}-planet ring "
                 f"(open = all {len(truth)} equilibria, green ★ = found)", fontsize=12)
    plt.tight_layout()
    fig.savefig("fig_ring_benchmark.pdf", bbox_inches="tight")
    fig.savefig("fig_ring_benchmark.png", dpi=160, bbox_inches="tight")
    print("\n-> fig_ring_benchmark.pdf / .png")
