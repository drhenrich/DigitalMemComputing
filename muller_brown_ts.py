"""
Müller-Brown transition-state search — honest proof-of-concept.
Does the memory-as-dissipation saddle flow find index-1 saddles as robustly as
the standard Hessian-based TS method (partitioned RFO)?  No claim is made until
the numbers say so.

Method under test (faithful to the paper, generalized to find index-1 saddles):
  gentlest-ascent force  F = -grad V + 2 (grad V . v_min) v_min
  (reverse the force along the lowest Hessian eigenvector -> index-1 saddle is an
   attractor; at the RTBP collinear points the lowest mode IS the negative-
   curvature y-direction, so this reduces to the paper's sigma-flip there),
  second-order dynamics with MEMORY-controlled damping:
     x'' = F - gamma_eff x',  m' = beta|grad V|,  gamma_eff = gamma0 + kappa m.
  gamma0 = 0  ->  memory is the sole dissipation.

Baseline: partitioned RFO (P-RFO), the gold-standard Hessian-based saddle search.
"""
import numpy as np
from scipy.optimize import fsolve

# ── Müller-Brown potential ────────────────────────────────────────────────────
A  = np.array([-200., -100., -170., 15.])
aa = np.array([-1., -1., -6.5, 0.7])
bb = np.array([0., 0., 11., 0.6])
cc = np.array([-10., -10., -6.5, 0.7])
X0 = np.array([1., 0., -0.5, -1.])
Y0 = np.array([0., 0.5, 1.5, 1.])

def V(x, y):
    dx, dy = x-X0, y-Y0
    return np.sum(A*np.exp(aa*dx*dx + bb*dx*dy + cc*dy*dy))

def grad(p):
    x, y = p; dx, dy = x-X0, y-Y0
    E = A*np.exp(aa*dx*dx + bb*dx*dy + cc*dy*dy)
    gx = np.sum(E*(2*aa*dx + bb*dy))
    gy = np.sum(E*(bb*dx + 2*cc*dy))
    return np.array([gx, gy])

def hess(p):
    x, y = p; dx, dy = x-X0, y-Y0
    E = A*np.exp(aa*dx*dx + bb*dx*dy + cc*dy*dy)
    px = 2*aa*dx + bb*dy
    py = bb*dx + 2*cc*dy
    hxx = np.sum(E*(px*px + 2*aa))
    hyy = np.sum(E*(py*py + 2*cc))
    hxy = np.sum(E*(px*py + bb))
    return np.array([[hxx, hxy],[hxy, hyy]])

DOM = dict(xlo=-1.6, xhi=1.1, ylo=-0.4, yhi=2.1)

def in_domain(p):
    return DOM["xlo"]<=p[0]<=DOM["xhi"] and DOM["ylo"]<=p[1]<=DOM["yhi"]

def index_of(p):
    lam = np.linalg.eigvalsh(hess(p))
    return int(np.sum(lam < 0))   # 0 = min, 1 = saddle, 2 = max

# ── Ground-truth critical points (refine literature values) ───────────────────
def refine(p0):
    sol, _, ier, _ = fsolve(grad, p0, fprime=hess, full_output=True, xtol=1e-12)
    return sol if ier == 1 and np.linalg.norm(grad(sol)) < 1e-8 else None

MIN_GUESS = [(-0.558, 1.442), (0.623, 0.028), (-0.050, 0.467)]
SAD_GUESS = [(-0.822, 0.624), (0.212, 0.293)]
MINIMA  = [refine(p) for p in MIN_GUESS]
SADDLES = [refine(p) for p in SAD_GUESS]


# ── Solvers ───────────────────────────────────────────────────────────────────
def dmm_gad(start, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=4.0,
            dt=0.002, max_steps=60000, thr=1e-3):
    """Memory-as-dissipation gentlest-ascent flow -> index-1 saddle."""
    pos = np.array(start, float); vel = np.zeros(2); mem = 0.0
    nev = 0
    for _ in range(max_steps):
        g = grad(pos); H = hess(pos); nev += 1
        gn = np.linalg.norm(g)
        if gn < thr:
            sol, _, ier, _ = fsolve(grad, pos, fprime=hess, full_output=True, xtol=1e-12)
            if ier == 1 and np.linalg.norm(grad(sol)) < 1e-8 and in_domain(sol):
                return sol, index_of(sol), nev
            return None, None, nev
        lam, Vv = np.linalg.eigh(H)
        vmin = Vv[:, 0]                          # lowest-curvature eigenvector
        F = -g + 2*(g @ vmin)*vmin               # gentlest-ascent force
        mem = min(mem + beta*gn*dt, m_cap)
        ge = gamma0 + kappa*mem
        vel += (F - ge*vel)*dt
        pos = pos + vel*dt
        if not in_domain(pos) or not np.isfinite(pos).all():
            return None, None, nev
    return None, None, nev

def prfo(start, max_iter=400, trust=0.15, tol=1e-8):
    """Partitioned RFO: Hessian-based gold-standard index-1 saddle search."""
    p = np.array(start, float); nev = 0
    for _ in range(max_iter):
        g = grad(p); H = hess(p); nev += 1
        if np.linalg.norm(g) < tol:
            return p, index_of(p), nev
        lam, Vv = np.linalg.eigh(H); gp = Vv.T @ g
        l1, g1 = lam[0], gp[0]                    # maximize lowest mode
        lp = 0.5*(l1 + np.sqrt(l1*l1 + 4*g1*g1))
        d0 = -g1/(l1 - lp) if abs(l1-lp) > 1e-12 else 0.0
        l2, g2 = lam[1], gp[1]                    # minimize the other mode
        ln = 0.5*(l2 - np.sqrt(l2*l2 + 4*g2*g2))
        d1 = -g2/(l2 - ln) if abs(l2-ln) > 1e-12 else 0.0
        d = Vv @ np.array([d0, d1])
        nd = np.linalg.norm(d)
        if nd > trust: d = d*trust/nd
        p = p + d
        if not in_domain(p) or not np.isfinite(p).all():
            return None, None, nev
    return None, None, nev


# ── Benchmark ─────────────────────────────────────────────────────────────────
def which_saddle(p):
    if p is None: return None
    for i, s in enumerate(SADDLES):
        if np.linalg.norm(p - s) < 0.03:
            return i
    return None

def run(label, solver, starts):
    found = set(); n_saddle = 0; n_min = 0; n_fail = 0; tot_ev = 0
    for s in starts:
        p, idx, nev = solver(s); tot_ev += nev
        if p is None: n_fail += 1; continue
        if idx == 1:
            n_saddle += 1
            w = which_saddle(p)
            if w is not None: found.add(w)
        elif idx == 0:
            n_min += 1
        else:
            n_fail += 1
    print(f"  {label:28s}: reached a saddle {n_saddle:3d}/{len(starts)} | "
          f"distinct saddles {len(found)}/2 | fell to a min {n_min:3d} | "
          f"failed {n_fail:3d} | mean evals {tot_ev/len(starts):.0f}")
    return found

if __name__ == "__main__":
    print("Müller-Brown surface:")
    print(f"  minima  (index 0): {[tuple(np.round(m,3)) for m in MINIMA]}")
    print(f"  saddles (index 1): {[tuple(np.round(s,3)) for s in SADDLES]}")
    print(f"  saddle indices verified: {[index_of(s) for s in SADDLES]}")

    G = 26
    xs = np.linspace(DOM['xlo']+0.1, DOM['xhi']-0.1, G)
    ys = np.linspace(DOM['ylo']+0.1, DOM['yhi']-0.1, G)
    starts = [(x, y) for x in xs for y in ys]
    print(f"\nTS search from {len(starts)} generic grid starts (goal: find both index-1 saddles):")
    run("P-RFO (Hessian gold std)", prfo, starts)
    run("DMM-GAD  gamma0=0 (memory)", lambda s: dmm_gad(s, gamma0=0.0), starts)
    run("DMM-GAD  gamma0=0.3",       lambda s: dmm_gad(s, gamma0=0.3), starts)
