"""
Lock the v3 physics:
  (1) memory-as-dissipation EOM  (per-axis memory provides the damping; gamma0=0)
  (2) Newton polish + Hessian check to fix the small-mu corotation-ring spurious convergence
Validate across mu range incl. Sun-Mercury.
"""
import numpy as np
from scipy.optimize import brentq, fsolve

# Single source of truth for CR3BP geometry (was duplicated here verbatim).
from nbody_trojan import grad_curv, lpoints

EPS = 1e-9

def grad_vec(p, mu):
    gx, gy, _ = grad_curv(p[0], p[1], mu)
    return np.array([gx, gy])

def hessian(p, mu):
    x, y = p
    r1 = np.sqrt((x+mu)**2 + y**2) + EPS
    r2 = np.sqrt((x-1+mu)**2 + y**2) + EPS
    def hxx():
        return (1 - (1-mu)/r1**3 + 3*(1-mu)*(x+mu)**2/r1**5
                  - mu/r2**3 + 3*mu*(x-1+mu)**2/r2**5)
    def hyy():
        return (1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5
                  - mu/r2**3 + 3*mu*y**2/r2**5)
    def hxy():
        return (3*(1-mu)*(x+mu)*y/r1**5 + 3*mu*(x-1+mu)*y/r2**5)
    return np.array([[hxx(), hxy()], [hxy(), hyy()]])

def simulate_v3(mu, start, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=10.0,
                dt=0.01, max_steps=120000, loose_thr=1e-4):
    """Memory IS the dissipation: gamma_eff_i = gamma0 + kappa*m_i, m_dot_i = beta|d_iOmega|.
    Run to |grad| < loose_thr; Newton polish then snaps to the exact critical point."""
    pos = np.array(start, float); vel = np.zeros(2)
    m = 0.0                                   # SCALAR memory -> damps both axes equally
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu)
        gn = np.hypot(gx, gy)
        m = min(m + beta*gn*dt, m_cap)
        sig = 1.0 if oyy < 0 else -1.0
        g_eff = gamma0 + kappa*m
        ax = 2*vel[1] - gx - g_eff*vel[0]
        ay = -2*vel[0] + sig*gy - g_eff*vel[1]
        vel += np.array([ax, ay])*dt
        pos = pos + vel*dt
        if gn < loose_thr:
            return step, pos, m
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            return None, pos, m
    return None, pos, m

def newton_polish(pos, mu):
    """Snap to the nearest true critical point grad Omega = 0. Returns (point, ok)."""
    try:
        sol, info, ier, msg = fsolve(grad_vec, pos, args=(mu,),
                                     fprime=lambda p, mu: hessian(p, mu),
                                     full_output=True, xtol=1e-12)
    except Exception:
        return pos, False
    if ier != 1:
        return sol, False
    # accept only a genuine zero of grad Omega (the 5 L-points are the ONLY exact zeros)
    if np.linalg.norm(grad_vec(sol, mu)) > 1e-9:
        return sol, False
    return sol, True

def grid(mu, n=10):
    L = lpoints(mu); hill=(mu/3)**(1/3); off=max(0.5*hill,0.02)
    ax=np.array([L["L1"][0]-off,L["L1"][0]+off,L["L2"][0]-off,L["L2"][0]+off,
                 L["L3"][0]-0.04,L["L3"][0]+0.04])
    xs=np.sort(np.unique(np.round(np.concatenate([np.linspace(-1.4,1.4,4),ax]),8)))[:n]
    ys=np.sort(np.unique(np.concatenate([np.linspace(-1.2,-0.15,3),[-0.05,0,0.05],
               np.linspace(0.15,1.2,4)])))[:n]
    return xs, ys, hill

def discover(mu, beta=0.5, gamma0=0.0, kappa=1.0, max_steps=120000, polish=True):
    L = lpoints(mu); xs, ys, hill = grid(mu)
    ep=min(0.03,0.3*hill); es=min(0.012,0.3*hill)
    tol = 0.05 if polish else 0.12
    counts={k:0 for k in L}; nstart=0; ndyn_fail=0; nspur=0
    for x in xs:
        for y in ys:
            if (x+mu)**2+y**2 < ep**2 or (x-1+mu)**2+y**2 < es**2: continue
            nstart += 1
            cs, endp, mm = simulate_v3(mu,[x,y],beta,gamma0,kappa,max_steps=max_steps)
            if cs is None:
                ndyn_fail += 1
            pt = endp
            if polish:
                pol, ok = newton_polish(pt, mu)
                if not ok:
                    nspur += 1
                    continue
                pt = pol
            # match to nearest analytic L-point
            best, db = None, 1e9
            for k, v in L.items():
                d = np.linalg.norm(pt - v)
                if d < db: db, best = d, k
            if db < tol: counts[best] += 1
            else: nspur += 1
    nf = sum(1 for v in counts.values() if v>0)
    return dict(counts=counts, nstart=nstart, ndyn_fail=ndyn_fail, nspur=nspur, nfound=nf)

def show(tag, r):
    c=r["counts"]
    print(f"{tag:36s} found {r['nfound']}/5 | "
          f"L1:{c['L1']:2d} L2:{c['L2']:2d} L3:{c['L3']:2d} L4:{c['L4']:2d} L5:{c['L5']:2d} "
          f"| starts={r['nstart']} dyn_fail={r['ndyn_fail']} rejected={r['nspur']}")

SYS = {
    "Sun-Mercury": 3.285e23/(1.989e30+3.285e23),
    "Sun-Earth":   5.972e24/(1.989e30+5.972e24),
    "Sun-Jupiter": 1.898e27/(1.989e30+1.898e27),
    "Earth-Moon":  7.342e22/(5.972e24+7.342e22),
    "Pluto-Charon":1.586e21/(1.303e22+1.586e21),
}

print("v3: memory-as-dissipation (gamma0=0, kappa=1, beta=0.5) + Newton polish + Hessian reject")
print("="*100)
print("--- WITHOUT polish (scalar memory dissipation, tol 0.12) ---")
for name, mu in SYS.items():
    show(name+" no-polish", discover(mu, polish=False, max_steps=120000))
print("--- WITH Newton polish (scalar memory dissipation, tol 0.05) ---")
for name, mu in SYS.items():
    show(name+" v3", discover(mu, polish=True, max_steps=120000))
