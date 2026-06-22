"""
Diagnostic for the four reviewer concerns:
  1. Sun-Mercury / Sun-Jupiter fail to find all 5 L-points with Table 3 params.
  2. Grid / trajectory-count confusion.
  3. β=0 (no memory) makes no difference  ->  is memory doing any computation?
  4. Robustness across the μ range.

Reproduces the EXACT app physics (copied verbatim from solar_system_dmm_v2.py).
"""
import numpy as np
from scipy.optimize import brentq

# Single source of truth for CR3BP geometry (was duplicated here verbatim).
from nbody_trojan import analytical_collinear, grad_curv

EPS = 1e-9
ROUTH = 0.03852

# Exact μ for the systems in question
SYSTEMS = {
    "Sun-Mercury": 3.285e23 / (1.989e30 + 3.285e23),
    "Sun-Earth":   5.972e24 / (1.989e30 + 5.972e24),
    "Sun-Jupiter": 1.898e27 / (1.989e30 + 1.898e27),
    "Sun-Saturn":  5.683e26 / (1.989e30 + 5.683e26),
    "Earth-Moon":  7.342e22 / (5.972e24 + 7.342e22),
    "Pluto-Charon":1.586e21 / (1.303e22 + 1.586e21),
}

def grad_and_curvature(pos, mu):
    """Return (grad=[gx,gy], Omega_yy). Delegates to nbody_trojan.grad_curv;
    kept as a pos-tuple adapter for the local simulate_dmm call sites."""
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    return np.array([gx, gy]), oyy

def simulate_dmm(mu, start, beta, mem_cap, gamma, dt, max_steps, conv_thr):
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    lmx = lmy = 1.0
    conv_step = None
    for step in range(max_steps):
        grad, oyy = grad_and_curvature(pos, mu)
        gx, gy = grad
        gn = np.linalg.norm(grad)
        lmx = min(lmx + beta*abs(gx)*dt, mem_cap)
        lmy = min(lmy + beta*abs(gy)*dt, mem_cap)
        sign_y = +1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - lmx*gx - gamma*vel[0]
        ay = -2*vel[0] + sign_y*lmy*gy - gamma*vel[1]
        vel += np.array([ax, ay]) * dt
        pos = pos + vel * dt
        if gn < conv_thr:
            conv_step = step
            break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return conv_step, pos, gn, (lmx, lmy)

def build_grid(mu, n_x=10, n_y=10):
    L1x, L2x, L3x = analytical_collinear(mu)
    hill_r = (mu/3)**(1/3)
    offset = max(0.5*hill_r, 0.02)
    anchors_x = np.array([L1x-offset, L1x+offset, L2x-offset, L2x+offset,
                          L3x-0.04, L3x+0.04])
    n_fill = max(n_x-6, 1)
    fill_x = np.linspace(-1.4, 1.4, n_fill)
    xs = np.sort(np.unique(np.round(np.concatenate([fill_x, anchors_x]), 8)))[:n_x]
    n_neg = (n_y-3)//2
    n_pos = n_y-3-n_neg
    ys = np.sort(np.unique(np.concatenate([
        np.linspace(-1.2,-0.15,max(n_neg,1)), [-0.05,0.0,0.05],
        np.linspace(0.15,1.2,max(n_pos,1))])))[:n_y]
    return xs, ys, hill_r

def run(mu, beta=0.001, mem_cap=8.0, gamma=0.6, dt=0.01,
        max_steps=200_000, conv_thr=1e-4, n_x=10, n_y=10):
    L1x, L2x, L3x = analytical_collinear(mu)
    analytic = {"L1":np.array([L1x,0.0]),"L2":np.array([L2x,0.0]),
                "L3":np.array([L3x,0.0]),"L4":np.array([0.5-mu,np.sqrt(3)/2]),
                "L5":np.array([0.5-mu,-np.sqrt(3)/2])}
    xs, ys, hill_r = build_grid(mu, n_x, n_y)
    excl_p = min(0.03, 0.3*hill_r)
    excl_s = min(0.012, 0.3*hill_r)
    counts = {k:0 for k in analytic}
    n_starts = 0; n_div = 0; n_nolabel = 0
    for x in xs:
        for y in ys:
            if (x+mu)**2 + y**2 < excl_p**2: continue
            if (x-1+mu)**2 + y**2 < excl_s**2: continue
            n_starts += 1
            conv, final, gn, lm = simulate_dmm(mu,[x,y],beta,mem_cap,gamma,dt,max_steps,conv_thr)
            if conv is None: n_div += 1
            label=None
            if np.isfinite(final).all():
                best,db=None,1e9
                for k,v in analytic.items():
                    d=np.linalg.norm(final-v)
                    if d<db: db,best=d,k
                if db<0.12: label=best
            if label: counts[label]+=1
            else: n_nolabel += 1
    n_found = sum(1 for v in counts.values() if v>0)
    return dict(counts=counts, n_starts=n_starts, n_div=n_div,
                n_nolabel=n_nolabel, n_found=n_found, hill_r=hill_r)

def fmt(r):
    c = r["counts"]
    return (f"starts={r['n_starts']:3d} | "
            f"L1:{c['L1']:2d} L2:{c['L2']:2d} L3:{c['L3']:2d} "
            f"L4:{c['L4']:2d} L5:{c['L5']:2d} | "
            f"found {r['n_found']}/5 | div={r['n_div']} nolabel={r['n_nolabel']}")

print("="*78)
print("CONCERN 1 & 4: Table-3 (default) params across systems")
print("  beta=0.001, mem_cap=8, gamma=0.6, dt=0.01, max_steps=2e5, thr=1e-4, 10x10")
print("="*78)
for name, mu in SYSTEMS.items():
    r = run(mu)
    print(f"{name:13s} mu={mu:.3e}  {fmt(r)}")

print()
print("="*78)
print("CONCERN 3: does memory (beta) matter?  Earth-Moon, gamma=0.6")
print("="*78)
mu = SYSTEMS["Earth-Moon"]
for beta in [0.0, 0.001, 0.01, 0.1, 1.0, 5.0]:
    r = run(mu, beta=beta, max_steps=80_000)
    print(f"beta={beta:6.3f} gamma=0.6  {fmt(r)}")

print()
print("="*78)
print("CONCERN 3b: memory WITHOUT damping (gamma=0). Does memory alone converge?")
print("  If memory is a real compute resource, beta>0 should converge where beta=0 fails")
print("="*78)
for gamma in [0.0]:
    for beta in [0.0, 0.001, 0.1, 1.0, 10.0]:
        r = run(mu, beta=beta, gamma=gamma, max_steps=80_000)
        print(f"gamma={gamma} beta={beta:6.3f}  {fmt(r)}")
