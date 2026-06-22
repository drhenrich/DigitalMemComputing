"""
Robust grid: dense on-axis (y=0) seeds at Hill-radius multiples from the
secondary (for L1/L2) and near x=-1 (for L3), plus an off-axis layer for L4/L5.
Uses only structural knowledge (axis symmetry + Hill scaling), not the exact
L-point coordinates. Goal: 5/5 for ALL 23 systems.
"""
import numpy as np
from scipy.optimize import fsolve
from nbody_trojan import grad_curv, lpoints
EPS = 1e-9

BODIES = {
    "Sun":1.989e30,"Mercury":3.285e23,"Venus":4.867e24,"Earth":5.972e24,
    "Moon":7.342e22,"Mars":6.390e23,"Phobos":1.066e16,"Deimos":1.476e15,
    "Jupiter":1.898e27,"Io":8.932e22,"Europa":4.800e22,"Ganymede":1.482e23,
    "Callisto":1.076e23,"Saturn":5.683e26,"Titan":1.345e23,"Enceladus":1.080e20,
    "Rhea":2.307e21,"Uranus":8.681e25,"Titania":3.527e21,"Oberon":3.014e21,
    "Neptune":1.024e26,"Triton":2.139e22,"Pluto":1.303e22,"Charon":1.586e21}
SYSTEMS = [("Sun-Mercury","Sun","Mercury"),("Sun-Venus","Sun","Venus"),
    ("Sun-Earth","Sun","Earth"),("Sun-Mars","Sun","Mars"),("Sun-Jupiter","Sun","Jupiter"),
    ("Sun-Saturn","Sun","Saturn"),("Sun-Uranus","Sun","Uranus"),("Sun-Neptune","Sun","Neptune"),
    ("Sun-Pluto","Sun","Pluto"),("Earth-Moon","Earth","Moon"),("Mars-Phobos","Mars","Phobos"),
    ("Mars-Deimos","Mars","Deimos"),("Jupiter-Io","Jupiter","Io"),("Jupiter-Europa","Jupiter","Europa"),
    ("Jupiter-Ganymede","Jupiter","Ganymede"),("Jupiter-Callisto","Jupiter","Callisto"),
    ("Saturn-Titan","Saturn","Titan"),("Saturn-Enceladus","Saturn","Enceladus"),
    ("Saturn-Rhea","Saturn","Rhea"),("Uranus-Titania","Uranus","Titania"),
    ("Uranus-Oberon","Uranus","Oberon"),("Neptune-Triton","Neptune","Triton"),
    ("Pluto-Charon","Pluto","Charon")]

def grad_vec(p, mu):
    gx, gy, _ = grad_curv(p[0], p[1], mu); return np.array([gx, gy])
def hessian(p, mu):
    x, y = p
    r1 = np.sqrt((x+mu)**2+y**2)+EPS; r2 = np.sqrt((x-1+mu)**2+y**2)+EPS
    hxx = 1 - (1-mu)/r1**3 + 3*(1-mu)*(x+mu)**2/r1**5 - mu/r2**3 + 3*mu*(x-1+mu)**2/r2**5
    hyy = 1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5 - mu/r2**3 + 3*mu*y**2/r2**5
    hxy = 3*(1-mu)*(x+mu)*y/r1**5 + 3*mu*(x-1+mu)*y/r2**5
    return np.array([[hxx,hxy],[hxy,hyy]])

def simulate(mu, start, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=10.0,
             dt=0.01, max_steps=60000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); m = 0.0
    for _ in range(max_steps):
        x, y = pos; gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        m = min(m + beta*gn*dt, m_cap); g_eff = gamma0 + kappa*m
        sig = 1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - gx - g_eff*vel[0]; ay = -2*vel[0] + sig*gy - g_eff*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: return pos
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: return pos
    return pos

def polish(pos, mu):
    try:
        sol,_,ier,_ = fsolve(grad_vec, pos, args=(mu,), fprime=lambda p,mu:hessian(p,mu),
                             full_output=True, xtol=1e-12)
    except Exception: return pos, False
    if ier!=1 or np.linalg.norm(grad_vec(sol,mu))>1e-9: return pos, False
    return sol, True

def make_grid(mu):
    hill = (mu/3)**(1/3); sec = 1.0 - mu
    es = min(0.012, 0.3*hill); ep = min(0.03, 0.3*hill)
    starts = []
    # (A) collinear on-axis seeds (y=0): Hill-multiples around the secondary -> L1, L2
    for c in [0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0]:
        for xs in (sec - c*hill, sec + c*hill):
            if abs(xs-sec) > es and abs(xs+mu) > ep:
                starts.append((xs, 0.0))
    # (B) L3 region (opposite the primary), near x=-1
    for xs in (-1.0-5*mu/12, -0.96, -1.04, -1.10):
        starts.append((xs, 0.0))
    # (C) equilateral region for L4/L5: a small block around (1/2, +-sqrt3/2)
    for dx in (-0.12, -0.04, 0.04, 0.12):
        for sgn in (+1, -1):
            for dy in (-0.10, 0.0, 0.10):
                starts.append((0.5 + dx, sgn*(np.sqrt(3)/2) + dy))
    # (D) coarse off-axis fill (extra robustness / large-mu basins)
    for x in np.linspace(-1.3, 1.3, 6):
        for y in (0.9, 0.45, -0.45, -0.9):
            starts.append((x, y))
    return starts

def discover(mu):
    L = lpoints(mu); found = {k:0 for k in L}
    for s in make_grid(mu):
        endp = simulate(mu, s)
        pt, ok = (endp, False)
        if np.isfinite(endp).all():
            pol, k = polish(endp, mu)
            if k: pt, ok = pol, True
        if np.isfinite(pt).all():
            best, db = None, 1e9
            for kk, v in L.items():
                d = np.linalg.norm(pt-v)
                if d < db: db, best = d, kk
            if db < (0.05 if ok else 0.12): found[best] += 1
    return found

print(f"{'system':18} {'mu':>10}  found   L1 L2 L3 L4 L5")
print("-"*58)
fails=[]
for name,p1,p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2])
    f = discover(mu); nf = sum(1 for v in f.values() if v>0)
    if nf<5: fails.append(name)
    print(f"{name:18} {mu:10.2e}  {nf}/5    {f['L1']:2d} {f['L2']:2d} {f['L3']:2d} {f['L4']:2d} {f['L5']:2d}"
          + ("  <-- FAIL" if nf<5 else ""))
print("-"*58)
print(f"{len(SYSTEMS)-len(fails)}/{len(SYSTEMS)} find all 5." + (f"  FAILS: {fails}" if fails else "  ALL PASS"))
