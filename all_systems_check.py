"""
Does the DMM v3 discovery find all 5 Lagrange points for EVERY two-body system?
Self-contained: imports the physics from nbody_trojan, copies the exact v3
discovery (simulate_v3 / newton_polish / run_discovery), runs all 23 systems.
"""
import numpy as np
from scipy.optimize import fsolve
from nbody_trojan import grad_curv, analytical_collinear, lpoints
EPS = 1e-9

BODIES = {
    "Sun":1.989e30,"Mercury":3.285e23,"Venus":4.867e24,"Earth":5.972e24,
    "Moon":7.342e22,"Mars":6.390e23,"Phobos":1.066e16,"Deimos":1.476e15,
    "Jupiter":1.898e27,"Io":8.932e22,"Europa":4.800e22,"Ganymede":1.482e23,
    "Callisto":1.076e23,"Saturn":5.683e26,"Titan":1.345e23,"Enceladus":1.080e20,
    "Rhea":2.307e21,"Uranus":8.681e25,"Titania":3.527e21,"Oberon":3.014e21,
    "Neptune":1.024e26,"Triton":2.139e22,"Pluto":1.303e22,"Charon":1.586e21,
}
SYSTEMS = [
    ("Sun-Mercury","Sun","Mercury"),("Sun-Venus","Sun","Venus"),
    ("Sun-Earth","Sun","Earth"),("Sun-Mars","Sun","Mars"),
    ("Sun-Jupiter","Sun","Jupiter"),("Sun-Saturn","Sun","Saturn"),
    ("Sun-Uranus","Sun","Uranus"),("Sun-Neptune","Sun","Neptune"),
    ("Sun-Pluto","Sun","Pluto"),("Earth-Moon","Earth","Moon"),
    ("Mars-Phobos","Mars","Phobos"),("Mars-Deimos","Mars","Deimos"),
    ("Jupiter-Io","Jupiter","Io"),("Jupiter-Europa","Jupiter","Europa"),
    ("Jupiter-Ganymede","Jupiter","Ganymede"),("Jupiter-Callisto","Jupiter","Callisto"),
    ("Saturn-Titan","Saturn","Titan"),("Saturn-Enceladus","Saturn","Enceladus"),
    ("Saturn-Rhea","Saturn","Rhea"),("Uranus-Titania","Uranus","Titania"),
    ("Uranus-Oberon","Uranus","Oberon"),("Neptune-Triton","Neptune","Triton"),
    ("Pluto-Charon","Pluto","Charon"),
]

def grad_vec(p, mu):
    gx, gy, _ = grad_curv(p[0], p[1], mu); return np.array([gx, gy])

def hessian(p, mu):
    x, y = p
    r1 = np.sqrt((x+mu)**2+y**2)+EPS; r2 = np.sqrt((x-1+mu)**2+y**2)+EPS
    hxx = 1 - (1-mu)/r1**3 + 3*(1-mu)*(x+mu)**2/r1**5 - mu/r2**3 + 3*mu*(x-1+mu)**2/r2**5
    hyy = 1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5 - mu/r2**3 + 3*mu*y**2/r2**5
    hxy = 3*(1-mu)*(x+mu)*y/r1**5 + 3*mu*(x-1+mu)*y/r2**5
    return np.array([[hxx,hxy],[hxy,hyy]])

def simulate_v3(mu, start, beta, gamma0, kappa, m_cap, dt, max_steps, conv_thr):
    pos = np.array(start, float); vel = np.zeros(2); m = 0.0
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        m = min(m + beta*gn*dt, m_cap); g_eff = gamma0 + kappa*m
        sig = 1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - gx - g_eff*vel[0]
        ay = -2*vel[0] + sig*gy - g_eff*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: return pos
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: return pos
    return pos

def newton_polish(pos, mu):
    try:
        sol, _, ier, _ = fsolve(grad_vec, pos, args=(mu,),
                                fprime=lambda p, mu: hessian(p, mu),
                                full_output=True, xtol=1e-12)
    except Exception:
        return pos, False
    if ier != 1 or np.linalg.norm(grad_vec(sol, mu)) > 1e-9:
        return pos, False
    return sol, True

def run_discovery(mu, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=10.0, dt=0.01,
                  max_steps=60000, conv_thr=1e-4, n_x=10, n_y=10, polish=True):
    L = lpoints(mu)
    hill = (mu/3)**(1/3); off = max(0.5*hill, 0.02)
    excl_p = min(0.03, 0.3*hill); excl_s = min(0.012, 0.3*hill)
    anchors = np.array([L["L1"][0]-off,L["L1"][0]+off,L["L2"][0]-off,L["L2"][0]+off,
                        L["L3"][0]-0.04,L["L3"][0]+0.04])
    xs = np.sort(np.unique(np.round(np.concatenate(
        [np.linspace(-1.4,1.4,max(n_x-6,1)), anchors],),8)))[:n_x]
    nn = (n_y-3)//2; npz = n_y-3-nn
    ys = np.sort(np.unique(np.concatenate(
        [np.linspace(-1.2,-0.15,max(nn,1)),[-0.05,0,0.05],np.linspace(0.15,1.2,max(npz,1))])))[:n_y]
    found = {k:0 for k in L}
    for x in xs:
        for y in ys:
            if (x+mu)**2+y**2 < excl_p**2 or (x-1+mu)**2+y**2 < excl_s**2: continue
            endp = simulate_v3(mu,[x,y],beta,gamma0,kappa,m_cap,dt,max_steps,conv_thr)
            pt = endp; polished = False
            if polish and np.isfinite(endp).all():
                pol, ok = newton_polish(endp, mu)
                if ok: pt, polished = pol, True
            if np.isfinite(pt).all():
                best, db = None, 1e9
                for k, v in L.items():
                    d = np.linalg.norm(pt-v)
                    if d < db: db, best = d, k
                if db < (0.05 if polished else 0.12): found[best] += 1
    return found

print(f"{'system':18} {'mu':>10}  {'found':>6}  {'L1 L2 L3 L4 L5':>16}")
print("-"*60)
fails = []
for name, p1, p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2])
    f = run_discovery(mu)
    nf = sum(1 for v in f.values() if v>0)
    flag = "" if nf==5 else "  <-- FAIL"
    if nf<5: fails.append((name, mu, nf, f))
    print(f"{name:18} {mu:10.2e}  {nf}/5    "
          f"{f['L1']:2d} {f['L2']:2d} {f['L3']:2d} {f['L4']:2d} {f['L5']:2d}{flag}")
print("-"*60)
print(f"{len(SYSTEMS)-len(fails)}/{len(SYSTEMS)} systems find all 5.")
if fails:
    print("FAILURES:", [(n, f"{nf}/5") for n,mu,nf,fd in fails])
