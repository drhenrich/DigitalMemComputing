"""
Does the Broyden leak (Eq 9) matter, or is the plain Broyden update (Eq 8) enough?
leak=0 is pure Broyden (accumulates curvature over the whole path, never forgets).
We report (a) off-axis capture of the Earth-Moon saddle L1 and minimum L4, with the
FINAL learned B_yy vs the true Omega_yy at the target, and (b) full-pipeline 5/5 on a
few systems. eps_c=0.3, thr=-0.6, alpha=15, gamma=0.3 (paper params).
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
grad_curv = _ns["grad_curv"]; lpoints = _ns["lpoints"]
build_grid = _ns["build_grid"]; newton_polish = _ns["newton_polish"]

def sim(mu, start, leak, gamma=0.3, alpha=15.0, eps_c=0.3, thr=-0.6, dpmin=1e-4,
        dt=0.01, max_steps=200000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu); p_prev = pos.copy(); g_prev = np.array([gx, gy])
    conv = None
    for _ in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy); g = np.array([gx, gy])
        dp = pos - p_prev; dg = g - g_prev; nd = dp @ dp
        if nd > dpmin*dpmin:
            B = B + np.outer(dg - B @ dp, dp) / nd
            B = 0.5*(B + B.T); B = B + leak*(np.eye(2) - B)*dt
        s += alpha*(-np.tanh((B[1,1] - thr)/eps_c) - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        ax = 2*vel[1] - gx - gamma*vel[0]; ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: conv = True; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos, conv, B[1, 1]

mu = 7.342e22/(5.972e24+7.342e22); L = lpoints(mu)
print(f"Earth-Moon off-axis capture (raw dynamics).  true Ω_yy: L1={grad_curv(*L['L1'],mu)[2]:.2f}  L4={grad_curv(*L['L4'],mu)[2]:.2f}\n")
print(f"{'leak λ':>7} | {'L1 hits':>8} {'mean Byy@L1':>12} | {'L4 hits':>8} {'mean Byy@L4':>12}")
for leak in (0.0, 0.5, 2.0):
    row = {}
    for tgt, offs in (("L1",(0.02,0.05,0.10,-0.05)), ("L4",(0.02,0.05,-0.05,-0.10))):
        p = L[tgt]; seeds = [p+np.array([dx,dy]) for dx in (-0.04,0,0.04) for dy in offs]
        hits, byys = 0, []
        for sd in seeds:
            endp, conv, byy = sim(mu, sd, leak)
            if conv and np.isfinite(endp).all() and np.linalg.norm(endp-p) < 0.02:
                hits += 1; byys.append(byy)
        row[tgt] = (hits, len(seeds), np.mean(byys) if byys else float('nan'))
    print(f"{leak:>7.1f} | {row['L1'][0]:>3}/{row['L1'][1]:<3} {row['L1'][2]:>12.2f} | "
          f"{row['L4'][0]:>3}/{row['L4'][1]:<3} {row['L4'][2]:>12.2f}")

print("\nFull-pipeline 5/5 (structural grid + polish):")
SYS = [("Earth-Moon",7.342e22/(5.972e24+7.342e22)), ("Sun-Jupiter",1.898e27/(1.989e30+1.898e27)),
       ("Sun-Earth",5.972e24/(1.989e30+5.972e24)), ("Mars-Deimos",1.476e15/(6.390e23+1.476e15))]
def discover(mu, leak):
    Lp = lpoints(mu); found = {k:0 for k in Lp}
    for (x,y) in build_grid(mu, fill=6):
        endp,conv,_ = sim(mu,[x,y],leak,max_steps=120000)
        pt = endp
        if np.isfinite(endp).all():
            pol,ok = newton_polish(endp,mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d,lab = 1e9,None
            for k,v in Lp.items():
                dd = np.linalg.norm(pt-v)
                if dd<d: d,lab = dd,k
            if d<0.05: found[lab]+=1
    return sum(1 for v in found.values() if v>0)
print(f"{'system':13} | {'λ=0 (pure Broyden)':>20} | {'λ=2 (with leak)':>18}")
for name,m in SYS:
    print(f"{name:13} | {str(discover(m,0.0))+'/5':>20} | {str(discover(m,2.0))+'/5':>18}")
