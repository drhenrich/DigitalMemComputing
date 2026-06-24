"""
Close the small-mu L4 gap in the Broyden machine. The L4/L5 asymmetry is intrinsic
(Coriolis breaks y->-y in the damped frame), so the fix is to make sigma robustly
default to the SAFE state s=-1 and flip only on a CLEARLY negative Hessian estimate:
    s_target = -tanh((B_yy - thr)/eps_c),  thr < 0   (flip needs B_yy < thr)
This protects triangular points and the corotation ridge (B_yy ~ 0) while still
flipping at genuine saddles (B_yy strongly negative). Sweep thr; report per-system
n/5 on the 9 previously-failing systems + 2 controls.  No analytic Hessian used.
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
build_grid = _ns["build_grid"]; newton_polish = _ns["newton_polish"]
grad_curv = _ns["grad_curv"]; lpoints = _ns["lpoints"]

def sim(mu, start, thr, gamma=0.3, alpha=15.0, eps_c=0.3, leak=2.0, dpmin=1e-4,
        dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    p_prev = pos.copy(); g_prev = np.array([gx, gy])
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
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos

def nfound(mu, thr):
    L = lpoints(mu); found = {k: 0 for k in L}
    for (x, y) in build_grid(mu, fill=6):
        pt = sim(mu, [x, y], thr)
        if np.isfinite(pt).all():
            pol, ok = newton_polish(pt, mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d, lab = 1e9, None
            for k, v in L.items():
                dd = np.linalg.norm(pt - v)
                if dd < d: d, lab = dd, k
            if d < 0.05: found[lab] += 1
    return sum(1 for v in found.values() if v > 0)

SYS = [("Sun-Mercury", 3.285e23/(1.989e30+3.285e23)),
       ("Sun-Venus",   4.867e24/(1.989e30+4.867e24)),
       ("Sun-Uranus",  8.681e25/(1.989e30+8.681e25)),
       ("Sun-Pluto",   1.303e22/(1.989e30+1.303e22)),
       ("Mars-Phobos", 1.066e16/(6.390e23+1.066e16)),
       ("Mars-Deimos", 1.476e15/(6.390e23+1.476e15)),
       ("Saturn-Encel",1.080e20/(5.683e26+1.080e20)),
       ("Uranus-Titan",3.527e21/(8.681e25+3.527e21)),
       ("Uranus-Ober", 3.014e21/(8.681e25+3.014e21)),
       ("Earth-Moon*", 7.342e22/(5.972e24+7.342e22)),   # control (was 5/5)
       ("Sun-Jupiter*",1.898e27/(1.989e30+1.898e27))]   # control (was 5/5)
THR = (0.0, -0.3, -0.6, -1.0)
print(f"{'system':14} | " + " | ".join(f"thr={t:>4}" for t in THR))
print("-"*52)
tally = {t: 0 for t in THR}
for name, mu in SYS:
    row = []
    for t in THR:
        n = nfound(mu, t); tally[t] += (n == 5); row.append(f"{n}/5")
    print(f"{name:14} | " + " | ".join(f"{c:>6}" for c in row))
print("-"*52)
print("5/5 count (of 11): " + " | ".join(f"thr={t}:{tally[t]}" for t in THR))
