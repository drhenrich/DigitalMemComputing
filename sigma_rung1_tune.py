"""
Close the loop on the user's question: make the smooth MEMORY-sigma (rung 1) a full
drop-in for the discrete switch. Diagnosis from sigma_necessity: at triangular points
Omega_yy = +2.25 always, so -tanh(2.25/eps_c) must saturate to ~-1 (need small eps_c)
while at saddles Omega_yy<0 saturates to +1. Sweep eps_c (and alpha) on the 4 systems,
structural grid + polish, looking for 5/5 on all four (matching discrete).
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

def sim(mu, start, mode, gamma=0.3, alpha=8.0, eps_c=0.5,
        dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        if mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        else:
            st = -np.tanh(oyy/eps_c); s += alpha*(st - s)*dt; s = max(-1, min(1, s))
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos

def discover(mu, mode, alpha=8.0, eps_c=0.5):
    L = lpoints(mu); found = {k: 0 for k in L}
    for (x, y) in build_grid(mu, fill=6):
        pt = sim(mu, [x, y], mode, alpha=alpha, eps_c=eps_c)
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

SYS = [("Earth-Moon", 7.342e22/(5.972e24+7.342e22)),
       ("Sun-Earth",  5.972e24/(1.989e30+5.972e24)),
       ("Sun-Jupiter", 1.898e27/(1.989e30+1.898e27)),
       ("Mars-Deimos", 1.476e15/(6.390e23+1.476e15))]

print("reference  discrete:        " +
      "  ".join(f"{n}:{discover(mu,'discrete')}/5" for n, mu in SYS))
print("memory-sigma rung1  s_dot = alpha*(-tanh(Omega_yy/eps_c) - s):")
for eps_c in (0.1, 0.2, 0.5, 1.0):
    for alpha in (8.0, 20.0):
        res = "  ".join(f"{n}:{discover(mu,'mem',alpha=alpha,eps_c=eps_c)}/5" for n, mu in SYS)
        print(f"  eps_c={eps_c:.1f} alpha={alpha:4.1f}:  {res}")
