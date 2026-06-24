"""Which L-point does the Broyden machine miss at small mu? Print per-label counts."""
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

def sim(mu, start, gamma=0.3, alpha=15.0, eps_c=0.3, leak=2.0, dpmin=1e-4,
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
        s += alpha*(-np.tanh(B[1,1]/eps_c) - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        ax = 2*vel[1] - gx - gamma*vel[0]; ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos

def per_label(mu):
    L = lpoints(mu); found = {k: 0 for k in L}
    for (x, y) in build_grid(mu, fill=6):
        pt = sim(mu, [x, y])
        if np.isfinite(pt).all():
            pol, ok = newton_polish(pt, mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d, lab = 1e9, None
            for k, v in L.items():
                dd = np.linalg.norm(pt - v)
                if dd < d: d, lab = dd, k
            if d < 0.05: found[lab] += 1
    return found

for name, mu in [("Mars-Deimos", 1.476e15/(6.390e23+1.476e15)),
                 ("Sun-Pluto",   1.303e22/(1.989e30+1.303e22)),
                 ("Saturn-Enceladus", 1.080e20/(5.683e26+1.080e20))]:
    f = per_label(mu)
    missing = [k for k, v in f.items() if v == 0]
    print(f"{name:18} mu={mu:.1e}  counts={f}  MISSING={missing}")
