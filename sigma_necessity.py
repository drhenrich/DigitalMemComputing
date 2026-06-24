"""
Is sigma actually NECESSARY in the pipeline, or is it inert like the dissipation?

Two grids:
  STRUCTURAL  = build_grid (on-axis seeds for collinear -> gy=0 there, sigma idle;
                sigma=-1 at triangular anyway). Hypothesis: sigma is REDUNDANT here.
  GENERIC     = uniform off-axis grid (no symmetry crutch). Here the unstable
                y-direction IS excited near the saddles, so a position-dependent
                sigma should be REQUIRED; no constant should get all five.

modes: const-1, const+1, discrete sign(-Omega_yy), mem_curv (smooth memory, rung1).
Full machine, constant gamma=0.3, Newton polish ON (same labelling as the paper).
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

def sim(mu, start, mode, gamma=0.3, alpha=8.0, eps_c=2.0,
        dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0
    conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        if   mode == "const-1":  s = -1.0
        elif mode == "const+1":  s = +1.0
        elif mode == "discrete": s = 1.0 if oyy < 0 else -1.0
        elif mode == "mem_curv":
            st = -np.tanh(oyy/eps_c); s += alpha*(st - s)*dt; s = max(-1, min(1, s))
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: conv = step; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos

def generic_grid(mu):
    pts = []
    for x in np.linspace(-1.4, 1.4, 11):
        for y in np.linspace(-1.1, 1.1, 11):
            if abs(y) < 0.05:      # keep OFF the symmetry axis
                y = 0.08 if y >= 0 else -0.08
            pts.append((x, y))
    return pts

def discover(mu, mode, grid):
    L = lpoints(mu); found = {k: 0 for k in L}
    for (x, y) in grid:
        pt = sim(mu, [x, y], mode)
        if np.isfinite(pt).all():
            pol, ok = newton_polish(pt, mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d, lab = 1e9, None
            for k, v in L.items():
                dd = np.linalg.norm(pt - v)
                if dd < d: d, lab = dd, k
            if d < 0.05: found[lab] += 1
    return sum(1 for v in found.values() if v > 0), found

SYS = [("Earth-Moon", 7.342e22/(5.972e24+7.342e22)),
       ("Sun-Earth",  5.972e24/(1.989e30+5.972e24)),
       ("Sun-Jupiter", 1.898e27/(1.989e30+1.898e27)),
       ("Mars-Deimos", 1.476e15/(6.390e23+1.476e15))]

for label, gridfn in (("STRUCTURAL grid (on-axis seeds)", lambda mu: build_grid(mu, fill=6)),
                      ("GENERIC grid (off-axis uniform)",  generic_grid)):
    print(f"\n=== {label} ===")
    print(f"{'system':13} | {'const-1':>8} | {'const+1':>8} | {'discrete':>8} | {'mem_curv':>8}")
    for name, mu in SYS:
        row = []
        for mode in ("const-1", "const+1", "discrete", "mem_curv"):
            n, _ = discover(mu, mode, gridfn(mu)); row.append(f"{n}/5")
        print(f"{name:13} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8}")
