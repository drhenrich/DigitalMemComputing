"""
Where is sigma actually load-bearing?  The collinear points lie on the axis y=0,
where g_y = dOmega/dy ~ y = 0 -- so on-axis seeds reach L1/L2/L3 by pure x-descent
and sigma does nothing. sigma's real job shows up OFF-axis, where the unstable
y-direction is excited. This test seeds OFF-axis around a saddle and asks which
sigma-rule actually converts it into an attractor (raw dynamics, NO Newton polish).

modes:
  const-1  : s == -1 (no correction; natural descent -> saddle stays a saddle)
  const+1  : s == +1 (always flip)
  discrete : s = sign(-Omega_yy)            (current machine)
  mem_curv : smooth memory of the curvature  (rung 1; uses Omega_yy)
  memdyn   : Hessian-free memory from gradient history (rung 2)
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

def sim(mu, start, mode, gamma=0.3, alpha=8.0, eps_c=2.0, delta=1e-10,
        dt=0.01, max_steps=200000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu); y_prev, gy_prev = pos[1], gy
    s_peak = -1.0; conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        if   mode == "const-1":  s = -1.0
        elif mode == "const+1":  s = +1.0
        elif mode == "discrete": s = 1.0 if oyy < 0 else -1.0
        elif mode == "mem_curv":
            st = -np.tanh(oyy/eps_c); s += alpha*(st - s)*dt; s = max(-1, min(1, s))
        elif mode == "memdyn":
            dy = y - y_prev; dgy = gy - gy_prev
            c = dgy*dy/(dy*dy + delta); st = -np.tanh(c/eps_c)
            s += alpha*(st - s)*dt; s = max(-1, min(1, s))
        y_prev, gy_prev = y, gy
        s_peak = max(s_peak, s)
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: conv = step; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos, conv, s, s_peak

def test_point(mu, target, name, offsets):
    L = lpoints(mu); tgt = L[target]
    seeds = [tgt + np.array([dx, dy]) for dx in (-0.04, 0.0, 0.04) for dy in offsets]
    print(f"\n{name}  {target}={tgt.round(4)}  ({len(seeds)} off-axis seeds, raw dynamics)")
    for mode in ("const-1", "const+1", "discrete", "mem_curv", "memdyn"):
        hits = 0; peaks = []
        for sd in seeds:
            pt, conv, s_end, s_peak = sim(mu, sd, mode)
            if np.isfinite(pt).all() and np.linalg.norm(pt - tgt) < 0.02 and conv is not None:
                hits += 1; peaks.append(s_peak)
        pk = round(float(np.mean(peaks)), 2) if peaks else None
        print(f"   {mode:9s}: reached {target} from {hits}/{len(seeds)} seeds   "
              f"(mean peak s = {pk})")

if __name__ == "__main__":
    mu = 7.342e22/(5.972e24+7.342e22)
    print(f"Earth-Moon  mu={mu:.3e}")
    test_point(mu, "L1", "collinear saddle", offsets=(0.02, 0.05, 0.10, -0.05))
    test_point(mu, "L2", "collinear saddle", offsets=(0.02, 0.05, 0.10, -0.05))
    test_point(mu, "L4", "triangular point", offsets=(0.02, 0.05, -0.05, -0.10))
