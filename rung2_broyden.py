"""
Rung 2, attempt 3 (predictive, Hessian-free): a Broyden/quasi-Newton MEMORY.

The machine evaluates g = grad Omega every step anyway. The Jacobian of g is the
Hessian H. A memory matrix B accumulates an estimate of H from consecutive gradient
samples (Broyden 'good' update), with a mild leak toward I so it tracks the LOCAL
curvature. No analytic second derivative, no extra evaluations.

    on each step with dp = p_k - p_{k-1}, dg = g_k - g_{k-1}  (|dp| not tiny):
        B += outer(dg - B dp, dp) / (dp . dp)        # Broyden update
        B += leak*(I - B)*dt                          # forget stale curvature
    s_target = -tanh(B_yy / eps_c)                     # B_yy estimates Omega_yy
    s_dot    = alpha*(s_target - s)
    y-force  = s * g_y

Predictive: during the off-axis approach (x and y both vary) B builds a curvature
estimate, so it is READY when the trajectory nears the saddle -- flipping s before
the runaway develops. Default B=I -> s~-1 keeps triangular points safe.
Tested on the clean off-axis discriminator, raw dynamics, NO Newton polish.
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

def sim(mu, start, mode, gamma=0.3, alpha=15.0, eps_c=0.5, leak=1.0, dpmin=1e-4,
        dt=0.01, max_steps=200000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0
    B = np.eye(2)
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    p_prev = pos.copy(); g_prev = np.array([gx, gy]); s_peak = -1.0; conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        g = np.array([gx, gy])
        if mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        elif mode == "broyden":
            dp = pos - p_prev; dg = g - g_prev
            nd = dp @ dp
            if nd > dpmin*dpmin:
                B = B + np.outer(dg - B @ dp, dp) / nd
                B = 0.5*(B + B.T)                       # Hessian is symmetric
                B = B + leak*(np.eye(2) - B)*dt          # forget stale curvature
            byy = B[1, 1]
            s_target = -np.tanh(byy / eps_c)
            s += alpha*(s_target - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        s_peak = max(s_peak, s)
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: conv = step; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos, conv, s_peak

def test_point(mu, target, offsets, **kw):
    L = lpoints(mu); tgt = L[target]
    seeds = [tgt + np.array([dx, dy]) for dx in (-0.04, 0.0, 0.04) for dy in offsets]
    out = {}
    for mode in ("discrete", "broyden"):
        hits, peaks = 0, []
        for sd in seeds:
            pt, conv, sp = sim(mu, sd, mode, **kw)
            if np.isfinite(pt).all() and np.linalg.norm(pt - tgt) < 0.02 and conv is not None:
                hits += 1; peaks.append(sp)
        out[mode] = (hits, len(seeds), round(float(np.mean(peaks)), 2) if peaks else None)
    return out

if __name__ == "__main__":
    mu = 7.342e22/(5.972e24+7.342e22)
    print(f"Earth-Moon  mu={mu:.3e}  (off-axis, raw dynamics; 'broyden' uses NO Hessian)")
    targets = (("L1", (0.02, 0.05, 0.10, -0.05)),
               ("L2", (0.02, 0.05, 0.10, -0.05)),
               ("L4", (0.02, 0.05, -0.05, -0.10)))
    for eps_c in (0.3, 0.8):
        for leak in (0.5, 2.0):
            print(f"\n-- broyden: eps_c={eps_c} leak={leak} --")
            for target, offs in targets:
                r = test_point(mu, target, offs, eps_c=eps_c, leak=leak)
                line = "  ".join(f"{m}:{r[m][0]}/{r[m][1]}(s={r[m][2]})"
                                  for m in ("discrete", "broyden"))
                print(f"   {target}:  {line}")
