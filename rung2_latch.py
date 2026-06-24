"""
Rung 2, attempt 2 (Hessian-free, latching memory).  No analytic curvature anywhere.

Physical detector: the conservative force does positive work on the transverse
motion ONLY where that direction is unstable. A leaky memory accumulates the SIGN
of the natural y-force's power, p = (-g_y) * v_y :

    W_dot = -lambda*W + sign(p)             (leaky ratchet; scale-free)
    s_target = -1 + 2*sigmoid(kappa*(W - theta))   (default -1, flips to +1 on evidence)
    s_dot    = alpha*(s_target - s)
    y-force  = s * g_y

Default s=-1 keeps triangular points (no runaway -> W stays low). At a collinear
saddle the natural y-motion runs away -> p>0 persistently -> W crosses theta -> s
latches to +1 -> saddle becomes an attractor. Tested on the clean off-axis
discriminator (saddles L1,L2 vs triangular L4), raw dynamics, no Newton polish.
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
sigmoid = lambda z: 1.0/(1.0 + np.exp(-z))

def sim(mu, start, mode, gamma=0.3, lam=1.0, theta=0.3, kappa=20.0, alpha=20.0,
        dt=0.01, max_steps=200000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; W = 0.0
    s_peak = -1.0; conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        if mode == "const-1":
            s = -1.0
        elif mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        elif mode == "latch":
            p = (-gy) * vel[1]                       # natural y-force power
            W = (1 - lam*dt)*W + (np.sign(p) if abs(vel[1]) > 1e-12 else 0.0)*dt
            s_target = -1.0 + 2.0*sigmoid(kappa*(W - theta))
            s += alpha*(s_target - s)*dt; s = max(-1, min(1, s))
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
    for mode in ("const-1", "discrete", "latch"):
        hits, peaks = 0, []
        for sd in seeds:
            pt, conv, sp = sim(mu, sd, mode, **kw)
            if np.isfinite(pt).all() and np.linalg.norm(pt - tgt) < 0.02 and conv is not None:
                hits += 1; peaks.append(sp)
        out[mode] = (hits, len(seeds), round(float(np.mean(peaks)), 2) if peaks else None)
    return out

if __name__ == "__main__":
    mu = 7.342e22/(5.972e24+7.342e22)
    print(f"Earth-Moon  mu={mu:.3e}   (off-axis seeds, raw dynamics, NO Hessian in 'latch')")
    for theta in (0.15, 0.3, 0.5):
        for lam in (1.0, 3.0):
            print(f"\n-- latch params: theta={theta} lambda={lam} --")
            for target, offs in (("L1", (0.02, 0.05, 0.10, -0.05)),
                                  ("L2", (0.02, 0.05, 0.10, -0.05)),
                                  ("L4", (0.02, 0.05, -0.05, -0.10))):
                r = test_point(mu, target, offs, theta=theta, lam=lam)
                line = "  ".join(f"{m}:{r[m][0]}/{r[m][1]}(s={r[m][2]})"
                                  for m in ("const-1", "discrete", "latch"))
                print(f"   {target}:  {line}")
