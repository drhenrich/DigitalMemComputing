"""
Prototype: turn the discrete curvature switch  sigma = sign(-Omega_yy) in {-1,+1}
into a CONTINUOUS memory variable s(t) evolving by a smooth ODE.

  discrete (current):  sig = +1 if Omega_yy<0 else -1          (non-smooth switch)
  memory  (proposed):  s_dot = alpha*( s_target - s ),  s in [-1,1]
                       s_target = -tanh(Omega_yy / eps_c)       (smooth, ->+-1)
                       y-force uses  s * gy   (continuous gain, was sig*gy)

The whole machine is then a SMOOTH autonomous flow with one memory d.o.f. per
inverted axis. Damping is a constant gamma (we showed a constant suffices for the
*dissipation*; the memory's real job is the position-dependent sign inversion, which
no constant can do). We test whether the continuous memory-sigma still converts the
collinear saddles into attractors and recovers all five Lagrange points.
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
build_grid   = _ns["build_grid"]
newton_polish= _ns["newton_polish"]
grad_curv    = _ns["grad_curv"]
lpoints      = _ns["lpoints"]

def simulate(mu, start, mode, alpha, eps_c, gamma, dt=0.01, max_steps=120000, conv_thr=1e-4):
    """mode='discrete' reproduces sign(-oyy); mode='memory' uses the smooth s(t)."""
    pos = np.array(start, float); vel = np.zeros(2)
    s = -1.0                       # memory starts in the 'natural' state
    conv = None; s_end = s
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu)
        gn = np.hypot(gx, gy)
        if mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        else:                       # smooth memory ODE
            s_target = -np.tanh(oyy / eps_c)
            s = s + alpha*(s_target - s)*dt
            s = max(-1.0, min(1.0, s))
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt
        pos = pos + vel*dt
        s_end = s
        if gn < conv_thr:
            conv = step; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return pos, conv, s_end

def discover(mu, mode, alpha=8.0, eps_c=0.5, gamma=0.3):
    L = lpoints(mu)
    found = {k: 0 for k in L}; convs = []
    for (x, y) in build_grid(mu, fill=6):
        endp, conv, s_end = simulate(mu, [x, y], mode, alpha, eps_c, gamma)
        pt = endp
        if np.isfinite(endp).all():
            pol, ok = newton_polish(endp, mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d, lab = 1e9, None
            for k, v in L.items():
                dd = np.linalg.norm(pt - v)
                if dd < d: d, lab = dd, k
            if d < 0.05:
                found[lab] += 1
                if conv is not None: convs.append(conv)
    n = sum(1 for v in found.values() if v > 0)
    med = int(np.median(convs)) if convs else -1
    return n, med

if __name__ == "__main__":
    # Earth-Moon: has all three collinear saddles where sigma matters most.
    mu_em = 7.342e22/(5.972e24+7.342e22)
    print(f"Earth-Moon (mu={mu_em:.3e})")
    nd, md = discover(mu_em, "discrete")
    print(f"  discrete sign(-oyy):           {nd}/5  (median conv-step {md})")
    print("  continuous memory-sigma  s_dot = alpha*(-tanh(oyy/eps_c) - s):")
    for alpha in (50.0, 8.0, 2.0):
        for eps_c in (0.2, 0.5, 1.0):
            n, m = discover(mu_em, "memory", alpha=alpha, eps_c=eps_c)
            print(f"    alpha={alpha:5.1f} eps_c={eps_c:.1f}:  {n}/5  (median conv-step {m})")
