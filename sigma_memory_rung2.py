"""
Rung 2: the memory INFERS the saddle-inversion from the dynamics -- never reading
the analytic Hessian Omega_yy.

Idea (quasi-Newton-like): the machine already evaluates the gradient g=grad Omega
every step. The y-curvature is the secant slope of g_y along y,
    Omega_yy  ~  d(g_y)/dy  ~  (g_y - g_y_prev)/(y - y_prev),
i.e. it can be estimated online from the gradient HISTORY. A memory variable
accumulates this estimate and sets a continuous inversion gain:

    c_est = dgy*dy / (dy^2 + delta)          # secant estimate of Omega_yy
    s_target = -tanh(c_est / eps_c)           # ->+1 where curvature<0 (saddle)
    s_dot   = alpha*(s_target - s),  s in [-1,1]
    y-force = s * g_y                          # was sign(-Omega_yy)*g_y

The memory's job a constant cannot do: hold the established inversion sign even as
the instantaneous signal weakens near the fixed point (dy->0). No analytic second
derivative is used anywhere.
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
build_grid    = _ns["build_grid"]
newton_polish = _ns["newton_polish"]
grad_curv     = _ns["grad_curv"]
lpoints       = _ns["lpoints"]

def simulate(mu, start, mode, alpha, eps_c, gamma, delta=1e-10,
             dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2)
    s = -1.0
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    y_prev, gy_prev = pos[1], gy
    conv = None; s_end = s
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu)
        gn = np.hypot(gx, gy)
        if mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        else:  # memdyn: Hessian-free, curvature from gradient history
            dy = y - y_prev; dgy = gy - gy_prev
            c_est = dgy*dy / (dy*dy + delta)
            s_target = -np.tanh(c_est / eps_c)
            s = s + alpha*(s_target - s)*dt
            s = max(-1.0, min(1.0, s))
        y_prev, gy_prev = y, gy
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

def discover(mu, mode, alpha=8.0, eps_c=1.0, gamma=0.3):
    L = lpoints(mu)
    found = {k: 0 for k in L}; convs = []
    s_by_label = {k: [] for k in L}
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
                s_by_label[lab].append(s_end)
                if conv is not None: convs.append(conv)
    n = sum(1 for v in found.values() if v > 0)
    med = int(np.median(convs)) if convs else -1
    return n, med, s_by_label

if __name__ == "__main__":
    mu_em = 7.342e22/(5.972e24+7.342e22)
    print(f"Earth-Moon (mu={mu_em:.3e})   y-force gain s in [-1,1]\n")
    nd, md, _ = discover(mu_em, "discrete")
    print(f"discrete sign(-oyy):  {nd}/5  (median conv {md})\n")
    print("Hessian-free memory (curvature from gradient history):")
    for alpha in (30.0, 8.0, 2.0):
        for eps_c in (0.5, 2.0):
            n, m, sbl = discover(mu_em, "memdyn", alpha=alpha, eps_c=eps_c)
            sm = {k: (round(float(np.mean(v)), 2) if v else None) for k, v in sbl.items()}
            print(f"  alpha={alpha:4.1f} eps_c={eps_c:.1f}:  {n}/5  (median conv {m})  "
                  f"mean s@: {sm}")
    print("\n(expect s~+1 at collinear L1/L2/L3 [saddles], s~-1 at L4/L5)")
