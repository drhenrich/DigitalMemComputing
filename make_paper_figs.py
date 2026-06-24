"""
Two figures for dmm_curvature_ajp.tex:
  fig_landscape.pdf    Omega and its transverse curvature Omega_yy (invert/descend regions)
  fig_discovery_qn.pdf trajectories of the Hessian-free curvature-learning machine to all 5 L-points
"""
import sys, types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
grad_curv = _ns["grad_curv"]; lpoints = _ns["lpoints"]
effective_potential = _ns["effective_potential"]; build_grid = _ns["build_grid"]
newton_polish = _ns["newton_polish"]

mu = 7.342e22/(5.972e24+7.342e22)
L = lpoints(mu)
LC = {"L1":"#e41a1c","L2":"#ff7f00","L3":"#4daf4a","L4":"#377eb8","L5":"#984ea3"}

# ============================ Fig A: landscape ============================
xs = np.linspace(-1.5, 1.5, 500); ys = np.linspace(-1.25, 1.25, 500)
X, Y = np.meshgrid(xs, ys)
Om = effective_potential(X, Y, mu)
_, _, Oyy = grad_curv(X, Y, mu)
Lvals = np.array([effective_potential(*L[k], mu) for k in L])
vmin, vmax = Lvals.min() - 0.05, Lvals.max() + 0.35

figA, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3), constrained_layout=True)

LBL = {"L1": (-22, -13), "L2": (8, -13), "L3": (8, 6), "L4": (7, 6), "L5": (7, -14)}
cf = a1.contourf(X, Y, np.clip(Om, vmin, vmax), levels=24, cmap="viridis")
cf.set_rasterized(True)
a1.contour(X, Y, np.clip(Om, vmin, vmax), levels=12, colors="w", linewidths=0.3, alpha=0.4)
a1.plot(-mu, 0, "o", color="w", mec="k", ms=8); a1.plot(1-mu, 0, "o", color="w", mec="k", ms=5)
for k, p in L.items():
    a1.plot(*p, "*", color=LC[k], mec="k", ms=13, zorder=5)
    a1.annotate(k, p, textcoords="offset points", xytext=LBL[k], fontsize=8.5, color="k",
                bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.8))
a1.set_title(r"(a) effective potential $\Omega$  (saddles $L_{1,2,3}$, minima $L_{4,5}$)", fontsize=9.5)
a1.set_xlabel("x"); a1.set_ylabel("y"); a1.set_aspect("equal")
figA.colorbar(cf, ax=a1, fraction=0.046, pad=0.02)

pm = a2.pcolormesh(X, Y, np.clip(Oyy, -3, 3), cmap="coolwarm", vmin=-3, vmax=3,
                   shading="auto", rasterized=True)
a2.contour(X, Y, Oyy, levels=[0], colors="k", linewidths=1.0, linestyles="--")
for k, p in L.items():
    a2.plot(*p, "*", color="k", mfc=LC[k], ms=12, zorder=5)
a2.text(0.02, 0.96, r"$\Omega_{yy}<0$: invert  ($s=+1$)", transform=a2.transAxes,
        fontsize=8.5, va="top", color="#7a1010")
a2.text(0.02, 0.06, r"$\Omega_{yy}>0$: descend  ($s=-1$)", transform=a2.transAxes,
        fontsize=8.5, va="bottom", color="#10307a")
a2.set_title(r"(b) transverse curvature $\Omega_{yy}$ (dashed: $\Omega_{yy}=0$)", fontsize=9.5)
a2.set_xlabel("x"); a2.set_ylabel("y"); a2.set_aspect("equal")
figA.colorbar(pm, ax=a2, fraction=0.046, pad=0.02)
figA.savefig("fig_landscape.pdf", dpi=200); figA.savefig("fig_landscape.png", dpi=140)
print("wrote fig_landscape")

# ====================== Fig B: discovery trajectories ======================
def sim_traj(start, thr=-0.6, gamma=0.3, alpha=15.0, eps_c=0.3, leak=2.0, dpmin=1e-4,
             dt=0.01, max_steps=120000, conv_thr=1e-4, rec=8):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, _ = grad_curv(pos[0], pos[1], mu); p_prev = pos.copy(); g_prev = np.array([gx, gy])
    xs_, ys_ = [pos[0]], [pos[1]]; conv = False
    for _ in range(max_steps):
        x, y = pos
        gx, gy, _ = grad_curv(x, y, mu); gn = np.hypot(gx, gy); g = np.array([gx, gy])
        dp = pos - p_prev; dg = g - g_prev; nd = dp @ dp
        if nd > dpmin*dpmin:
            B = B + np.outer(dg - B @ dp, dp) / nd; B = 0.5*(B + B.T); B = B + leak*(np.eye(2)-B)*dt
        s += alpha*(-np.tanh((B[1,1]-thr)/eps_c) - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        ax_ = 2*vel[1] - gx - gamma*vel[0]; ay_ = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax_, ay_])*dt; pos = pos + vel*dt
        if len(xs_) == 0 or _ % rec == 0: xs_.append(pos[0]); ys_.append(pos[1])
        if gn < conv_thr: conv = True; break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return np.array(xs_), np.array(ys_), pos, conv

figB, b = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
b.contour(X, Y, np.clip(Om, vmin, vmax), levels=18, colors="0.78", linewidths=0.4)
# representative trajectories: launch from moderate offsets around each L-point and
# colour each path by the L-point it actually converges to.
seeds = list(build_grid(mu, fill=6))               # the actual structural seed grid
byL = {k: [] for k in L}
for sd in seeds:
    tx, ty, endp, conv = sim_traj(sd)
    if not conv: continue
    pol, ok = newton_polish(endp, mu); pt = pol if ok else endp
    if not np.isfinite(pt).all(): continue
    d, lab = 1e9, None
    for k, v in L.items():
        dd = np.linalg.norm(pt - v)
        if dd < d: d, lab = dd, k
    if d < 0.05:
        plen = float(np.sum(np.hypot(np.diff(tx), np.diff(ty))))
        byL[lab].append((plen, tx, ty, sd))
disp = {}                                          # selected trajectories to draw
shown = {}
for k, items in byL.items():
    items.sort(key=lambda t: -t[0])
    disp[k] = items[:3]
    shown[k] = len(items)

def draw_trajs(ax, ms_seed):
    for k, items in disp.items():
        for plen, tx, ty, sd in items:
            ax.plot(tx, ty, color=LC[k], lw=1.3, alpha=0.85)
            ax.plot(sd[0], sd[1], "o", color=LC[k], ms=ms_seed, alpha=0.8)

draw_trajs(b, 4.5)
b.plot(-mu, 0, "o", color="0.2", ms=8); b.plot(1-mu, 0, "o", color="0.4", ms=5)
for k, p in L.items():
    b.plot(*p, "*", color=LC[k], mec="k", ms=15, zorder=6)
    b.annotate(k, p, textcoords="offset points", xytext=(6, 5), fontsize=10)
b.set_title("Trajectories of the curvature-learning machine (Earth–Moon)\n"
            "circles: structural-grid seeds; stars: discovered Lagrange points", fontsize=9.5)
b.set_xlabel("x"); b.set_ylabel("y"); b.set_aspect("equal")
b.set_xlim(-1.5, 1.5); b.set_ylim(-1.1, 1.1)

# zoom inset on the L1/L2 (Moon) region, which is cramped at full scale
axins = b.inset_axes([0.60, 0.62, 0.37, 0.36])
axins.contour(X, Y, np.clip(Om, vmin, vmax), levels=24, colors="0.8", linewidths=0.3)
draw_trajs(axins, 5.0)
axins.plot(1-mu, 0, "o", color="0.4", ms=6)
for k in ("L1", "L2"):
    axins.plot(*L[k], "*", color=LC[k], mec="k", ms=14, zorder=6)
    axins.annotate(k, L[k], textcoords="offset points", xytext=(5, 4), fontsize=8)
axins.set_xlim(0.78, 1.22); axins.set_ylim(-0.13, 0.13)
axins.set_xticks([]); axins.set_yticks([]); axins.set_aspect("equal")
b.indicate_inset_zoom(axins, edgecolor="0.4")
figB.savefig("fig_discovery_qn.pdf"); figB.savefig("fig_discovery_qn.png", dpi=140)
print("wrote fig_discovery_qn; shown per L-point:", shown)
