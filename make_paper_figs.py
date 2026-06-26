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
a1.set_title("(a)", loc='left', fontsize=11, fontweight='bold', pad=4)
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
a2.set_title("(b)", loc='left', fontsize=11, fontweight='bold', pad=4)
a2.set_xlabel("x"); a2.set_ylabel("y"); a2.set_aspect("equal")
figA.colorbar(pm, ax=a2, fraction=0.046, pad=0.02)
figA.savefig("fig_landscape.pdf", dpi=200); figA.savefig("fig_landscape.png", dpi=140)
print("wrote fig_landscape")

# ====================== Fig B: discovery trajectories (Streamlit style) ======================
# Dense grid → long spiralling paths coloured by destination, white background,
# Earth and Moon shown. Matches the interactive Streamlit visualisation.
LC2 = {"L1":"#e74c3c", "L2":"#e67e22", "L3":"#9b59b6", "L4":"#27ae60", "L5":"#2980b9"}

def sim_traj(start, thr=-0.6, gamma=0.3, alpha=15.0, eps_c=0.3, leak=2.0, dpmin=1e-4,
             dt=0.01, max_steps=150000, conv_thr=1e-4, rec=12):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, _ = grad_curv(pos[0], pos[1], mu)
    p_prev = pos.copy(); g_prev = np.array([gx, gy])
    xs_, ys_ = [pos[0]], [pos[1]]
    for step in range(max_steps):
        x, y = pos
        gx, gy, _ = grad_curv(x, y, mu); gn = np.hypot(gx, gy); g = np.array([gx, gy])
        dp = pos - p_prev; dg = g - g_prev; nd = dp @ dp
        if nd > dpmin*dpmin:
            B = B + np.outer(dg - B @ dp, dp) / nd
            B = 0.5*(B + B.T); B = B + leak*(np.eye(2) - B)*dt
        s += alpha*(-np.tanh((B[1,1]-thr)/eps_c) - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        ax_ = 2*vel[1] - gx - gamma*vel[0]; ay_ = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax_, ay_])*dt; pos = pos + vel*dt
        if step % rec == 0: xs_.append(pos[0]); ys_.append(pos[1])
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return np.array(xs_), np.array(ys_), pos

# Dense 20x15 regular grid + structural seeds (ensures L3 on-axis coverage)
gxv2 = np.linspace(-1.5, 1.5, 20)
gyv2 = np.linspace(-1.1, 1.1, 15)
all_seeds = (
    [(x, y) for x in gxv2 for y in gyv2
     if np.hypot(x + mu, y) > 0.12 and np.hypot(x - 1 + mu, y) > 0.07]
    + list(build_grid(mu, fill=6))
)

byL2 = {k: [] for k in L}
for sd in all_seeds:
    tx, ty, endp = sim_traj(sd)
    pol, ok = newton_polish(endp, mu); pt = pol if ok else endp
    if not np.isfinite(pt).all(): continue
    dist, lab = 1e9, None
    for k, v in L.items():
        dd = np.linalg.norm(pt - v)
        if dd < dist: dist, lab = dd, k
    if dist < 0.05:
        plen = float(np.sum(np.hypot(np.diff(tx), np.diff(ty))))
        byL2[lab].append((plen, tx, ty, sd))

shown2 = {}
for k, items in byL2.items():
    items.sort(key=lambda t: -t[0])
    cap = 3 if k == "L3" else 12
    byL2[k] = items[:cap]
    shown2[k] = len(items)

figB = plt.figure(figsize=(6.8, 6.3), facecolor="white")
b = figB.add_subplot(111, facecolor="white")
figB.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.09)

# Faint grey level curves of Omega
b.contour(X, Y, np.clip(Om, vmin, vmax), levels=22, colors="0.78", linewidths=0.35, zorder=1)

# Trajectories coloured by destination L-point
for k, items in byL2.items():
    for plen, tx, ty, sd in items:
        b.plot(tx, ty, color=LC2[k], lw=0.9, alpha=0.75, zorder=2)
        b.plot(sd[0], sd[1], "o", color=LC2[k], ms=4.5, mec="k", mew=0.35, zorder=3)

# Earth and Moon
b.plot(-mu,  0, "o", color="#2e86c1", ms=14, mec="k", mew=0.6, zorder=6, label="Earth")
b.plot(1-mu, 0, "o", color="0.55",   ms=8,  mec="k", mew=0.6, zorder=6, label="Moon")

# L-point stars and labels
for k, p in L.items():
    b.plot(*p, "*", color=LC2[k], mec="k", mew=0.6, ms=16, zorder=7, label=k)
    b.annotate(k, p, textcoords="offset points", xytext=(7, 5), fontsize=10, fontweight="bold")

b.legend(loc="upper right", fontsize=9, framealpha=0.9)
b.set_xlabel("$x$ (co-rotating)"); b.set_ylabel("$y$ (co-rotating)")
b.set_aspect("equal"); b.set_xlim(-1.6, 1.6); b.set_ylim(-1.2, 1.2)

figB.savefig("fig_discovery_qn.pdf", facecolor="white")
figB.savefig("fig_discovery_qn.png", dpi=140, facecolor="white")
print("wrote fig_discovery_qn; shown per L-point:", shown2)
