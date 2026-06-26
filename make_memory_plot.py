"""
Memory figure: the quasi-Newton memory LEARNS the local curvature from gradient
history (no analytic Hessian) and adaptively sets the force-inversion gain s.

Two Earth-Moon trajectories on the SAME machine:
  - off-axis approach to the collinear saddle L1 (Omega_yy<0): memory discovers
    B_yy<0 and ramps s: -1 -> +1, converting the saddle into an attractor;
  - off-axis approach to the triangular point L4 (Omega_yy=+2.25>0): memory finds
    B_yy>0 and keeps s ~ -1 (no inversion; Coriolis already stabilizes it).
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

def run(mu, start, thr=-0.6, gamma=0.3, alpha=15.0, eps_c=0.3, leak=2.0, dpmin=1e-4,
        dt=0.01, max_steps=60000, conv_thr=1e-4, rec=5):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    p_prev = pos.copy(); g_prev = np.array([gx, gy])
    H = dict(t=[], x=[], y=[], byy=[], oyy=[], s=[], gn=[])
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy); g = np.array([gx, gy])
        dp = pos - p_prev; dg = g - g_prev; nd = dp @ dp
        if nd > dpmin*dpmin:
            B = B + np.outer(dg - B @ dp, dp) / nd
            B = 0.5*(B + B.T); B = B + leak*(np.eye(2) - B)*dt
        s += alpha*(-np.tanh((B[1,1] - thr)/eps_c) - s)*dt; s = max(-1, min(1, s))
        p_prev = pos.copy(); g_prev = g
        if step % rec == 0:
            H["t"].append(step); H["x"].append(x); H["y"].append(y)
            H["byy"].append(B[1,1]); H["oyy"].append(oyy); H["s"].append(s); H["gn"].append(gn)
        ax = 2*vel[1] - gx - gamma*vel[0]; ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return {k: np.array(v) for k, v in H.items()}

mu = 7.342e22/(5.972e24+7.342e22)
L = lpoints(mu)
A  = run(mu, L["L1"] + np.array([0.0, 0.05]))   # saddle
Bt = run(mu, L["L4"] + np.array([0.0, 0.05]))   # triangular minimum
C_SAD, C_TRI = "#c0392b", "#2471a3"

fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)

# panels (a) and (b): phase-plane trajectories coloured by s
def traj_panel(a, H, Lpt, label, seed):
    pts  = np.column_stack([H["x"], H["y"]]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="coolwarm", norm=plt.Normalize(-1, 1), lw=1.3, alpha=0.9)
    lc.set_array(H["s"][:-1]); a.add_collection(lc)
    a.plot(*Lpt, "*", color="k", ms=14, zorder=5)
    a.plot(*seed, "s", color="0.3", ms=6, zorder=5)
    a.annotate("seed", seed, textcoords="offset points", xytext=(6, 4), fontsize=8)
    # label outside, above panel (left-aligned)
    a.set_title(label, loc='left', fontsize=11, fontweight='bold', pad=4)
    a.set_xlabel("x"); a.set_ylabel("y")
    a.autoscale(); a.margins(0.15); a.set_aspect("equal", "datalim")
    return lc

lc = traj_panel(ax[0,0], A,  L["L1"], "(a)", L["L1"] + np.array([0, 0.05]))
traj_panel(     ax[0,1], Bt, L["L4"], "(b)", L["L4"] + np.array([0, 0.05]))
cb = fig.colorbar(lc, ax=ax[0,:].tolist(), fraction=0.046, pad=0.02)
cb.set_label("inversion gain $s$", fontsize=9)

# panel (c): Broyden estimate B_yy tracks true Omega_yy
c = ax[1,0]
c.plot(A["t"],  A["byy"],  color=C_SAD, lw=2,      label=r"$B_{yy}$ ($L_1$)")
c.plot(A["t"],  A["oyy"],  color=C_SAD, lw=1, ls="--", label=r"true $\Omega_{yy}$ ($L_1$)")
c.plot(Bt["t"], Bt["byy"], color=C_TRI, lw=2,      label=r"$B_{yy}$ ($L_4$)")
c.plot(Bt["t"], Bt["oyy"], color=C_TRI, lw=1, ls="--", label=r"true $\Omega_{yy}$ ($L_4$)")
c.axhline(0, color="0.6", lw=0.8)
c.set_ylim(-7.5, 3.5)
# label outside, above panel
c.set_title("(c)", loc='left', fontsize=11, fontweight='bold', pad=4)
c.set_xlabel("step"); c.set_ylabel(r"$y$-curvature", fontsize=10)
c.legend(fontsize=7, ncol=2, loc="lower right")
# annotations placed in empty regions away from the main curves
c.annotate(r"$B_{yy}\to\Omega_{yy}<0$  (saddle)",
           (0.50, 0.08), xycoords="axes fraction", color=C_SAD, fontsize=8)
c.annotate(r"$B_{yy}\to\Omega_{yy}>0$  (triangular)",
           (0.38, 0.92), xycoords="axes fraction", color=C_TRI, fontsize=8)

# panel (d): s ramps adaptively; gradient decays shown as dotted lines
d  = ax[1,1]
dd = d.twinx()
# gradient: dotted, same colours, fully opaque (no alpha fading)
dd.semilogy(A["t"],  A["gn"],  color=C_SAD, lw=1.1, ls=":", zorder=1)
dd.semilogy(Bt["t"], Bt["gn"], color=C_TRI, lw=1.1, ls=":", zorder=1)
dd.set_ylabel(r"$\|\nabla\Omega\|$", fontsize=10)
# s curves on top (higher zorder, thicker)
d.plot(A["t"],  A["s"],  color=C_SAD, lw=2.2, label=r"$s$ ($L_1$ saddle): $-1\to+1$",    zorder=3)
d.plot(Bt["t"], Bt["s"], color=C_TRI, lw=2.2, label=r"$s$ ($L_4$ triangular): stays $-1$", zorder=3)
d.set_ylim(-1.15, 1.15)
# label outside, above panel
d.set_title("(d)", loc='left', fontsize=11, fontweight='bold', pad=4)
d.set_xlabel("step"); d.set_ylabel("$s$", fontsize=11)
d.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.94))

fig.savefig("fig_memory_curvature.pdf")
fig.savefig("fig_memory_curvature.png", dpi=140)
print("wrote fig_memory_curvature.pdf / .png")
print(f"L1 run: {len(A['t'])} samples, final s={A['s'][-1]:.2f}, "
      f"B_yy={A['byy'][-1]:.2f}, true oyy={A['oyy'][-1]:.2f}")
print(f"L4 run: {len(Bt['t'])} samples, final s={Bt['s'][-1]:.2f}, "
      f"B_yy={Bt['byy'][-1]:.2f}, true oyy={Bt['oyy'][-1]:.2f}")
