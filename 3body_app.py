import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="DMM · All Lagrange Points", layout="wide")

st.title("Digital MemComputing — All 5 Lagrange Points")
st.markdown(
    "A DMM emulator finds any of the 5 Lagrange points in the Earth-Moon rotating frame "
    "by treating **∇Ω = 0** as the clause to satisfy."
)

# ── Analytic L-point positions ─────────────────────────────────────────────────
def lagrange_points(mu):
    x1, x2 = -mu, 1 - mu
    def gx(x):
        r1 = abs(x - x1) + 1e-12
        r2 = abs(x - x2) + 1e-12
        return x - (1-mu)*(x-x1)/r1**3 - mu*(x-x2)/r2**3
    L1 = np.array([brentq(gx, x1+0.01, x2-0.01), 0.0])
    L2 = np.array([brentq(gx, x2+1e-4,  x2+1.0),  0.0])
    L3 = np.array([brentq(gx, x1-2.0,   x1-1e-4), 0.0])
    L4 = np.array([0.5-mu,  np.sqrt(3)/2])
    L5 = np.array([0.5-mu, -np.sqrt(3)/2])
    return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}

DEFAULT_STARTS = {
    "L1": ( 0.05,  0.02),
    "L2": ( 0.08,  0.02),
    "L3": (-0.08,  0.02),
    "L4": ( 0.20, -0.30),
    "L5": ( 0.20,  0.30),
}
DEFAULT_GAMMA = {"L1": 0.8, "L2": 0.8, "L3": 0.8, "L4": 0.1, "L5": 0.1}
LP_STYLE = {
    "L1": ("cyan",    "cross",   "L1"),
    "L2": ("lime",    "cross",   "L2"),
    "L3": ("magenta", "cross",   "L3"),
    "L4": ("gold",    "diamond", "L4"),
    "L5": ("orange",  "diamond", "L5"),
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Target")
    target = st.radio("Lagrange point", ["L1","L2","L3","L4","L5"],
                      index=3, horizontal=True)

    st.header("System")
    mu = st.slider("μ  (Moon/total mass)", 0.001, 0.038, 0.0121, 0.0001,
                   help="Routh: L4/L5 stable only for μ < 0.0385")

    Lpts = lagrange_points(mu)
    L_target = Lpts[target]
    st.info(f"**{target} analytic:** ({L_target[0]:.5f}, {L_target[1]:.5f})")

    st.header("Starting position")
    st.caption(f"Offset from {target}")
    def_dx, def_dy = DEFAULT_STARTS[target]
    dx = st.slider("Δx offset", -0.5, 0.5, def_dx, 0.01)
    dy = st.slider("Δy offset", -0.5, 0.5, def_dy, 0.01)

    st.header("DMM memory")
    alpha   = st.slider("α  (short-term decay)", 0.01, 0.30, 0.05, 0.01)
    beta    = st.slider("β  (long-term growth)", 0.0001, 0.005, 0.001, 0.0001)
    mem_cap = st.slider("long_mem cap",           1.0, 20.0, 5.0, 0.5)

    st.header("Integration")
    gamma = st.slider("γ  (damping)", 0.01, 2.0, DEFAULT_GAMMA[target], 0.01,
                      help="L4/L5: ~0.1 (underdamped, Coriolis stabilises). "
                           "L1/L2/L3: ~0.8 (overdamped, suppresses Coriolis drift).")
    dt        = st.slider("Δt", 0.001, 0.02, 0.01, 0.001)
    max_steps = st.select_slider("Max steps",
                                 [50_000,100_000,200_000,500_000], 200_000)

    st.header("Display")
    show_all  = st.checkbox("Show all 5 L-points", value=True)
    show_mem  = st.checkbox("Memory dynamics panel", value=True)
    om_ceil   = st.slider("Ω ceiling (clip wells)", 1.5, 5.0, 3.2, 0.1)

    run = st.button(f"▶  Find {target}", type="primary", use_container_width=True)

# ── Simulation ─────────────────────────────────────────────────────────────────
def simulate(mu, start, alpha, beta, mem_cap, gamma, dt, max_steps):
    x1, x2 = -mu, 1-mu
    eps = 1e-8
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    sm, lm = 0.0, 1.0
    traj, sl, ll, gl = [], [], [], []
    conv = None
    for step in range(max_steps):
        r1 = np.sqrt((pos[0]-x1)**2 + pos[1]**2) + eps
        r2 = np.sqrt((pos[0]-x2)**2 + pos[1]**2) + eps
        gx = pos[0] - (1-mu)*(pos[0]-x1)/r1**3 - mu*(pos[0]-x2)/r2**3
        gy = pos[1] - (1-mu)* pos[1]      /r1**3 - mu* pos[1]      /r2**3
        gn = np.sqrt(gx*gx + gy*gy)
        sm = (1-alpha)*sm + alpha*gn
        lm = min(lm + beta*sm*dt, mem_cap)
        accel = (np.array([2*vel[1], -2*vel[0]])
                 - np.array([gx, gy]) * lm
                 - gamma * vel)
        vel += accel * dt
        pos  = pos + vel * dt
        if step % 10 == 0:
            traj.append(pos.copy()); sl.append(sm); ll.append(lm); gl.append(gn)
        if gn < 1e-6:
            conv = step
            traj.append(pos.copy()); sl.append(sm); ll.append(lm); gl.append(gn)
            break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 15:
            break
    return np.array(traj), np.array(sl), np.array(ll), np.array(gl), conv, pos, gn

# ── Potential grid ─────────────────────────────────────────────────────────────
@st.cache_data
def potential_grid(mu, nx=130, ny=110):
    x1, x2 = -mu, 1-mu
    xs = np.linspace(-1.6, 1.5, nx)
    ys = np.linspace(-1.1, 1.1, ny)
    X, Y = np.meshgrid(xs, ys)
    R1 = np.sqrt((X-x1)**2 + Y**2) + 1e-6
    R2 = np.sqrt((X-x2)**2 + Y**2) + 1e-6
    return xs, ys, X, Y, (X**2+Y**2)/2 + (1-mu)/R1 + mu/R2

def oz(xy, xs, ys, Om):
    xi = int(np.clip((xy[0]-xs[0])/(xs[-1]-xs[0])*len(xs), 0, len(xs)-1))
    yi = int(np.clip((xy[1]-ys[0])/(ys[-1]-ys[0])*len(ys), 0, len(ys)-1))
    return float(Om[yi, xi])

# ── Static welcome ─────────────────────────────────────────────────────────────
if not run:
    st.info(f"Choose a target in the sidebar and click **▶ Find {target}**. "
            "The 3D surface is fully interactive — drag to rotate, scroll to zoom.")
    xs, ys, X, Y, Om = potential_grid(0.0121)
    Om_c = np.clip(Om, 1.4, 3.2)
    Lpts0 = lagrange_points(0.0121)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Om_c, colorscale="Blues", reversescale=True,
        opacity=0.7, showscale=True, colorbar=dict(title="Ω", thickness=12),
        hoverinfo="skip"))
    for name, pt in Lpts0.items():
        col, sym, lbl = LP_STYLE[name]
        fig.add_trace(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[oz(pt,xs,ys,Om_c)],
            mode="markers+text",
            marker=dict(size=10, color=col, symbol=sym, line=dict(color="black",width=1)),
            text=[f"<b>{lbl}</b>"], textposition="top center", name=lbl))
    fig.add_trace(go.Scatter3d(x=[-0.0121], y=[0], z=[oz([-0.0121,0],xs,ys,Om_c)],
        mode="markers", marker=dict(size=12,color="red",symbol="circle"), name="Earth"))
    fig.add_trace(go.Scatter3d(x=[0.9879], y=[0], z=[oz([0.9879,0],xs,ys,Om_c)],
        mode="markers", marker=dict(size=9,color="blue",symbol="circle"), name="Moon"))
    fig.update_layout(height=640, margin=dict(l=0,r=0,t=45,b=0),
        title="All 5 Lagrange points on Ω surface — drag to rotate",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="Ω",
            camera=dict(eye=dict(x=0.8,y=-2.0,z=0.8)),
            aspectmode="manual", aspectratio=dict(x=1.5,y=1.0,z=0.5)),
        legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.8)"),
        paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────────
start_pos = L_target + np.array([dx, dy])
with st.spinner(f"DMM finding {target}..."):
    traj, sl, ll, gl, conv, final_pos, final_gn = simulate(
        mu, start_pos, alpha, beta, mem_cap, gamma, dt, max_steps)
    xs, ys, X, Y, Om = potential_grid(mu)
    Om_clip = np.clip(Om, 1.4, om_ceil)

x1, x2 = -mu, 1-mu
error = np.linalg.norm(final_pos - L_target) if np.isfinite(final_pos).all() else float('nan')

# ── Metrics ────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Target", target)
if conv is not None:
    c2.metric("Status","✅ Converged"); c3.metric("Steps",f"{conv:,}")
elif np.isfinite(final_pos).all():
    c2.metric("Status","⏳ Max steps"); c3.metric("Steps",f"{max_steps:,}")
else:
    c2.metric("Status","💥 Diverged");  c3.metric("Steps","—")
c4.metric("Position error", f"{error:.2e}" if np.isfinite(error) else "—")
c5.metric("|∇Ω| final",    f"{final_gn:.2e}" if np.isfinite(final_gn) else "—")

st.markdown("---")

# ── 3D Plotly chart ────────────────────────────────────────────────────────────
n = len(traj)
fig3d = go.Figure()
fig3d.add_trace(go.Surface(x=X, y=Y, z=Om_clip, colorscale="Blues", reversescale=True,
    opacity=0.65, showscale=True, colorbar=dict(title="Ω", thickness=12),
    hoverinfo="skip", name="Ω surface"))

if n > 1:
    tz = np.array([oz(p, xs, ys, Om_clip) for p in traj])
    t_norm = np.linspace(0, 1, n)
    fig3d.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=tz, mode="lines",
        line=dict(color=t_norm, colorscale="Plasma", width=5), name="Instanton path"))
    fig3d.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[tz[0]],
        mode="markers", marker=dict(size=8,color="cyan",symbol="square"), name="Start"))
    fig3d.add_trace(go.Scatter3d(x=[traj[-1,0]], y=[traj[-1,1]], z=[tz[-1]],
        mode="markers", marker=dict(size=11,color="black",symbol="diamond"),
        name=f"End ({traj[-1,0]:.4f}, {traj[-1,1]:.4f})"))

fig3d.add_trace(go.Scatter3d(x=[x1],y=[0],z=[oz([x1,0],xs,ys,Om_clip)],
    mode="markers", marker=dict(size=12,color="red",symbol="circle",
    line=dict(color="black",width=1)), name="Earth"))
fig3d.add_trace(go.Scatter3d(x=[x2],y=[0],z=[oz([x2,0],xs,ys,Om_clip)],
    mode="markers", marker=dict(size=9,color="blue",symbol="circle",
    line=dict(color="black",width=1)), name="Moon"))

for name, pt in Lpts.items():
    if not show_all and name != target:
        continue
    col, sym, lbl = LP_STYLE[name]
    is_t = (name == target)
    fig3d.add_trace(go.Scatter3d(x=[pt[0]],y=[pt[1]],z=[oz(pt,xs,ys,Om_clip)],
        mode="markers+text",
        marker=dict(size=13 if is_t else 8, color=col, symbol=sym,
                    line=dict(color="black",width=2 if is_t else 1)),
        text=[f"<b>{lbl}</b>" if is_t else lbl], textposition="top center", name=lbl))

fig3d.update_layout(height=660, margin=dict(l=0,r=0,t=45,b=0),
    title=f"DMM Instanton Path → {target}  |  drag to rotate · scroll to zoom · double-click to reset",
    scene=dict(xaxis_title="x (rotating frame)", yaxis_title="y (rotating frame)",
        zaxis_title="Ω", camera=dict(eye=dict(x=0.9,y=-2.0,z=0.8)),
        aspectmode="manual", aspectratio=dict(x=1.5,y=1.0,z=0.5)),
    legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.85)"),
    paper_bgcolor="white")
st.plotly_chart(fig3d, use_container_width=True)

# ── 2D projection ──────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors
st.markdown("### 2D Projection — co-rotating frame")
fig2d, ax2d = plt.subplots(figsize=(8, 6))
cf = ax2d.contourf(X, Y, Om_clip, levels=40, cmap="Blues_r", alpha=0.88)
plt.colorbar(cf, ax=ax2d, label="Ω (clipped)", shrink=0.8)
ax2d.contour(X, Y, Om_clip, levels=15, colors="white", linewidths=0.35, alpha=0.35)
if n > 1:
    t_norm2d = np.linspace(0, 1, n)
    for i in range(n - 1):
        c2d = plt.cm.plasma(t_norm2d[i])
        ax2d.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=c2d, lw=1.0,
                  solid_capstyle="round")
    ax2d.scatter(traj[0, 0], traj[0, 1], color="cyan", s=80, zorder=6,
                 marker="s", edgecolors="black", lw=0.8, label="Start")
    ax2d.scatter(traj[-1, 0], traj[-1, 1], color="black", s=100, zorder=6,
                 marker="D", edgecolors="white", lw=0.8,
                 label=f"End ({traj[-1,0]:.4f}, {traj[-1,1]:.4f})")
ax2d.scatter(x1, 0, color="red",  s=180, zorder=7, edgecolors="black", lw=0.8, label="Earth")
ax2d.scatter(x2, 0, color="blue", s=100, zorder=7, edgecolors="black", lw=0.8, label="Moon")
LP2D_STYLE = {"L1":("cyan","P",120),"L2":("lime","P",120),"L3":("magenta","P",120),
              "L4":("gold","D",180),"L5":("orange","D",180)}
LP2D_OFF   = {"L1":(0.03,0.05),"L2":(0.03,0.05),"L3":(-0.20,0.05),
              "L4":(0.05,0.05),"L5":(0.05,-0.08)}
for lname, lpt in Lpts.items():
    col2, mk2, sz2 = LP2D_STYLE[lname]
    ax2d.scatter(lpt[0], lpt[1], color=col2, s=sz2, marker=mk2,
                 zorder=8, edgecolors="black", lw=0.8)
    dx2, dy2 = LP2D_OFF[lname]
    ax2d.text(lpt[0]+dx2, lpt[1]+dy2, lname, fontsize=9, fontweight="bold",
              color=col2, bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))
sm_cb2 = plt.cm.ScalarMappable(cmap="plasma",
                                norm=mcolors.Normalize(0, 1))
sm_cb2.set_array([])
cb3 = plt.colorbar(sm_cb2, ax=ax2d, shrink=0.45, pad=0.02)
cb3.set_label("Trajectory time (norm.)", fontsize=8)
cb3.set_ticks([0, 0.5, 1]); cb3.set_ticklabels(["Start", "Mid", "End"])
ax2d.set_xlabel("x (rotating frame)"); ax2d.set_ylabel("y (rotating frame)")
ax2d.set_title(f"DMM Instanton Path → {target}  |  2D top-down view")
ax2d.legend(loc="upper left", fontsize=8, framealpha=0.85)
ax2d.set_xlim(-1.6, 1.5); ax2d.set_ylim(-1.1, 1.1)
ax2d.set_aspect("equal")
plt.tight_layout()
st.pyplot(fig2d, use_container_width=True)

# ── Memory dynamics ────────────────────────────────────────────────────────────
if show_mem and n > 0:
    sx = np.arange(len(sl)) * 10
    fig_m, ax = plt.subplots(figsize=(11, 3.4))
    ax2 = ax.twinx()
    l1, = ax.plot(sx, sl, color="royalblue",  lw=1.3, label="short_mem  xₛ")
    l2, = ax.plot(sx, ll, color="darkorange", lw=1.8, label="long_mem  w_L")
    l3, = ax2.semilogy(sx, np.clip(gl,1e-10,None),
                        color="seagreen", lw=1.1, alpha=0.85, label="|∇Ω|")
    ax2.axhline(1e-6, ls="--", color="seagreen", lw=0.8, alpha=0.5)
    if conv:
        ax.axvline(conv, ls=":", color="grey", lw=1.2, label=f"Converged @ {conv:,}")
    ax.set_xlabel("Simulation step"); ax.set_ylabel("Memory value")
    ax2.set_ylabel("|∇Ω|  (log)", color="seagreen")
    ax2.tick_params(axis="y", colors="seagreen")
    ax.set_title(f"DMM Memory Dynamics — finding {target}")
    ax.grid(alpha=0.3)
    ax.legend([l1,l2,l3], ["short_mem xₛ","long_mem w_L","|∇Ω|"], fontsize=9, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig_m, use_container_width=True)

# ── L-point table ──────────────────────────────────────────────────────────────
with st.expander("All 5 Lagrange point positions (analytic)", expanded=True):
    rows = []
    for name, pt in Lpts.items():
        stab = "✅ Stable (Routh's criterion)" if name in ("L4","L5") else "⚠️ Unstable saddle"
        marker = " ◀ target" if name == target else ""
        rows.append(f"| **{name}**{marker} | {pt[0]:.6f} | {pt[1]:.6f} | {stab} |")
    st.markdown(
        "| Point | x | y | Stability |\n"
        "|-------|---|---|----------|\n" + "\n".join(rows))

with st.expander("Physics & DMM equations", expanded=False):
    st.markdown(r"""
### Effective potential
$$\Omega = \frac{x^2+y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

### Rotating-frame EOM with DMM memory
$$\ddot{x} = \underbrace{2\dot{y}}_{\text{Coriolis}} - w_L\,\partial_x\Omega - \gamma\dot{x}, \qquad
\ddot{y} = \underbrace{-2\dot{x}}_{\text{Coriolis}} - w_L\,\partial_y\Omega - \gamma\dot{y}$$

### DMM memory (dynamic Lagrange multipliers)
$$x_s \leftarrow (1-\alpha)\,x_s + \alpha\,|\nabla\Omega|, \qquad
w_L \leftarrow \min(w_L + \beta\,x_s\,\Delta t,\ w_{cap})$$

| Points | Stability | Damping strategy |
|--------|-----------|-----------------|
| **L4, L5** | ✅ Stable (Routh: μ < 0.0385) | Light γ ~0.1 — Coriolis is the stabiliser |
| **L1, L2, L3** | ⚠️ Unstable saddle | Heavy γ ~0.8 — suppresses Coriolis y-drift |
    """)
