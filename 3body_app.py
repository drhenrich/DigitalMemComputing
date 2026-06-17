import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DMM · 3-Body L4 Finder", layout="wide")

st.title("Digital MemComputing — Restricted 3-Body L4 Lagrange Point")
st.markdown(
    "A DMM emulator finds L4 by following the rotating-frame equations of motion "
    "with memory-amplified potential force and Coriolis stabilisation."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System parameters")
    mu = st.slider("μ  (Moon/total mass ratio)", 0.001, 0.038, 0.0121, 0.0001,
                   help="Routh's criterion: μ < 0.0385 for L4/L5 stability")

    st.header("Starting position")
    st.caption("Offset from L4 analytic position")
    dx = st.slider("Δx offset", -0.4, 0.4,  0.20, 0.01)
    dy = st.slider("Δy offset", -0.5, 0.1, -0.30, 0.01)

    st.header("DMM memory")
    alpha   = st.slider("α  (short-term decay)",  0.01, 0.30, 0.05, 0.01)
    beta    = st.slider("β  (long-term growth)",  0.0001, 0.005, 0.001, 0.0001)
    mem_cap = st.slider("long_mem cap",            1.0, 20.0, 5.0, 0.5)

    st.header("Integration")
    gamma     = st.slider("γ  (damping)", 0.01, 0.50, 0.10, 0.01,
                          help="Must be ≪ 2·ω_libration ≈ 1.7 — too high → overdamped, misses L4")
    dt        = st.slider("Δt",           0.001, 0.02, 0.01, 0.001)
    max_steps = st.select_slider("Max steps",
                                 [50_000, 100_000, 200_000, 500_000], 200_000)

    st.header("Display")
    show_2d     = st.checkbox("Also show 2D contour view", value=False)
    show_memory = st.checkbox("Memory dynamics panel",      value=True)
    omega_clip  = st.slider("Ω ceiling (clip deep wells)", 1.5, 5.0, 3.2, 0.1,
                             help="Clips the deep gravity wells so the surface shape is visible")

    run = st.button("▶  Run simulation", type="primary", use_container_width=True)

# ── Simulation ─────────────────────────────────────────────────────────────────
def run_simulation(mu, dx, dy, alpha, beta, mem_cap, gamma, dt, max_steps):
    x1, x2 = -mu, 1 - mu
    L4  = np.array([0.5 - mu, np.sqrt(3) / 2])
    eps = 1e-8

    pos       = L4 + np.array([dx, dy])
    vel       = np.array([0.0, 0.0])
    short_mem = 0.0
    long_mem  = 1.0

    traj, short_log, long_log, grad_log = [], [], [], []
    converged_at = None

    for step in range(max_steps):
        r1 = np.sqrt((pos[0] - x1)**2 + pos[1]**2) + eps
        r2 = np.sqrt((pos[0] - x2)**2 + pos[1]**2) + eps

        gx = pos[0] - (1-mu)*(pos[0]-x1)/r1**3 - mu*(pos[0]-x2)/r2**3
        gy = pos[1] - (1-mu)* pos[1]      /r1**3 - mu* pos[1]      /r2**3
        grad_norm = np.sqrt(gx*gx + gy*gy)

        short_mem = (1 - alpha)*short_mem + alpha*grad_norm
        long_mem  = min(long_mem + beta*short_mem*dt, mem_cap)

        accel = (np.array([2.0*vel[1], -2.0*vel[0]])
                 - np.array([gx, gy]) * long_mem
                 - gamma * vel)

        vel += accel * dt
        pos  = pos + vel * dt

        if step % 10 == 0:
            traj.append(pos.copy())
            short_log.append(short_mem)
            long_log.append(long_mem)
            grad_log.append(grad_norm)

        if grad_norm < 1e-6:
            converged_at = step
            traj.append(pos.copy())
            short_log.append(short_mem); long_log.append(long_mem); grad_log.append(grad_norm)
            break

        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 10:
            break

    return (np.array(traj), np.array(short_log), np.array(long_log),
            np.array(grad_log), L4, converged_at, pos, grad_norm,
            np.array([-mu, 1-mu]))

# ── Potential grid (cached) ────────────────────────────────────────────────────
@st.cache_data
def potential_grid(mu, nx=120, ny=100):
    x1, x2 = -mu, 1 - mu
    xs = np.linspace(-0.6, 1.3, nx)
    ys = np.linspace(-0.4, 1.2, ny)
    X, Y = np.meshgrid(xs, ys)
    R1 = np.sqrt((X - x1)**2 + Y**2) + 1e-6
    R2 = np.sqrt((X - x2)**2 + Y**2) + 1e-6
    return xs, ys, X, Y, (X**2 + Y**2)/2 + (1-mu)/R1 + mu/R2

# ── Omega lookup ───────────────────────────────────────────────────────────────
def omega_at(xy, xs, ys, Om_clip):
    xi = int(np.clip((xy[0] + 0.6) / 1.9 * len(xs), 0, len(xs)-1))
    yi = int(np.clip((xy[1] + 0.4) / 1.6 * len(ys), 0, len(ys)-1))
    return float(Om_clip[yi, xi])

# ── Static welcome ─────────────────────────────────────────────────────────────
if not run:
    st.info("Adjust parameters in the sidebar, then click **▶ Run simulation**. "
            "The 3D surface is fully interactive — drag to rotate, scroll to zoom.")
    xs, ys, X, Y, Om = potential_grid(0.0121)
    Om_c = np.clip(Om, 1.4, 3.2)
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Om_c, colorscale="Blues", reversescale=True,
        opacity=0.85, showscale=True,
        colorbar=dict(title="Ω", thickness=14),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
    ))
    L4_0 = np.array([0.5 - 0.0121, np.sqrt(3)/2])
    for (lx, ly, col, sym, name) in [
        (-0.0121, 0, "red",    "circle",        "Earth"),
        ( 0.9879, 0, "blue",   "circle",        "Moon"),
        (*L4_0,      "yellow", "diamond",   "L4 (analytic)"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[omega_at([lx, ly], xs, ys, np.clip(Om, 1.4, 3.2))],
            mode="markers", marker=dict(size=8, color=col, symbol=sym),
            name=name
        ))
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="x (rotating frame)",
            yaxis_title="y (rotating frame)",
            zaxis_title="Ω",
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.8)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.6),
        ),
        legend=dict(x=0.01, y=0.99),
        title="Effective potential Ω — drag to rotate, scroll to zoom",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────────
with st.spinner("Running DMM simulation..."):
    (traj, short_log, long_log, grad_log,
     L4, converged_at, final_pos, final_grad,
     bodies) = run_simulation(mu, dx, dy, alpha, beta, mem_cap, gamma, dt, max_steps)
    xs, ys, X, Y, Om = potential_grid(mu)

x1, x2 = bodies
Om_clip = np.clip(Om, 1.4, omega_clip)

# ── Metrics ────────────────────────────────────────────────────────────────────
error = np.linalg.norm(final_pos - L4) if np.isfinite(final_pos).all() else float('nan')
c1, c2, c3, c4 = st.columns(4)
if converged_at is not None:
    c1.metric("Status", "✅ Converged")
    c2.metric("Steps", f"{converged_at:,}")
elif np.isfinite(final_pos).all():
    c1.metric("Status", "⏳ Max steps reached")
    c2.metric("Steps", f"{max_steps:,}")
else:
    c1.metric("Status", "💥 Diverged")
    c2.metric("Steps", "—")
c3.metric("Position error", f"{error:.2e}" if np.isfinite(error) else "—")
c4.metric("|∇Ω| final",    f"{final_grad:.2e}" if np.isfinite(final_grad) else "—")

st.markdown("---")

# ── 3D interactive Plotly chart ────────────────────────────────────────────────
n_pts = len(traj)

fig3d = go.Figure()

# surface
fig3d.add_trace(go.Surface(
    x=X, y=Y, z=Om_clip,
    colorscale="Blues", reversescale=True, opacity=0.75,
    showscale=True, colorbar=dict(title="Ω", thickness=14),
    name="Ω surface", hoverinfo="skip",
))

# trajectory on the surface
if n_pts > 1:
    tz = np.array([omega_at(p, xs, ys, Om_clip) for p in traj])
    # colour by progress
    t_norm = np.linspace(0, 1, n_pts)
    fig3d.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=tz,
        mode="lines",
        line=dict(color=t_norm, colorscale="Plasma", width=4),
        name="Instanton path",
    ))
    # start / end markers
    fig3d.add_trace(go.Scatter3d(
        x=[traj[0, 0]], y=[traj[0, 1]], z=[tz[0]],
        mode="markers", marker=dict(size=7, color="cyan", symbol="square"),
        name=f"Start ({traj[0,0]:.3f}, {traj[0,1]:.3f})",
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[traj[-1, 0]], y=[traj[-1, 1]], z=[tz[-1]],
        mode="markers", marker=dict(size=10, color="black", symbol="diamond"),
        name=f"End ({traj[-1,0]:.4f}, {traj[-1,1]:.4f})",
    ))

# bodies + L4
for (lx, ly, col, sym, lname) in [
    (x1, 0,    "red",    "circle",      "Earth"),
    (x2, 0,    "blue",   "circle",      "Moon"),
    (*L4,      "yellow", "diamond", f"L4 analytic ({L4[0]:.4f}, {L4[1]:.4f})"),
]:
    fig3d.add_trace(go.Scatter3d(
        x=[lx], y=[ly], z=[omega_at([lx, ly], xs, ys, Om_clip)],
        mode="markers", marker=dict(size=9, color=col, symbol=sym,
                                    line=dict(color="black", width=1)),
        name=lname,
    ))

fig3d.update_layout(
    height=640, margin=dict(l=0, r=0, t=40, b=0),
    title="DMM instanton path on Ω surface  —  drag to rotate · scroll to zoom · double-click to reset",
    scene=dict(
        xaxis_title="x (rotating frame)",
        yaxis_title="y (rotating frame)",
        zaxis_title="Ω (eff. potential)",
        camera=dict(eye=dict(x=1.3, y=-1.5, z=0.7)),
        aspectmode="manual",
        aspectratio=dict(x=1.2, y=1.2, z=0.5),
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
)

st.plotly_chart(fig3d, use_container_width=True)

# ── Optional 2D contour ────────────────────────────────────────────────────────
if show_2d:
    fig2d = go.Figure()
    fig2d.add_trace(go.Contour(
        x=xs, y=ys, z=Om_clip,
        colorscale="Blues", reversescale=True, opacity=0.6,
        contours=dict(showlabels=True, labelfont=dict(size=9)),
        colorbar=dict(title="Ω", thickness=12),
        name="Ω contours",
    ))
    if n_pts > 1:
        fig2d.add_trace(go.Scatter(
            x=traj[:, 0], y=traj[:, 1],
            mode="lines",
            line=dict(color="darkorange", width=2),
            name="Instanton path",
        ))
        fig2d.add_trace(go.Scatter(
            x=[traj[0, 0]], y=[traj[0, 1]],
            mode="markers", marker=dict(size=10, color="cyan", symbol="square"),
            name="Start",
        ))
        fig2d.add_trace(go.Scatter(
            x=[traj[-1, 0]], y=[traj[-1, 1]],
            mode="markers", marker=dict(size=12, color="black", symbol="star"),
            name="End",
        ))
    for (lx, ly, col, sym, lname) in [
        (x1, 0, "red",    "circle",        "Earth"),
        (x2, 0, "blue",   "circle",        "Moon"),
        (*L4,   "yellow", "diamond",   "L4 analytic"),
    ]:
        fig2d.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers",
            marker=dict(size=12, color=col, symbol=sym,
                        line=dict(color="black", width=1)),
            name=lname,
        ))
    fig2d.update_layout(
        height=500, title="2D contour view",
        xaxis_title="x (rotating frame)", yaxis_title="y (rotating frame)",
        yaxis_scaleanchor="x",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig2d, use_container_width=True)

# ── Memory dynamics (matplotlib) ──────────────────────────────────────────────
if show_memory and n_pts > 0:
    steps_x = np.arange(len(short_log)) * 10
    fig_mem, ax = plt.subplots(figsize=(11, 3.5))
    ax2 = ax.twinx()

    l1, = ax.plot(steps_x, short_log, color="royalblue",  lw=1.3, label="short_mem  xₛ")
    l2, = ax.plot(steps_x, long_log,  color="darkorange", lw=1.8, label="long_mem  w_L")
    l3, = ax2.semilogy(steps_x, np.clip(grad_log, 1e-10, None),
                        color="seagreen", lw=1.1, alpha=0.85, label="|∇Ω|")
    ax2.axhline(1e-6, ls="--", color="seagreen", lw=0.8, alpha=0.5, label="threshold 10⁻⁶")
    if converged_at:
        ax.axvline(converged_at, ls=":", color="grey", lw=1.2,
                   label=f"Converged @ {converged_at:,}")

    ax.set_xlabel("Simulation step"); ax.set_ylabel("Memory value")
    ax2.set_ylabel("|∇Ω|  (log scale)", color="seagreen")
    ax2.tick_params(axis="y", colors="seagreen")
    ax.set_title("DMM Memory Variables & Constraint Norm")
    ax.grid(alpha=0.3)
    lines  = [l1, l2, l3]; labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig_mem, use_container_width=True)

# ── Equations ──────────────────────────────────────────────────────────────────
with st.expander("Physics & DMM equations", expanded=False):
    st.markdown(r"""
### Effective potential
$$\Omega = \frac{x^2+y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

### Rotating-frame EOM (DMM emulator)
$$\ddot{x} = \underbrace{2\dot{y}}_{\text{Coriolis}} - \underbrace{w_L\,\frac{\partial\Omega}{\partial x}}_{\text{memory force}} - \gamma\dot{x}$$
$$\ddot{y} = \underbrace{-2\dot{x}}_{\text{Coriolis}} - \underbrace{w_L\,\frac{\partial\Omega}{\partial y}}_{\text{memory force}} - \gamma\dot{y}$$

### DMM memory update
$$x_s \leftarrow (1-\alpha)\,x_s + \alpha\,|\nabla\Omega| \qquad \text{(short-term: tracks violation)}$$
$$w_L \leftarrow \min\!\bigl(w_L + \beta\,x_s\,\Delta t,\; w_{cap}\bigr) \qquad \text{(long-term: amplifies force)}$$

### L4 analytic position (equilateral triangle with both primaries)
$$x_{L4} = \tfrac{1}{2}-\mu \approx 0.4879, \qquad y_{L4} = \tfrac{\sqrt{3}}{2} \approx 0.8660, \qquad r_1 = r_2 = 1$$

**Routh's criterion:** L4/L5 stable only for $\mu < 0.0385$. Coriolis is essential — without it, gradient descent falls into the Earth's gravity well.
    """)
