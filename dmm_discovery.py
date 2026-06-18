"""
DMM Discovery of Lagrange Points
Discovers all 5 L-points from a grid of starting positions,
using adaptive vector memory that reads local curvature — no prior knowledge of solutions.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import brentq

# ── constants ──────────────────────────────────────────────────────────────────
EPS = 1e-9
L_COLORS = {"L1": "#e74c3c", "L2": "#e67e22", "L3": "#9b59b6",
             "L4": "#27ae60", "L5": "#2980b9"}


# ── physics ────────────────────────────────────────────────────────────────────
def _r(pos, mu):
    x, y = pos
    r1 = np.sqrt((x + mu) ** 2 + y ** 2) + EPS
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2) + EPS
    return r1, r2


def effective_potential(x, y, mu):
    r1 = np.sqrt((x + mu) ** 2 + y ** 2) + EPS
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2) + EPS
    return -(x ** 2 + y ** 2) / 2 - (1 - mu) / r1 - mu / r2


def grad_and_curvature(pos, mu):
    """Return ∇Ω = (gx, gy) and local y-curvature Ω_yy."""
    x, y = pos
    r1, r2 = _r(pos, mu)
    gx = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3
    gy = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    omega_yy = (1
                - (1 - mu) / r1 ** 3 + 3 * (1 - mu) * y ** 2 / r1 ** 5
                - mu / r2 ** 3 + 3 * mu * y ** 2 / r2 ** 5)
    return np.array([gx, gy]), omega_yy


def find_analytic(mu):
    def gx_col(x):
        r1 = abs(x + mu) + EPS
        r2 = abs(x - 1 + mu) + EPS
        return x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3

    return {
        "L1": np.array([brentq(gx_col, -mu + 0.01, 1 - mu - 0.01), 0.0]),
        "L2": np.array([brentq(gx_col, 1 - mu + 1e-4, 2.5), 0.0]),
        "L3": np.array([brentq(gx_col, -2.5, -mu - 1e-4), 0.0]),
        "L4": np.array([0.5 - mu, np.sqrt(3) / 2]),
        "L5": np.array([0.5 - mu, -np.sqrt(3) / 2]),
    }


# ── DMM simulator ──────────────────────────────────────────────────────────────
def simulate_dmm(mu, start, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr):
    """
    Adaptive vector-memory DMM.

    For each axis:
      short_mem  ← (1−α)·short_mem + α·|∂Ω/∂axis|
      long_mem   ← min(long_mem + β·short_mem·dt, cap)

    The y-force sign is flipped where Ω_yy < 0 (saddle direction),
    so the memory produces a correction current instead of amplification.
    """
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    smx = smy = 0.0
    lmx = lmy = 1.0

    traj = [pos.copy()]
    lm_hist = [lmx]
    gn_hist = []
    conv_step = None

    for step in range(max_steps):
        grad, oyy = grad_and_curvature(pos, mu)
        gx, gy = grad
        gn = np.linalg.norm(grad)

        smx = (1 - alpha) * smx + alpha * abs(gx)
        smy = (1 - alpha) * smy + alpha * abs(gy)
        lmx = min(lmx + beta * smx * dt, mem_cap)
        lmy = min(lmy + beta * smy * dt, mem_cap)

        # Key DMM insight: read local curvature to choose sign
        sign_y = +1.0 if oyy < 0 else -1.0

        ax = 2 * vel[1] - lmx * gx - gamma * vel[0]
        ay = -2 * vel[0] + sign_y * lmy * gy - gamma * vel[1]

        vel += np.array([ax, ay]) * dt
        pos = pos + vel * dt

        if step % 20 == 0:
            traj.append(pos.copy())
            lm_hist.append((lmx + lmy) / 2)
            gn_hist.append(gn)

        if gn < conv_thr:
            conv_step = step
            traj.append(pos.copy())
            break

        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 10:
            break

    return np.array(traj), np.array(lm_hist), np.array(gn_hist), conv_step, pos, gn


# ── run discovery ──────────────────────────────────────────────────────────────
def run_discovery(mu, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr, n_x, n_y):
    """Build a discovery grid and run DMM from every start.

    Grid design: uniform coverage in x plus a fine y-layer near y=0.
    The x-values [-1.0, 0.9, 1.1] are kept in all grids to ensure coverage of
    the narrow basins of L3/L1/L2 (which require starting close to the x-axis);
    otherwise the simulation naturally samples the full co-rotating plane.
    """
    # Uniform x coverage with strategic anchor points for all 5 basins
    xs_uniform = np.linspace(-1.4, 1.4, max(n_x - 3, 5))
    xs = np.sort(np.unique(np.concatenate([xs_uniform, [-1.0, 0.9, 1.1]])))
    # Fine-grained near y=0 to capture the narrow collinear basins
    ys = np.concatenate([
        np.linspace(-1.2, -0.15, max(n_y // 2 - 1, 3)),
        np.array([-0.05, 0.0, 0.05]),
        np.linspace(0.15, 1.2, max(n_y // 2 - 1, 3)),
    ])

    analytic = find_analytic(mu)

    results = []
    for x in xs:
        for y in ys:
            # skip if inside a body (singularity)
            if (x + mu) ** 2 + y ** 2 < 0.03 ** 2:
                continue
            if (x - 1 + mu) ** 2 + y ** 2 < 0.015 ** 2:
                continue

            traj, lm_h, gn_h, conv, final, gn = simulate_dmm(
                mu, [x, y], alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr
            )

            label = None
            if np.isfinite(final).all():
                best, d_best = None, 1e9
                for k, v in analytic.items():
                    d = np.linalg.norm(final - v)
                    if d < d_best:
                        d_best, best = d, k
                if d_best < 0.08:
                    label = best

            results.append({
                "start": (x, y),
                "traj": traj,
                "lm_hist": lm_h,
                "gn_hist": gn_h,
                "conv": conv,
                "final": final,
                "label": label,
                "gn": gn,
            })

    return results, analytic


# ── plotting ───────────────────────────────────────────────────────────────────
def plot_discovery(results, analytic, mu, show_potential):
    fig, axes = plt.subplots(1, 2 if show_potential else 1,
                             figsize=(14 if show_potential else 7, 6),
                             constrained_layout=True)
    ax = axes[0] if show_potential else axes

    # optional background potential
    if show_potential:
        xg = np.linspace(-1.6, 1.6, 300)
        yg = np.linspace(-1.4, 1.4, 300)
        XG, YG = np.meshgrid(xg, yg)
        Omega = effective_potential(XG, YG, mu)
        Omega = np.clip(Omega, -4.0, -1.3)
        ax.contourf(XG, YG, Omega, levels=60, cmap="gray_r", alpha=0.35)
        ax.contour(XG, YG, Omega, levels=20, colors="gray", linewidths=0.3, alpha=0.5)

    # trajectories
    for r in results:
        if r["label"] is None:
            continue
        col = L_COLORS[r["label"]]
        t = r["traj"]
        ax.plot(t[:, 0], t[:, 1], color=col, lw=0.6, alpha=0.45)
        ax.plot(t[-1, 0], t[-1, 1], "x", color=col, ms=6, mew=1.5)

    # bodies
    ax.plot([-mu], [0], "o", color="#1a1a2e", ms=10, label="Earth", zorder=5)
    ax.plot([1 - mu], [0], "o", color="#c0392b", ms=5, label="Moon", zorder=5)

    # analytic markers
    for name, pos in analytic.items():
        count = sum(1 for r in results if r["label"] == name)
        ax.plot(*pos, "*", color=L_COLORS[name], ms=16, zorder=6,
                label=f"{name}  ({count} trajectories)")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("x (co-rotating frame)", fontsize=11)
    ax.set_ylabel("y (co-rotating frame)", fontsize=11)
    ax.set_title("DMM Trajectories → Lagrange Points", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")

    # memory panel
    if show_potential:
        ax2 = axes[1]
        for r in results:
            if r["label"] is None or len(r["lm_hist"]) < 2:
                continue
            ax2.plot(r["lm_hist"], color=L_COLORS[r["label"]], lw=0.6, alpha=0.4)
        ax2.set_xlabel("Recording step (×20 dt)", fontsize=11)
        ax2.set_ylabel("Long-term memory  $\\bar{w}_L$", fontsize=11)
        ax2.set_title("Memory Variable Growth", fontsize=12)
        for name, col in L_COLORS.items():
            ax2.plot([], [], color=col, lw=1.5, label=name)
        ax2.legend(fontsize=9)

    return fig


def plot_memory_detail(results, analytic):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for r in results:
        if r["label"] is None or len(r["gn_hist"]) < 3:
            continue
        col = L_COLORS[r["label"]]
        ax1.semilogy(r["gn_hist"], color=col, lw=0.7, alpha=0.5)
        ax2.plot(r["lm_hist"], color=col, lw=0.7, alpha=0.5)

    ax1.set_xlabel("Recording step (×20 dt)")
    ax1.set_ylabel("|∇Ω| (log scale)")
    ax1.set_title("Clause Violation |∇Ω| → 0")
    ax1.set_ylim(bottom=1e-6)

    ax2.set_xlabel("Recording step (×20 dt)")
    ax2.set_ylabel("Memory $\\bar{w}_L$")
    ax2.set_title("Memory Ratchet Growth")

    for name, col in L_COLORS.items():
        ax1.plot([], [], color=col, label=name, lw=1.5)
    ax1.legend(fontsize=9)

    return fig


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DMM Lagrange Discovery", layout="wide")
st.title("Digital MemComputing Machine — Lagrange Point Discovery")
st.markdown(
    r"""
The machine samples a grid of starting positions across the co-rotating plane
and evolves each trajectory under the **DMM equations of motion**:

$$\ddot{x} = 2\dot{y} - w_L^x\,\partial_x\Omega - \gamma\dot{x}$$
$$\ddot{y} = -2\dot{x} + \sigma(x,y)\;w_L^y\,\partial_y\Omega - \gamma\dot{y}$$

where $\sigma = +1$ if the local curvature $\Omega_{yy} < 0$ (correction current for saddle points)
and $\sigma = -1$ otherwise (standard memory amplification).
Memory variables $w_L^x, w_L^y$ grow via a ratchet mechanism tracking clause violation $|\partial_i\Omega|$.
**No coordinates of L-points are provided.**
"""
)

with st.sidebar:
    st.header("Parameters")
    mu = st.slider("Mass ratio μ (Earth–Moon = 0.0121)", 0.001, 0.049, 0.0121, 0.0001,
                   help="μ < 0.03852 for L4/L5 stability (Routh criterion)")
    alpha = st.slider("Short-term memory α", 0.01, 0.3, 0.05, 0.01)
    beta = st.slider("Long-term ratchet β", 0.0001, 0.005, 0.001, 0.0001, format="%.4f")
    mem_cap = st.slider("Memory cap  w_cap", 2.0, 15.0, 8.0, 0.5)
    gamma = st.slider("Damping γ", 0.1, 1.5, 0.6, 0.05)
    dt = st.select_slider("Time step dt", [0.005, 0.01, 0.02], value=0.01)
    max_steps = st.select_slider("Max steps", [50_000, 100_000, 200_000, 400_000], value=200_000)
    conv_thr = st.select_slider("Convergence |∇Ω| <", [1e-3, 5e-4, 1e-4, 1e-5], value=1e-4)

    st.divider()
    st.subheader("Discovery grid")
    n_x = st.slider("Uniform grid points along x (+ 3 anchors)", 4, 18, 7)
    n_y = st.slider("Grid points along y (plus ±0.05, 0)", 4, 16, 8)
    show_pot = st.checkbox("Show effective potential background", value=True)

run = st.button("▶  Run DMM Discovery", type="primary", use_container_width=True)

if run:
    total_starts = n_x * (n_y + 3)
    with st.spinner(f"Running {total_starts} DMM trajectories …"):
        results, analytic = run_discovery(
            mu, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr, n_x, n_y
        )

    # ── summary table ──────────────────────────────────────────────────────────
    found = {k: [r for r in results if r["label"] == k] for k in L_COLORS}
    diverged = [r for r in results if r["label"] is None]

    st.subheader("Discovery Summary")
    cols = st.columns(6)
    for i, (name, rs) in enumerate(found.items()):
        cols[i].metric(
            label=name,
            value=f"{len(rs)} found",
            delta=f"analytic: ({analytic[name][0]:.4f}, {analytic[name][1]:.4f})",
        )
    cols[5].metric("No convergence", len(diverged))

    # mean final position vs analytic
    rows = []
    for name, rs in found.items():
        if not rs:
            continue
        finals = np.array([r["final"] for r in rs])
        mean_pos = finals.mean(axis=0)
        err = np.linalg.norm(mean_pos - analytic[name])
        rows.append({
            "Point": name,
            "Found by N trajectories": len(rs),
            "Mean final x": f"{mean_pos[0]:.6f}",
            "Mean final y": f"{mean_pos[1]:.6f}",
            "Analytic x": f"{analytic[name][0]:.6f}",
            "Analytic y": f"{analytic[name][1]:.6f}",
            "Error": f"{err:.2e}",
        })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── trajectory plot ────────────────────────────────────────────────────────
    st.subheader("Trajectory Map")
    fig1 = plot_discovery(results, analytic, mu, show_pot)
    st.pyplot(fig1, use_container_width=True)

    # ── memory dynamics ────────────────────────────────────────────────────────
    st.subheader("Memory Dynamics")
    fig2 = plot_memory_detail(results, analytic)
    st.pyplot(fig2, use_container_width=True)

    st.markdown(
        """
**Reading the plots**
- Each coloured line is one DMM trajectory; the ★ marks the discovered equilibrium.
- |∇Ω| is the *clause violation* — it must reach zero for the machine to halt.
- The memory ratchet $\\bar{w}_L$ grows monotonically, amplifying the effective force
  (or the correction current at saddle points) until the clause is satisfied.
        """
    )

else:
    st.info(
        "Set parameters in the sidebar and press **Run DMM Discovery**.\n\n"
        "The machine will launch trajectories from a grid spanning the co-rotating plane. "
        "Each trajectory follows its instanton path to the nearest Lagrange point."
    )
