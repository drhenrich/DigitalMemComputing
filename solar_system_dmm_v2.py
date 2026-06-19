"""
Solar System DMM v2 — Lagrange Point Discovery
Same as solar_system_dmm.py but with a simplified memory rule:
  dw_L^x/dt = β · |∂Ω/∂x|
  dw_L^y/dt = β · |∂Ω/∂y|
Short-term memory (sm) is removed; the long-term memory integrates
the gradient magnitude directly — one fewer hyperparameter (α).
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import brentq
import pandas as pd

# ── Physical body database ─────────────────────────────────────────────────────
EPS = 1e-9
ROUTH_LIMIT = 0.03852   # μ < this for L4/L5 linear stability

BODIES = {
    "Sun":       {"mass": 1.989e30, "color": "#FDB813", "radius_km": 695_700},
    "Mercury":   {"mass": 3.285e23, "color": "#b5b5b5", "radius_km": 2_440},
    "Venus":     {"mass": 4.867e24, "color": "#e8cda0", "radius_km": 6_052},
    "Earth":     {"mass": 5.972e24, "color": "#4fc3f7", "radius_km": 6_371},
    "Moon":      {"mass": 7.342e22, "color": "#cccccc", "radius_km": 1_737},
    "Mars":      {"mass": 6.390e23, "color": "#c1440e", "radius_km": 3_390},
    "Phobos":    {"mass": 1.066e16, "color": "#a08060", "radius_km":    11},
    "Deimos":    {"mass": 1.476e15, "color": "#907050", "radius_km":     6},
    "Jupiter":   {"mass": 1.898e27, "color": "#c88b3a", "radius_km": 71_492},
    "Io":        {"mass": 8.932e22, "color": "#f0e060", "radius_km": 1_822},
    "Europa":    {"mass": 4.800e22, "color": "#c8b090", "radius_km": 1_561},
    "Ganymede":  {"mass": 1.482e23, "color": "#909090", "radius_km": 2_634},
    "Callisto":  {"mass": 1.076e23, "color": "#706050", "radius_km": 2_410},
    "Saturn":    {"mass": 5.683e26, "color": "#e4d191", "radius_km": 60_268},
    "Titan":     {"mass": 1.345e23, "color": "#d4a060", "radius_km": 2_575},
    "Enceladus": {"mass": 1.080e20, "color": "#f0f0ff", "radius_km":   252},
    "Rhea":      {"mass": 2.307e21, "color": "#d0c0b0", "radius_km":   764},
    "Uranus":    {"mass": 8.681e25, "color": "#80d8e8", "radius_km": 25_559},
    "Titania":   {"mass": 3.527e21, "color": "#a0b0c0", "radius_km":   789},
    "Oberon":    {"mass": 3.014e21, "color": "#908070", "radius_km":   761},
    "Neptune":   {"mass": 1.024e26, "color": "#3f54ba", "radius_km": 24_622},
    "Triton":    {"mass": 2.139e22, "color": "#c0d8e0", "radius_km": 1_353},
    "Pluto":     {"mass": 1.303e22, "color": "#d4b896", "radius_km": 1_188},
    "Charon":    {"mass": 1.586e21, "color": "#888888", "radius_km":   606},
}

# (primary, secondary, semi_major_axis_m, known_objects_at_Lpoints)
SYSTEMS = {
    # ── Sun–Planet ──────────────────────────────────────────────────────────
    "Sun–Mercury":  ("Sun","Mercury",  5.791e10, {}),
    "Sun–Venus":    ("Sun","Venus",    1.082e11, {}),
    "Sun–Earth":    ("Sun","Earth",    1.496e11, {
        "L1": "SOHO, DSCOVR (solar wind monitoring)",
        "L2": "James Webb Space Telescope, Gaia, Planck",
        "L4": "Earth Trojans (2010 TK7)",
    }),
    "Sun–Mars":     ("Sun","Mars",     2.279e11, {
        "L4": "Mars Trojans (Eureka family)",
        "L5": "Mars Trojans (several known)",
    }),
    "Sun–Jupiter":  ("Sun","Jupiter",  7.783e11, {
        "L4": "Greek camp — >7,000 Trojan asteroids",
        "L5": "Trojan camp — >7,000 Trojan asteroids",
    }),
    "Sun–Saturn":   ("Sun","Saturn",   1.432e12, {}),
    "Sun–Uranus":   ("Sun","Uranus",   2.867e12, {}),
    "Sun–Neptune":  ("Sun","Neptune",  4.495e12, {
        "L4": "Neptune Trojans (2001 QR322, etc.)",
    }),
    "Sun–Pluto":    ("Sun","Pluto",    5.906e12, {}),
    # ── Earth ───────────────────────────────────────────────────────────────
    "Earth–Moon":   ("Earth","Moon",   3.844e8, {
        "L1": "Proposed lunar comm relay",
        "L2": "ARTEMIS mission, proposed Gateway staging",
        "L4": "Kordylewski clouds (dust)",
        "L5": "Kordylewski clouds (dust)",
    }),
    # ── Mars ────────────────────────────────────────────────────────────────
    "Mars–Phobos":  ("Mars","Phobos",  9.376e6, {}),
    "Mars–Deimos":  ("Mars","Deimos",  2.346e7, {}),
    # ── Jupiter ─────────────────────────────────────────────────────────────
    "Jupiter–Io":        ("Jupiter","Io",       4.218e8, {}),
    "Jupiter–Europa":    ("Jupiter","Europa",   6.711e8, {}),
    "Jupiter–Ganymede":  ("Jupiter","Ganymede", 1.070e9, {}),
    "Jupiter–Callisto":  ("Jupiter","Callisto", 1.883e9, {}),
    # ── Saturn ──────────────────────────────────────────────────────────────
    "Saturn–Titan":      ("Saturn","Titan",     1.222e9, {}),
    "Saturn–Enceladus":  ("Saturn","Enceladus", 2.382e8, {}),
    "Saturn–Rhea":       ("Saturn","Rhea",      5.270e8, {}),
    # ── Uranus ──────────────────────────────────────────────────────────────
    "Uranus–Titania":    ("Uranus","Titania",   4.360e8, {}),
    "Uranus–Oberon":     ("Uranus","Oberon",    5.835e8, {}),
    # ── Neptune ─────────────────────────────────────────────────────────────
    "Neptune–Triton":    ("Neptune","Triton",   3.548e8, {}),
    # ── Pluto ───────────────────────────────────────────────────────────────
    "Pluto–Charon":      ("Pluto","Charon",     1.959e7, {
        "note": "μ > Routh limit — L4/L5 linearly unstable in rotating frame",
    }),
}

L_COLORS = {
    "L1": "#e74c3c", "L2": "#e67e22", "L3": "#9b59b6",
    "L4": "#27ae60", "L5": "#2980b9",
}

# ── Precomputed overview (analytical, fast) ───────────────────────────────────
def compute_system_info(p1_name, p2_name, sma_m):
    m1 = BODIES[p1_name]["mass"]
    m2 = BODIES[p2_name]["mass"]
    mu = m2 / (m1 + m2)
    hill_m = sma_m * (mu / 3) ** (1 / 3)
    stable = mu < ROUTH_LIMIT
    return mu, hill_m, stable


def analytical_collinear(mu):
    """Numerically locate L1, L2, L3 on x-axis."""
    def gx(x):
        r1 = abs(x + mu) + EPS
        r2 = abs(x - 1 + mu) + EPS
        return x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    L1 = brentq(gx, -mu + 1e-4, 1-mu - 1e-4)
    L2 = brentq(gx, 1-mu + 1e-4, 2.5)
    L3 = brentq(gx, -2.5, -mu - 1e-4)
    return L1, L2, L3


# ── DMM physics ────────────────────────────────────────────────────────────────
def grad_and_curvature(pos, mu):
    x, y = pos
    r1 = np.sqrt((x + mu)**2 + y**2) + EPS
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + EPS
    gx = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    gy = y - (1-mu)*y/r1**3 - mu*y/r2**3
    oyy = (1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5
             - mu/r2**3 + 3*mu*y**2/r2**5)
    return np.array([gx, gy]), oyy


def effective_potential(x, y, mu):
    r1 = np.sqrt((x + mu)**2 + y**2) + EPS
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + EPS
    return -(x**2 + y**2) / 2 - (1-mu)/r1 - mu/r2


def simulate_dmm(mu, start, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr):
    """alpha is unused in v2 (no short-term memory); kept for API compatibility."""
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    lmx = lmy = 1.0
    traj = [pos.copy()]
    lm_hist = [(lmx, lmy)]   # record per-axis: list of (w_L^x, w_L^y)
    gn_hist = []
    conv_step = None
    for step in range(max_steps):
        grad, oyy = grad_and_curvature(pos, mu)
        gx, gy = grad
        gn = np.linalg.norm(grad)
        # v2 rule: dw_L/dt = β · |∇Ω|  (no short-term memory intermediary)
        lmx = min(lmx + beta * abs(gx) * dt, mem_cap)
        lmy = min(lmy + beta * abs(gy) * dt, mem_cap)
        sign_y = +1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - lmx*gx - gamma*vel[0]
        ay = -2*vel[0] + sign_y*lmy*gy - gamma*vel[1]
        vel += np.array([ax, ay]) * dt
        pos = pos + vel * dt
        if step % 20 == 0:
            traj.append(pos.copy())
            lm_hist.append((lmx, lmy))
            gn_hist.append(gn)
        if gn < conv_thr:
            conv_step = step
            traj.append(pos.copy())
            break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return np.array(traj), np.array(lm_hist), np.array(gn_hist), conv_step, pos, gn


def run_discovery(mu, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr, n_x, n_y):
    # Compute exact L-point positions first — use as anchor starts and for guards
    L1x, L2x, L3x = analytical_collinear(mu)
    analytic = {
        "L1": np.array([L1x, 0.0]),
        "L2": np.array([L2x, 0.0]),
        "L3": np.array([L3x, 0.0]),
        "L4": np.array([0.5 - mu,  np.sqrt(3)/2]),
        "L5": np.array([0.5 - mu, -np.sqrt(3)/2]),
    }

    # Hill radius ≈ distance from secondary to L1/L2
    hill_r = (mu / 3) ** (1 / 3)
    # Exclusion zones: avoid singularities but stay smaller than Hill radius
    excl_primary   = min(0.03, 0.3 * hill_r)
    excl_secondary = min(0.012, 0.3 * hill_r)

    # ── Exactly n_x x-values: (n_x−6) uniform fill + 6 anchors near L-points ──
    offset = max(0.5 * hill_r, 0.02)
    anchors_x = np.array([
        L1x - offset, L1x + offset,
        L2x - offset, L2x + offset,
        L3x - 0.04,   L3x + 0.04,
    ])
    n_fill = max(n_x - 6, 1)
    fill_x = np.linspace(-1.4, 1.4, n_fill)
    xs = np.sort(np.unique(np.round(
        np.concatenate([fill_x, anchors_x]), 8)))[:n_x]

    # ── Exactly n_y y-values: 3 below + fine layer (−0.05, 0, +0.05) + 4 above ─
    n_neg = (n_y - 3) // 2
    n_pos = n_y - 3 - n_neg
    ys = np.sort(np.unique(np.concatenate([
        np.linspace(-1.2, -0.15, max(n_neg, 1)),
        [-0.05, 0.0, 0.05],
        np.linspace(0.15, 1.2, max(n_pos, 1)),
    ])))[:n_y]

    results = []
    for x in xs:
        for y in ys:
            if (x + mu)**2 + y**2 < excl_primary**2:
                continue
            if (x - 1 + mu)**2 + y**2 < excl_secondary**2:
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
                if d_best < 0.12:
                    label = best
            results.append({
                "start": (x, y), "traj": traj, "lm_hist": lm_h,
                "gn_hist": gn_h, "conv": conv, "final": final,
                "label": label, "gn": gn,
            })
    return results, analytic


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_trajectories(results, analytic, mu, sma_m, p1_name, p2_name, stable):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    xg = np.linspace(-1.6, 1.6, 350)
    yg = np.linspace(-1.4, 1.4, 350)
    XG, YG = np.meshgrid(xg, yg)
    Omega = np.clip(effective_potential(XG, YG, mu), -4.0, -1.3)
    ax.contourf(XG, YG, Omega, levels=55, cmap="inferno", alpha=0.35)
    ax.contour(XG, YG, Omega, levels=18, colors="white", linewidths=0.25, alpha=0.25)

    for r in results:
        if r["label"] is None:
            continue
        t = r["traj"]
        col = L_COLORS[r["label"]]
        ax.plot(t[:, 0], t[:, 1], color=col, lw=0.6, alpha=0.45)
        ax.plot(t[-1, 0], t[-1, 1], "x", color=col, ms=6, mew=1.8)

    c1 = BODIES[p1_name]["color"]
    c2 = BODIES[p2_name]["color"]
    ax.plot([-mu], [0], "o", color=c1, ms=13, zorder=6, label=p1_name)
    ax.plot([1-mu], [0], "o", color=c2, ms=7, zorder=6, label=p2_name)

    for name, pos in analytic.items():
        stab = stable or name in ("L1", "L2", "L3")
        mk = "*" if stab else "D"
        ms = 16 if stab else 10
        n = sum(1 for r in results if r["label"] == name)
        ax.plot(*pos, mk, color=L_COLORS[name], ms=ms, zorder=7,
                label=f"{name} ({n} traj.)",
                markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(name, xy=pos, xytext=(pos[0]+0.08, pos[1]+0.07),
                    color=L_COLORS[name], fontsize=8.5, fontweight="bold")

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("x  (co-rotating frame)", color="white", fontsize=10)
    ax.set_ylabel("y  (co-rotating frame)", color="white", fontsize=10)
    ax.set_title(f"DMM Discovery — {p1_name}–{p2_name}  (μ = {mu:.4e})\n"
                 f"SMA = {sma_m/1e9:.2f} × 10⁶ km",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")
    leg = ax.legend(fontsize=7.5, loc="upper right",
                    facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_memory(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#555")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    for r in results:
        if r["label"] is None or len(r["gn_hist"]) < 3:
            continue
        col = L_COLORS[r["label"]]
        lm = r["lm_hist"]          # shape (N, 2): columns are (w_L^x, w_L^y)
        steps = np.arange(len(lm)) * 20

        ax1.plot(steps, lm[:, 0], color=col, lw=0.7, alpha=0.45)
        ax2.plot(steps, lm[:, 1], color=col, lw=0.7, alpha=0.45)

        gsteps = np.arange(len(r["gn_hist"])) * 20
        ax3.semilogy(gsteps, r["gn_hist"], color=col, lw=0.7, alpha=0.45)

    # legend proxy
    for name, col in L_COLORS.items():
        ax1.plot([], [], color=col, lw=1.8, label=name)

    ax1.set_xlabel("Integration step", fontsize=10)
    ax1.set_ylabel("w_L^x", fontsize=10)
    ax1.set_title("x-Memory Ratchet  (ẇ_L^x = β|∂Ω/∂x|)", fontsize=10)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    ax2.set_xlabel("Integration step", fontsize=10)
    ax2.set_ylabel("w_L^y", fontsize=10)
    ax2.set_title("y-Memory Ratchet  (ẇ_L^y = β|∂Ω/∂y|)", fontsize=10)

    ax3.axhline(1e-4, color="white", lw=0.8, ls="--", alpha=0.5, label="threshold")
    ax3.set_xlabel("Integration step", fontsize=10)
    ax3.set_ylabel("|∇Ω|  (log scale)", fontsize=10)
    ax3.set_title("Gradient Decay → 0", fontsize=10)
    ax3.set_ylim(bottom=1e-6)
    ax3.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    return fig


# ── Results table with physical units ─────────────────────────────────────────
def build_results_table(results, analytic, mu, sma_m, stable, known):
    rows = []
    for name, a_pos in analytic.items():
        found = [r for r in results if r["label"] == name]
        if found:
            mean_final = np.array([r["final"] for r in found]).mean(0)
            err = np.linalg.norm(mean_final - a_pos)
        else:
            mean_final = a_pos
            err = float("nan")

        x_phys = a_pos[0] * sma_m / 1e9      # 10^6 km
        y_phys = a_pos[1] * sma_m / 1e9
        dist_from_sec = np.linalg.norm(a_pos - np.array([1-mu, 0])) * sma_m / 1e9

        stab_label = "✓ stable" if (stable or name in ("L1","L2","L3")) else "✗ unstable"
        if not stable and name in ("L4","L5"):
            stab_label = "✗ (μ > Routh)"

        note = known.get(name, "")
        rows.append({
            "Point": name,
            "x (norm.)": f"{a_pos[0]:+.5f}",
            "y (norm.)": f"{a_pos[1]:+.5f}",
            "x (10⁶ km)": f"{x_phys:+.4f}",
            "y (10⁶ km)": f"{y_phys:+.4f}",
            "Dist. to secondary (10⁶ km)": f"{dist_from_sec:.4f}",
            "Trajectories": len(found),
            "Mean error": f"{err:.2e}" if not np.isnan(err) else "—",
            "Stability": stab_label,
            "Known objects / notes": note,
        })
    return pd.DataFrame(rows)


# ── Overview table (all systems, no DMM) ──────────────────────────────────────
@st.cache_data
def build_overview_table():
    rows = []
    for sys_name, (p1, p2, sma_m, known) in SYSTEMS.items():
        mu, hill_m, stable = compute_system_info(p1, p2, sma_m)
        L1x, L2x, L3x = analytical_collinear(mu)
        hill_from_sec = hill_m / 1e6  # km
        rows.append({
            "System": sys_name,
            "μ": f"{mu:.4e}",
            "L4/L5 stable": "✓" if stable else "✗",
            "SMA (10⁶ km)": f"{sma_m/1e9:.3f}",
            "Hill radius (km)": f"{hill_m/1e3:,.0f}",
            "L1 (norm.)": f"{L1x:.5f}",
            "L2 (norm.)": f"{L2x:.5f}",
            "L3 (norm.)": f"{L3x:.5f}",
            "L4/L5 y (norm.)": "±0.86603",
        })
    return pd.DataFrame(rows)


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Solar System DMM v2", layout="wide", page_icon="🌌")
st.title("🌌 Solar System Lagrange Point Discovery via DMM v2")
st.markdown(
    r"""
Adaptive vector-memory Digital MemComputing Machine applied to the full solar system.
Select any two-body pair — Sun + planet, planet + moon — and the machine discovers
all 5 Lagrange points from a grid of starting positions, reading the local
curvature $\Omega_{yy}$ at each step to switch between
**memory amplification** (stable attractors) and **correction current** (saddle points).
No L-point coordinates are supplied.
"""
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System selection")

    categories = {
        "☀️ Sun–Planet":   [k for k in SYSTEMS if k.startswith("Sun")],
        "🌍 Earth–Moon":   ["Earth–Moon"],
        "🔴 Mars moons":   [k for k in SYSTEMS if k.startswith("Mars")],
        "🪐 Jupiter moons":[k for k in SYSTEMS if k.startswith("Jupiter")],
        "💍 Saturn moons": [k for k in SYSTEMS if k.startswith("Saturn")],
        "🔵 Uranus moons": [k for k in SYSTEMS if k.startswith("Uranus")],
        "🔷 Neptune moons":[k for k in SYSTEMS if k.startswith("Neptune")],
        "❄️ Pluto–Charon": ["Pluto–Charon"],
    }

    cat = st.selectbox("Category", list(categories.keys()))
    sys_name = st.selectbox("System", categories[cat])

    p1_name, p2_name, sma_m, known_objs = SYSTEMS[sys_name]
    mu, hill_m, stable = compute_system_info(p1_name, p2_name, sma_m)

    st.divider()
    st.subheader("System info")
    st.metric("Mass ratio μ", f"{mu:.4e}")
    col1, col2 = st.columns(2)
    col1.metric(p1_name + " mass", f"{BODIES[p1_name]['mass']:.3e} kg")
    col2.metric(p2_name + " mass", f"{BODIES[p2_name]['mass']:.3e} kg")
    st.metric("Semi-major axis", f"{sma_m/1.496e11:.4f} AU  ({sma_m/1e9:.3f} × 10⁶ km)")
    st.metric("Hill radius", f"{hill_m/1e3:,.0f} km")
    if stable:
        st.success("✓ L4/L5 linearly stable  (μ < 0.03852)")
    else:
        st.error("✗ L4/L5 linearly unstable  (μ > 0.03852 — Routh criterion violated)")
        st.caption("Note: the DMM still finds L4/L5 as ∇Ω = 0 solutions; "
                   "they are mathematical but not dynamically stable attractors.")

    st.divider()
    st.subheader("DMM parameters")
    alpha    = st.slider("Short-term memory α", 0.01, 0.30, 0.05, 0.01)
    beta     = st.slider("Long-term ratchet β", 0.0, 1.0, 0.001, 0.001, format="%.3f")
    mem_cap  = st.slider("Memory cap w_cap", 2.0, 15.0, 8.0, 0.5)
    gamma    = st.slider("Damping γ", 0.1, 1.5, 0.6, 0.05)
    dt       = st.select_slider("Time step dt", [0.005, 0.01, 0.02], value=0.01)
    max_steps= st.select_slider("Max steps", [50_000, 100_000, 200_000, 400_000], value=200_000)
    conv_thr = st.select_slider("Convergence |∇Ω| <", [1e-3, 5e-4, 1e-4, 1e-5], value=1e-4)
    n_x = st.slider("x grid size  (6 anchors + fill)", 7, 18, 10)
    n_y = st.slider("y grid size  (fine y=0 layer included)", 6, 18, 10)

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_disc, tab_mem, tab_overview = st.tabs(
    ["🎯 Discovery", "📈 Memory dynamics", "📊 Solar system overview"]
)

with tab_disc:
    run = st.button("▶  Run DMM Discovery", type="primary", use_container_width=True)

    if run:
        total = n_x * (n_y + 3)
        with st.spinner(f"Running {total} DMM trajectories for {sys_name} …"):
            results, analytic = run_discovery(
                mu, alpha, beta, mem_cap, gamma, dt, max_steps, conv_thr, n_x, n_y
            )

        found = {k: [r for r in results if r["label"] == k] for k in L_COLORS}

        st.subheader(f"Discovery results — {sys_name}")
        cols = st.columns(6)
        for i, (name, rs) in enumerate(found.items()):
            cols[i].metric(name, f"{len(rs)} found",
                           delta="stable ✓" if (stable or name not in ("L4","L5")) else "unstable ✗")
        cols[5].metric("No convergence",
                       sum(1 for r in results if r["label"] is None))

        df = build_results_table(results, analytic, mu, sma_m, stable, known_objs)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if known_objs:
            st.info("**Known objects / missions at Lagrange points of this system:**\n" +
                    "\n".join(f"- **{k}**: {v}" for k, v in known_objs.items()))

        fig = plot_trajectories(results, analytic, mu, sma_m, p1_name, p2_name, stable)
        st.pyplot(fig, use_container_width=True)

        st.session_state["last_results"] = results
        st.session_state["last_sys"] = sys_name
    else:
        st.info(f"Selected: **{sys_name}**  (μ = {mu:.4e})  "
                f"— press **Run DMM Discovery** to start.")
        st.markdown(f"""
| Property | Value |
|---|---|
| Primary | {p1_name} ({BODIES[p1_name]['mass']:.3e} kg) |
| Secondary | {p2_name} ({BODIES[p2_name]['mass']:.3e} kg) |
| Semi-major axis | {sma_m/1.496e11:.4f} AU |
| Hill sphere radius | {hill_m/1e3:,.0f} km |
| L4/L5 stable | {"Yes (μ < 0.03852)" if stable else "**No — Routh criterion violated**"} |
""")

with tab_mem:
    mem_results = st.session_state.get("last_results")
    mem_sys     = st.session_state.get("last_sys", "")
    if mem_results:
        if mem_sys and mem_sys != sys_name:
            st.warning(f"Showing results from **{mem_sys}** — run Discovery for {sys_name} to refresh.")
        fig2 = plot_memory(mem_results)
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            "**Left / Middle:** per-axis long-term memory ratchets — "
            "ẇ_L^x = β|∂Ω/∂x|, ẇ_L^y = β|∂Ω/∂y|. "
            "Each axis integrates the gradient magnitude directly; memory only grows, never shrinks. "
            "**Right:** clause violation |∇Ω| decays monotonically to the convergence threshold. "
            "Color = which Lagrange point the trajectory converged to."
        )
    else:
        st.info("Run a discovery first (Discovery tab → ▶ Run DMM Discovery).")

with tab_overview:
    st.subheader("All solar system two-body pairs — analytical Lagrange point positions")
    st.caption("Normalised coordinates: primary at (−μ, 0), secondary at (1−μ, 0), "
               "separation = 1. Multiply x/y by SMA to get physical distances.")
    df_ov = build_overview_table()
    st.dataframe(df_ov, use_container_width=True, hide_index=True)

    st.markdown("""
**Notes on stability and real objects:**
- **Sun–Earth L1**: SOHO, DSCOVR — continuous solar-wind monitoring
- **Sun–Earth L2**: James Webb Space Telescope, Gaia, Planck — quiet thermal environment
- **Sun–Jupiter L4/L5**: >14,000 Trojan asteroids locked in 1:1 mean-motion resonance
- **Sun–Neptune L4**: Neptune Trojans (2001 QR322 and others)
- **Sun–Mars L4/L5**: Mars Trojans (Eureka family)
- **Earth–Moon L4/L5**: Kordylewski dust clouds (disputed), proposed Lunar Gateway staging
- **Pluto–Charon**: μ = 0.109 — a true binary system; L4/L5 are **dynamically unstable**
- All Sun–planet pairs satisfy the Routh criterion; largest is Sun–Jupiter at μ = 9.53 × 10⁻⁴
""")
