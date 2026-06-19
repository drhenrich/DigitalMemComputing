"""
Publication-quality figure generation for dmm_lagrange.tex.
Produces three PDF figures with white background, no internal titles,
one representative trajectory per L-point.

Usage:
    python generate_paper_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import brentq

# ── Parameters ────────────────────────────────────────────────────────────────
MU       = 0.01215        # Earth–Moon mass ratio
ALPHA    = 0.05           # (unused in v2, kept for call signature)
BETA     = 0.001          # memory growth rate
MEM_CAP  = 8.0
GAMMA    = 0.6
DT       = 0.01
MAX_STEP = 200_000
CONV_THR = 1e-4

L_COLORS = {
    "L1": "#e74c3c", "L2": "#e67e22", "L3": "#9b59b6",
    "L4": "#27ae60", "L5": "#2980b9",
}

# Publication style
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.linewidth":    1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.7",
})

# ── Physics ───────────────────────────────────────────────────────────────────
def grad_and_curvature(pos, mu):
    x, y = pos
    r1 = np.sqrt((x + mu)**2 + y**2) + 1e-9
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + 1e-9
    gx = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    gy = y - (1-mu)*y/r1**3       - mu*y/r2**3
    oyy = (1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5
             - mu/r2**3      + 3*mu*y**2/r2**5)
    return np.array([gx, gy]), oyy

def effective_potential(x, y, mu):
    r1 = np.sqrt((x + mu)**2 + y**2) + 1e-9
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + 1e-9
    return (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2

def analytical_collinear(mu):
    def gx(x): return x - (1-mu)*(x+mu)/abs(x+mu)**3 - mu*(x-1+mu)/abs(x-1+mu)**3
    L1 = brentq(gx, -mu+1e-4, 1-mu-1e-4)
    L2 = brentq(gx, 1-mu+1e-4, 2.5)
    L3 = brentq(gx, -2.5, -mu-1e-4)
    return L1, L2, L3

def simulate_dmm(mu, start, beta, mem_cap, gamma, dt, max_steps, conv_thr):
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    lmx = lmy = 1.0
    traj   = [pos.copy()]
    lm_h   = [(lmx, lmy)]
    gn_h   = []
    conv_step = None
    for step in range(max_steps):
        grad, oyy = grad_and_curvature(pos, mu)
        gx, gy = grad
        gn = np.linalg.norm(grad)
        lmx = min(lmx + beta*abs(gx)*dt, mem_cap)
        lmy = min(lmy + beta*abs(gy)*dt, mem_cap)
        sign_y = +1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - lmx*gx - gamma*vel[0]
        ay = -2*vel[0] + sign_y*lmy*gy - gamma*vel[1]
        vel += np.array([ax, ay]) * dt
        pos  = pos + vel * dt
        if step % 20 == 0:
            traj.append(pos.copy())
            lm_h.append((lmx, lmy))
            gn_h.append(gn)
        if gn < conv_thr:
            conv_step = step
            traj.append(pos.copy())
            break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return np.array(traj), np.array(lm_h), np.array(gn_h), conv_step, pos, gn


def run_discovery_100(mu, beta, mem_cap, gamma, dt, max_steps, conv_thr):
    """Run on exactly 10×10 = 100 starting positions."""
    L1x, L2x, L3x = analytical_collinear(mu)
    analytic = {
        "L1": np.array([L1x, 0.0]),
        "L2": np.array([L2x, 0.0]),
        "L3": np.array([L3x, 0.0]),
        "L4": np.array([0.5-mu,  np.sqrt(3)/2]),
        "L5": np.array([0.5-mu, -np.sqrt(3)/2]),
    }

    hill_r         = (mu/3)**(1/3)
    excl_primary   = min(0.03, 0.3*hill_r)
    excl_secondary = min(0.012, 0.3*hill_r)
    offset         = max(0.5*hill_r, 0.02)

    # ── Exactly 10 x-values ──────────────────────────────────────────────────
    # 6 anchors near collinear L-points + 4 uniform fill
    anchors_x = np.array([
        L3x - 0.04, L3x + 0.04,
        L1x - offset, L1x + offset,
        L2x - offset, L2x + offset,
    ])
    fill_x = np.linspace(-1.4, 1.4, 4)          # 4 fill points
    xs = np.sort(np.unique(np.round(
        np.concatenate([fill_x, anchors_x]), 8)))  # ≤ 10 unique

    # ── Exactly 10 y-values ──────────────────────────────────────────────────
    # 3 below, 3 fine layer (−0.05, 0, +0.05), 4 above
    ys = np.sort(np.unique(np.concatenate([
        np.linspace(-1.2, -0.15, 3),
        [-0.05, 0.0, 0.05],
        np.linspace(0.15, 1.2, 4),
    ])))                                           # exactly 10

    print(f"Grid: {len(xs)} x-values × {len(ys)} y-values = {len(xs)*len(ys)} starts")

    results = []
    for x in xs:
        for y in ys:
            if (x+mu)**2 + y**2 < excl_primary**2:
                continue
            if (x-1+mu)**2 + y**2 < excl_secondary**2:
                continue
            traj, lm_h, gn_h, conv, final, gn = simulate_dmm(
                mu, [x, y], beta, mem_cap, gamma, dt, max_steps, conv_thr)
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
                "start": np.array([x, y]), "traj": traj,
                "lm_hist": lm_h, "gn_hist": gn_h,
                "conv": conv, "final": final, "label": label,
            })
    return results, analytic


def pick_representative(results):
    """One trajectory per L-point — most typical instanton path."""
    reps = {}
    for name in L_COLORS:
        group = [r for r in results if r["label"] == name and len(r["traj"]) > 5]
        if not group:
            continue
        if name in ("L1", "L2", "L3"):
            # collinear: pick start closest to y=0
            best = min(group, key=lambda r: abs(r["start"][1]))
        else:
            # triangular: pick trajectory with median number of steps
            lengths = [len(r["traj"]) for r in group]
            med = np.median(lengths)
            best = min(group, key=lambda r: abs(len(r["traj"]) - med))
        reps[name] = best
    return reps


# ══════════════════════════════════════════════════════════════════════════════
# Run simulation
# ══════════════════════════════════════════════════════════════════════════════
print("Running 10×10 DMM discovery (Earth–Moon, μ=0.01215) …")
results, analytic = run_discovery_100(
    MU, BETA, MEM_CAP, GAMMA, DT, MAX_STEP, CONV_THR)

counts = {k: sum(1 for r in results if r["label"]==k) for k in L_COLORS}
diverg = sum(1 for r in results if not r["conv"])
print(f"Found: {counts}  diverged: {diverg}")

reps = pick_representative(results)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Discovery map  (one curve per L-point)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1 …")

fig1, ax = plt.subplots(figsize=(6, 5.5))

xg = np.linspace(-1.6, 1.6, 400)
yg = np.linspace(-1.45, 1.45, 400)
XG, YG = np.meshgrid(xg, yg)
Omega_grid = np.clip(effective_potential(XG, YG, MU), -4.0, -1.3)
ax.contourf(XG, YG, Omega_grid, levels=60, cmap="YlOrRd_r", alpha=0.30)
ax.contour( XG, YG, Omega_grid, levels=18, colors="gray",
            linewidths=0.3, alpha=0.4)

# One representative trajectory per L-point
for name, r in reps.items():
    t = r["traj"]
    ax.plot(t[:, 0], t[:, 1], color=L_COLORS[name], lw=2.0,
            label=f"${name}$", zorder=3)
    ax.annotate(name, xy=t[-1], xytext=(t[-1, 0]+0.10, t[-1, 1]+0.08),
                color=L_COLORS[name], fontsize=10, fontweight="bold")

# Primaries
ax.plot([-MU], [0], "o", color="#3498db", ms=12, zorder=6, label="Earth")
ax.plot([1-MU], [0], "o", color="#bdc3c7", ms=7,  zorder=6, label="Moon")

# L-point markers
for name, pos in analytic.items():
    ax.plot(*pos, "*", color=L_COLORS[name], ms=18, zorder=7,
            markeredgecolor="k", markeredgewidth=0.5)

ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.48, 1.48)
ax.set_xlabel(r"$x$ (co-rotating frame)")
ax.set_ylabel(r"$y$ (co-rotating frame)")
ax.set_aspect("equal")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
leg = ax.legend(fontsize=9, loc="upper right", ncol=2)
plt.tight_layout()
fig1.savefig("fig1_discovery.pdf", dpi=300, bbox_inches="tight")
fig1.savefig("fig1_discovery.png", dpi=200, bbox_inches="tight")
print("  → fig1_discovery.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Memory dynamics  (one curve per L-point)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 …")

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

for name, r in reps.items():
    col  = L_COLORS[name]
    lm   = r["lm_hist"]                          # (N,2)
    gnh  = r["gn_hist"]
    steps_lm = np.arange(len(lm)) * 20
    steps_gn = np.arange(len(gnh)) * 20

    ax1.plot(steps_lm, lm[:, 0], color=col, lw=2.0, label=f"${name}$")
    ax2.plot(steps_lm, lm[:, 1], color=col, lw=2.0, label=f"${name}$")
    ax3.semilogy(steps_gn, gnh,  color=col, lw=2.0, label=f"${name}$")

# Formatting shared
for ax in (ax1, ax2, ax3):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel(r"Integration step $n$")

ax1.set_ylabel(r"$w_L^x$")
ax2.set_ylabel(r"$w_L^y$")
ax3.set_ylabel(r"$|\nabla\Omega|$")

ax3.axhline(CONV_THR, color="0.3", lw=1.2, ls="--",
            label=f"threshold {CONV_THR:.0e}")
ax3.set_ylim(bottom=1e-6)
ax3.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))

ax1.legend(fontsize=9)
ax3.legend(fontsize=9)

plt.tight_layout()
fig2.savefig("fig2_memory.pdf", dpi=300, bbox_inches="tight")
fig2.savefig("fig2_memory.png", dpi=200, bbox_inches="tight")
print("  → fig2_memory.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Effective potential + Ω_yy curvature map
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 …")

fig3, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

xg2 = np.linspace(-1.5, 1.5, 400)
yg2 = np.linspace(-1.3, 1.3, 400)
X2, Y2 = np.meshgrid(xg2, yg2)

# Left: Ω_yy curvature
Oyy = (1 - (1-MU)/(( X2+MU)**2+Y2**2)**1.5
         + 3*(1-MU)*Y2**2/((X2+MU)**2+Y2**2)**2.5
         - MU/((X2-1+MU)**2+Y2**2)**1.5
         + 3*MU*Y2**2/((X2-1+MU)**2+Y2**2)**2.5)
Oyy = np.clip(Oyy, -3, 3)

im1 = axL.pcolormesh(X2, Y2, Oyy, cmap="RdBu", vmin=-3, vmax=3,
                     shading="auto", rasterized=True)
axL.contour(X2, Y2, Oyy, levels=[0.0], colors="#f1c40f",
            linestyles="--", linewidths=1.8)
cb1 = fig3.colorbar(im1, ax=axL, fraction=0.046, pad=0.04)
cb1.set_label(r"$\Omega_{yy}$")
cb1.ax.yaxis.set_tick_params(width=1.2)

for name, pos in analytic.items():
    axL.plot(*pos, "*", color=L_COLORS[name], ms=14, zorder=5,
             markeredgecolor="k", markeredgewidth=0.4,
             label=f"${name}$" if name in ("L1","L4") else "")
axL.set_xlim(-1.5, 1.5); axL.set_ylim(-1.3, 1.3)
axL.set_xlabel(r"$x$"); axL.set_ylabel(r"$y$")
axL.set_aspect("equal")
axL.xaxis.set_minor_locator(ticker.AutoMinorLocator())
axL.yaxis.set_minor_locator(ticker.AutoMinorLocator())
axL.legend(fontsize=9, loc="upper right", ncol=2)

# Right: effective potential
Om = np.clip(effective_potential(X2, Y2, MU), -2.0, 1.5)
im2 = axR.pcolormesh(X2, Y2, Om, cmap="plasma", shading="auto", rasterized=True)
axR.contour(X2, Y2, Om, levels=20, colors="white", linewidths=0.3, alpha=0.3)
cb2 = fig3.colorbar(im2, ax=axR, fraction=0.046, pad=0.04)
cb2.set_label(r"$\Omega(x,y)$")
cb2.ax.yaxis.set_tick_params(width=1.2)

for name, pos in analytic.items():
    axR.plot(*pos, "*", color=L_COLORS[name], ms=14, zorder=5,
             markeredgecolor="k", markeredgewidth=0.4)
    axR.annotate(name, xy=pos,
                 xytext=(pos[0]+0.08, pos[1]+0.08),
                 color="white", fontsize=9, fontweight="bold")
axR.set_xlim(-1.5, 1.5); axR.set_ylim(-1.3, 1.3)
axR.set_xlabel(r"$x$"); axR.set_ylabel(r"$y$")
axR.set_aspect("equal")
axR.xaxis.set_minor_locator(ticker.AutoMinorLocator())
axR.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
fig3.savefig("fig3_potential_curvature.pdf", dpi=300, bbox_inches="tight")
fig3.savefig("fig3_potential_curvature.png", dpi=200, bbox_inches="tight")
print("  → fig3_potential_curvature.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# β = 0 test: does the correction current alone suffice?
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning β=0 test (no memory growth) …")
results_b0, _ = run_discovery_100(
    MU, beta=0.0, mem_cap=MEM_CAP, gamma=GAMMA,
    dt=DT, max_steps=MAX_STEP, conv_thr=CONV_THR)

counts_b0 = {k: sum(1 for r in results_b0 if r["label"]==k) for k in L_COLORS}
diverg_b0  = sum(1 for r in results_b0 if not r["conv"])
print(f"β=0 found: {counts_b0}  diverged: {diverg_b0}")
print(f"β=0 total converged: {sum(counts_b0.values())} / {len(results_b0)}")
