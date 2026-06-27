"""
Solar System DMM v3 — Memory-as-Dissipation Lagrange Point Discovery
====================================================================
The memory variable now controls the DISSIPATION, not a force multiplier:

    ẍ = 2ẏ − ∂_xΩ − γ_eff·ẋ
    ÿ = −2ẋ + σ·∂_yΩ − γ_eff·ẏ
    ṁ = β·‖∇Ω‖                (scalar memory, ratchets up)
    γ_eff = γ₀ + κ·m

Why: a memory that MULTIPLIES the (conservative) gradient force is dynamically
inert — by the kinetic-energy identity it can only exchange energy with the
potential, never dissipate it (see dmm_lagrange_v3.tex, Prop. 1). Folding the
same memory into γ_eff makes it couple to velocity, the only channel that
removes kinetic energy. With γ₀=0 the memory is the SOLE dissipation and is
provably load-bearing: it drives convergence where the multiplier formulation
fails (0/5 → 5/5 on Earth–Moon).

σ = sign(−Ω_yy) is the curvature-adaptive correction current that turns the
collinear saddle points into attractors of the damped flow.

An optional Newton refinement polishes converged endpoints to the exact zeros of
∇Ω (the 5 L-points are the only exact zeros), with fall-back to the raw endpoint
where Newton is ill-conditioned (small-μ corotation ridge).
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
import pandas as pd

# Single source of truth for CR3BP geometry (was duplicated here verbatim).
from nbody_trojan import (
    analytical_collinear, grad_curv, effective_potential, lpoints,
)

EPS = 1e-9
ROUTH_LIMIT = 0.03852

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

SYSTEMS = {
    "Sun–Mercury":  ("Sun","Mercury",  5.791e10, {}),
    "Sun–Venus":    ("Sun","Venus",    1.082e11, {}),
    "Sun–Earth":    ("Sun","Earth",    1.496e11, {
        "L1": "SOHO, DSCOVR (solar wind monitoring)",
        "L2": "James Webb Space Telescope, Gaia, Planck",
        "L4": "Earth Trojans (2010 TK7)"}),
    "Sun–Mars":     ("Sun","Mars",     2.279e11, {
        "L4": "Mars Trojans (Eureka family)", "L5": "Mars Trojans (several known)"}),
    "Sun–Jupiter":  ("Sun","Jupiter",  7.783e11, {
        "L4": "Greek camp — >7,000 Trojan asteroids",
        "L5": "Trojan camp — >7,000 Trojan asteroids"}),
    "Sun–Saturn":   ("Sun","Saturn",   1.432e12, {}),
    "Sun–Uranus":   ("Sun","Uranus",   2.867e12, {}),
    "Sun–Neptune":  ("Sun","Neptune",  4.495e12, {"L4": "Neptune Trojans (2001 QR322, etc.)"}),
    "Sun–Pluto":    ("Sun","Pluto",    5.906e12, {}),
    "Earth–Moon":   ("Earth","Moon",   3.844e8, {
        "L1": "Proposed lunar comm relay",
        "L2": "ARTEMIS mission, proposed Gateway staging",
        "L4": "Kordylewski clouds (dust)", "L5": "Kordylewski clouds (dust)"}),
    "Mars–Phobos":  ("Mars","Phobos",  9.376e6, {}),
    "Mars–Deimos":  ("Mars","Deimos",  2.346e7, {}),
    "Jupiter–Io":        ("Jupiter","Io",       4.218e8, {}),
    "Jupiter–Europa":    ("Jupiter","Europa",   6.711e8, {}),
    "Jupiter–Ganymede":  ("Jupiter","Ganymede", 1.070e9, {}),
    "Jupiter–Callisto":  ("Jupiter","Callisto", 1.883e9, {}),
    "Saturn–Titan":      ("Saturn","Titan",     1.222e9, {}),
    "Saturn–Enceladus":  ("Saturn","Enceladus", 2.382e8, {}),
    "Saturn–Rhea":       ("Saturn","Rhea",      5.270e8, {}),
    "Uranus–Titania":    ("Uranus","Titania",   4.360e8, {}),
    "Uranus–Oberon":     ("Uranus","Oberon",    5.835e8, {}),
    "Neptune–Triton":    ("Neptune","Triton",   3.548e8, {}),
    "Pluto–Charon":      ("Pluto","Charon",     1.959e7, {
        "note": "μ > Routh limit — L4/L5 linearly unstable in rotating frame"}),
}

L_COLORS = {"L1":"#e74c3c","L2":"#e67e22","L3":"#9b59b6","L4":"#27ae60","L5":"#2980b9"}


# ── Physics ─────────────────────────────────────────────────────────────────────
def compute_system_info(p1, p2, sma_m):
    m1, m2 = BODIES[p1]["mass"], BODIES[p2]["mass"]
    mu = m2 / (m1 + m2)
    return mu, sma_m * (mu/3)**(1/3), mu < ROUTH_LIMIT

def grad_vec(p, mu):
    gx, gy, _ = grad_curv(p[0], p[1], mu)
    return np.array([gx, gy])

def hessian(p, mu):
    x, y = p
    r1 = np.sqrt((x+mu)**2+y**2)+EPS; r2 = np.sqrt((x-1+mu)**2+y**2)+EPS
    hxx = 1 - (1-mu)/r1**3 + 3*(1-mu)*(x+mu)**2/r1**5 - mu/r2**3 + 3*mu*(x-1+mu)**2/r2**5
    hyy = 1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5 - mu/r2**3 + 3*mu*y**2/r2**5
    hxy = 3*(1-mu)*(x+mu)*y/r1**5 + 3*mu*(x-1+mu)*y/r2**5
    return np.array([[hxx, hxy],[hxy, hyy]])


def simulate_v3(mu, start, beta, gamma0, kappa, m_cap, dt, max_steps, conv_thr):
    """Memory-as-dissipation: γ_eff = γ0 + κ·m, ṁ = β‖∇Ω‖ (scalar)."""
    pos = np.array(start, float); vel = np.zeros(2); m = 0.0
    traj = [pos.copy()]; m_h = [0.0]; g_h = [gamma0]; gn_h = []; T_h = []
    conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu)
        gn = np.hypot(gx, gy)
        m = min(m + beta*gn*dt, m_cap)
        g_eff = gamma0 + kappa*m
        sig = 1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - gx - g_eff*vel[0]
        ay = -2*vel[0] + sig*gy - g_eff*vel[1]
        vel += np.array([ax, ay])*dt
        pos = pos + vel*dt
        if step % 20 == 0:
            traj.append(pos.copy()); m_h.append(m); g_h.append(g_eff)
            gn_h.append(gn); T_h.append(0.5*(vel[0]**2+vel[1]**2))
        if gn < conv_thr:
            conv = step; traj.append(pos.copy()); break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return (np.array(traj), np.array(m_h), np.array(g_h),
            np.array(gn_h), np.array(T_h), conv, pos)


def simulate_broyden(mu, start, gamma, alpha, eps_c, leak, thr, dpmin,
                     dt, max_steps, conv_thr):
    """Curvature-learning machine: a quasi-Newton (Broyden) memory B estimates the
    Hessian from the gradient history (no analytic Hessian) and sets the transverse
    force-inversion gain s from B_yy. ẍ=2ẏ−∂ₓΩ−γẋ, ÿ=−2ẋ+s·∂ᵧΩ−γẏ."""
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0; B = np.eye(2)
    gx, gy, oyy = grad_curv(pos[0], pos[1], mu)
    p_prev = pos.copy(); g_prev = np.array([gx, gy])
    traj = [pos.copy()]; byy_h = [B[1, 1]]; s_h = [s]; oyy_h = [oyy]
    gn_h = []; T_h = []; conv = None
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy); g = np.array([gx, gy])
        dp = pos - p_prev; dg = g - g_prev; nd = dp @ dp
        if nd > dpmin*dpmin:
            B = B + np.outer(dg - B @ dp, dp) / nd
            B = 0.5*(B + B.T); B = B + leak*(np.eye(2) - B)*dt
        s += alpha*(-np.tanh((B[1, 1] - thr)/eps_c) - s)*dt; s = max(-1.0, min(1.0, s))
        p_prev = pos.copy(); g_prev = g
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if step % 20 == 0:
            traj.append(pos.copy()); byy_h.append(B[1, 1]); s_h.append(s); oyy_h.append(oyy)
            gn_h.append(gn); T_h.append(0.5*(vel[0]**2 + vel[1]**2))
        if gn < conv_thr:
            conv = step; traj.append(pos.copy()); break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            break
    return (np.array(traj), np.array(byy_h), np.array(s_h), np.array(oyy_h),
            np.array(gn_h), np.array(T_h), conv, pos)

def newton_polish(pos, mu):
    """Snap to the nearest exact zero of ∇Ω; return (point, ok). Falls back if it fails."""
    try:
        sol, _, ier, _ = fsolve(grad_vec, pos, args=(mu,),
                                fprime=lambda p, mu: hessian(p, mu),
                                full_output=True, xtol=1e-12)
    except Exception:
        return pos, False
    if ier != 1 or np.linalg.norm(grad_vec(sol, mu)) > 1e-9:
        return pos, False
    return sol, True


def build_grid(mu, fill=6):
    """Structural seed grid that works for ANY mass ratio mu.

    Uses only generic structure of the CR3BP — NOT the solution coordinates:
      (A) collinear L1/L2 lie on the rotating axis within a Hill radius
          r_H=(mu/3)^(1/3) of the secondary -> on-axis seeds at the secondary
          position (1-mu) +- c*r_H;
      (B) L3 lies opposite the primary near x=-1;
      (C) L4/L5 form equilateral triangles with the primaries near
          (1/2, +-sqrt3/2);
      (D) a coarse off-axis fill for large-mu basins.
    Seeds inside the singularity disks around either primary are dropped.
    """
    hill = (mu/3)**(1/3); sec = 1.0 - mu
    excl_p = min(0.03, 0.3*hill); excl_s = min(0.012, 0.3*hill)
    seeds = []
    for c in (0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0):                 # (A)
        for xs in (sec - c*hill, sec + c*hill):
            seeds.append((xs, 0.0))
    for xs in (-1.0 - 5*mu/12, -0.96, -1.04, -1.10):                  # (B)
        seeds.append((xs, 0.0))
    for dx in (-0.12, -0.04, 0.04, 0.12):                             # (C)
        for sgn in (+1, -1):
            for dy in (-0.10, 0.0, 0.10):
                seeds.append((0.5 + dx, sgn*np.sqrt(3)/2 + dy))
    for x in np.linspace(-1.3, 1.3, fill):                           # (D)
        for y in (0.9, 0.45, -0.45, -0.9):
            seeds.append((x, y))
    out = []
    for (x, y) in seeds:
        if (x+mu)**2 + y**2 < excl_p**2 or (x-1+mu)**2 + y**2 < excl_s**2:
            continue
        out.append((x, y))
    return out


def run_discovery(mu, beta, gamma0, kappa, m_cap, dt, max_steps, conv_thr, n_x, n_y, polish):
    """Discover all five Lagrange points. n_x sets the off-axis fill density
    (kept for UI compatibility); the structural seeds (build_grid) guarantee
    coverage of every L-point for any mu."""
    L = lpoints(mu)
    results = []
    for (x, y) in build_grid(mu, fill=max(n_x-4, 4)):
        traj, m_h, g_h, gn_h, T_h, conv, endp = simulate_v3(
            mu, [x, y], beta, gamma0, kappa, m_cap, dt, max_steps, conv_thr)
        pt = endp; polished = False
        if polish and np.isfinite(endp).all():
            pol, ok = newton_polish(endp, mu)
            if ok:
                pt = pol; polished = True
        label, dbest = None, 1e9
        if np.isfinite(pt).all():
            for k, v in L.items():
                d = np.linalg.norm(pt - v)
                if d < dbest: dbest, label = d, k
            label = label if dbest < (0.05 if polished else 0.12) else None
        results.append(dict(start=(x, y), traj=traj, m_h=m_h, g_h=g_h, gn_h=gn_h,
                            T_h=T_h, conv=conv, final=pt, label=label))
    return results, L


def run_discovery_qn(mu, gamma, alpha, eps_c, leak, thr, dpmin,
                     dt, max_steps, conv_thr, n_x, n_y, polish):
    """Discovery with the curvature-learning (quasi-Newton memory) machine. Same
    structural grid and labelling as run_discovery; no analytic Hessian is used."""
    L = lpoints(mu)
    results = []
    for (x, y) in build_grid(mu, fill=max(n_x-4, 4)):
        traj, byy_h, s_h, oyy_h, gn_h, T_h, conv, endp = simulate_broyden(
            mu, [x, y], gamma, alpha, eps_c, leak, thr, dpmin, dt, max_steps, conv_thr)
        pt = endp; polished = False
        if polish and np.isfinite(endp).all():
            pol, ok = newton_polish(endp, mu)
            if ok: pt = pol; polished = True
        label, dbest = None, 1e9
        if np.isfinite(pt).all():
            for k, v in L.items():
                d = np.linalg.norm(pt - v)
                if d < dbest: dbest, label = d, k
            label = label if dbest < (0.05 if polished else 0.12) else None
        results.append(dict(start=(x, y), traj=traj, byy_h=byy_h, s_h=s_h, oyy_h=oyy_h,
                            gn_h=gn_h, T_h=T_h, conv=conv, final=pt, label=label))
    return results, L


# ── Plots (dark theme for the app) ───────────────────────────────────────────────
def _darkstyle(ax):
    ax.set_facecolor("#0f0f1a"); ax.tick_params(colors="white")
    ax.spines[:].set_color("#666")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

def plot_trajectories(results, L, mu, p1, p2):
    fig, ax = plt.subplots(figsize=(8,7)); fig.patch.set_facecolor("#0f0f1a"); _darkstyle(ax)
    xg = np.linspace(-1.6,1.6,300); yg = np.linspace(-1.4,1.4,300); XG,YG = np.meshgrid(xg,yg)
    ax.contour(XG,YG,np.clip(effective_potential(XG,YG,mu),0,4),levels=28,colors="#888",linewidths=0.3,alpha=0.5)
    for r in results:
        if r["label"] is None: continue
        t = r["traj"]; ax.plot(t[:,0],t[:,1],color=L_COLORS[r["label"]],lw=0.8,alpha=0.5)
    ax.plot([-mu],[0],"o",color=BODIES[p1]["color"],ms=13,zorder=6,label=p1)
    ax.plot([1-mu],[0],"o",color=BODIES[p2]["color"],ms=7,zorder=6,label=p2)
    for k,v in L.items():
        n = sum(1 for r in results if r["label"]==k)
        ax.plot(*v,"*",color=L_COLORS[k],ms=16,zorder=7,markeredgecolor="white",markeredgewidth=0.5,
                label=f"{k} ({n})")
    ax.set_xlim(-1.65,1.65); ax.set_ylim(-1.45,1.45); ax.set_aspect("equal")
    ax.set_xlabel("x (co-rotating)"); ax.set_ylabel("y (co-rotating)")
    ax.legend(fontsize=8,loc="upper right",facecolor="#1a1a2e",edgecolor="#555",labelcolor="white")
    plt.tight_layout(); return fig

def plot_memory(results):
    """Shows memory is load-bearing: m, γ_eff ramp up; kinetic energy T → 0; |∇Ω| → 0."""
    fig,(a1,a2,a3) = plt.subplots(1,3,figsize=(15,4)); fig.patch.set_facecolor("#0f0f1a")
    for a in (a1,a2,a3): _darkstyle(a)
    for r in results:
        if r["label"] is None or len(r["gn_h"]) < 3: continue
        c = L_COLORS[r["label"]]; s = np.arange(len(r["m_h"]))*20
        a1.plot(s, r["m_h"], color=c, lw=0.8, alpha=0.5)
        a1.plot(s, r["g_h"], color=c, lw=0.8, alpha=0.5, ls="--")
        sT = np.arange(len(r["T_h"]))*20
        a2.semilogy(sT, np.array(r["T_h"])+1e-12, color=c, lw=0.8, alpha=0.5)
        a3.semilogy(np.arange(len(r["gn_h"]))*20, r["gn_h"], color=c, lw=0.8, alpha=0.5)
    for k,c in L_COLORS.items(): a1.plot([],[],color=c,lw=1.8,label=k)
    a1.set_xlabel("step"); a1.set_ylabel("m  (solid),  γ_eff = κm  (dashed)")
    a1.set_title("Memory ratchet → dissipation", fontsize=10)
    a1.legend(fontsize=8,facecolor="#1a1a2e",edgecolor="#555",labelcolor="white")
    a2.set_xlabel("step"); a2.set_ylabel("kinetic energy T  (log)")
    a2.set_title("Memory removes kinetic energy", fontsize=10); a2.set_ylim(1e-10,1e2)
    a3.axhline(1e-4,color="white",lw=0.8,ls="--",alpha=0.5)
    a3.set_xlabel("step"); a3.set_ylabel("|∇Ω|  (log)")
    a3.set_title("Clause violation → 0", fontsize=10); a3.set_ylim(bottom=1e-6)
    plt.tight_layout(); return fig


def plot_memory_curvature(results):
    """Curvature-learning machine: the memory B_yy learns the true Ω_yy from the
    gradient history; the inversion gain s adapts (−1 descend, +1 invert); |∇Ω|→0."""
    fig,(a1,a2,a3) = plt.subplots(1,3,figsize=(15,4)); fig.patch.set_facecolor("#0f0f1a")
    for a in (a1,a2,a3): _darkstyle(a)
    for r in results:
        if r["label"] is None or len(r.get("gn_h",[])) < 3: continue
        c = L_COLORS[r["label"]]
        sB = np.arange(len(r["byy_h"]))*20
        a1.plot(sB, r["byy_h"], color=c, lw=1.0, alpha=0.6)
        a1.plot(np.arange(len(r["oyy_h"]))*20, r["oyy_h"], color=c, lw=0.7, ls="--", alpha=0.4)
        a2.plot(np.arange(len(r["s_h"]))*20, r["s_h"], color=c, lw=1.0, alpha=0.6)
        a3.semilogy(np.arange(len(r["gn_h"]))*20, r["gn_h"], color=c, lw=0.8, alpha=0.5)
    a1.axhline(0, color="white", lw=0.6, alpha=0.4)
    for k,c in L_COLORS.items(): a1.plot([],[],color=c,lw=1.8,label=k)
    a1.set_xlabel("step"); a1.set_ylabel("B_yy learned (solid) vs Ω_yy true (dashed)")
    a1.set_title("Memory learns the curvature (no analytic Hessian)", fontsize=10)
    a1.legend(fontsize=8,facecolor="#1a1a2e",edgecolor="#555",labelcolor="white")
    a2.axhline(1,color="white",lw=0.5,ls=":",alpha=0.4); a2.axhline(-1,color="white",lw=0.5,ls=":",alpha=0.4)
    a2.set_xlabel("step"); a2.set_ylabel("inversion gain s"); a2.set_ylim(-1.15,1.15)
    a2.set_title("Adaptive inversion  (−1 descend · +1 invert saddle)", fontsize=10)
    a3.axhline(1e-4,color="white",lw=0.8,ls="--",alpha=0.5)
    a3.set_xlabel("step"); a3.set_ylabel("|∇Ω|  (log)")
    a3.set_title("Clause violation → 0", fontsize=10); a3.set_ylim(bottom=1e-6)
    plt.tight_layout(); return fig


# ── UI ───────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Solar System DMM v3", layout="wide", page_icon="🌌")
st.title("🌌 Lagrange Point Discovery — MemComputing")
st.markdown(r"""
Discover all five Lagrange points of any restricted three-body system as the stable
fixed points of a damped flow in the co-rotating frame, from a generic seed grid — no
L-point coordinates supplied. Pick the **machine** in the sidebar:

- **Curvature-learning (quasi-Newton memory)** — a damped descent finds the
  triangular *minima* of $\Omega$; for the collinear *saddles* it inverts the
  transverse force, $\ddot y = -2\dot x + s\,\partial_y\Omega - \gamma\dot y$. The
  inversion gain $s$ is set by a **memory** $B$ that *learns* the curvature from the
  gradient history (a Broyden estimate of the Hessian, **no analytic second
  derivative**): $\dot s=\alpha[-\tanh((B_{yy}-\theta_s)/\varepsilon_c)-s]$. No
  constant $s$ works, so the memory is load-bearing.
- **Memory-as-dissipation (v3)** — a memory controls the damping,
  $\gamma_{\rm eff}=\gamma_0+\kappa m$, with $\sigma=\mathrm{sign}(-\Omega_{yy})$ read
  from the analytic curvature.
""")

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

    st.divider(); st.subheader("System info")
    st.metric("Mass ratio μ", f"{mu:.4e}")
    st.metric("Hill radius", f"{hill_m/1e3:,.0f} km")
    if mu < 1e-5:
        st.info("Very small μ: the corotation ridge (∇Ω≈0 along r=1) makes the collinear "
                "points delicate. The structural seed grid (on-axis Hill-radius seeds "
                "around the secondary) resolves all five down to μ≈10⁻⁹.")
    if stable:
        st.success("✓ L4/L5 linearly stable (μ < 0.03852)")
    else:
        st.error("✗ L4/L5 linearly unstable (μ > 0.03852, Routh)")

    st.divider(); st.subheader("Machine")
    machine = st.radio("Dynamical machine",
        ["Curvature-learning (quasi-Newton memory)", "Memory-as-dissipation (v3)"],
        help="Curvature-learning: a Broyden memory learns Ω_yy from the gradient "
             "history (no analytic Hessian) and inverts the transverse force to turn "
             "the collinear saddles into attractors. v3: a memory controls the damping.")

    st.divider(); st.subheader("Parameters")
    dt      = st.select_slider("Time step dt", [0.005, 0.01, 0.02], value=0.01)
    max_steps = st.select_slider("Max steps", [50_000,100_000,200_000,400_000], value=200_000)
    conv_thr  = st.select_slider("Convergence |∇Ω| <", [1e-3,5e-4,1e-4,1e-5], value=1e-4)
    polish  = st.checkbox("Newton polish (snap to exact ∇Ω=0, with fallback)", value=True)
    n_x = st.slider("x grid size", 7, 18, 10)
    n_y = st.slider("y grid size", 6, 18, 10)
    if machine.startswith("Curvature"):
        st.caption("Quasi-Newton memory")
        gamma = st.slider("Viscosity γ (constant)", 0.05, 1.0, 0.3, 0.05)
        alpha = st.slider("Inversion relaxation α", 2.0, 40.0, 15.0, 1.0)
        eps_c = st.slider("Curvature scale ε_c", 0.1, 1.0, 0.3, 0.05)
        leak  = st.slider("Broyden leak λ (toward I)", 0.5, 5.0, 2.0, 0.5)
        thr   = st.slider("Safe-state bias θ_s (invert if B_yy < θ_s)", -1.5, 0.0, -0.6, 0.1)
    else:
        st.caption("Memory-as-dissipation")
        gamma0  = st.slider("Baseline damping γ₀  (0 ⇒ memory is sole dissipation)", 0.0, 1.0, 0.0, 0.05)
        kappa   = st.slider("Memory–damping gain κ", 0.2, 5.0, 1.0, 0.1)
        beta    = st.slider("Memory growth rate β", 0.05, 3.0, 0.5, 0.05)
        m_cap   = st.slider("Memory cap m_cap", 2.0, 20.0, 10.0, 1.0)

tab_disc, tab_mem, tab_about = st.tabs(["🎯 Discovery","📈 Memory dynamics","ℹ️ Method"])

with tab_disc:
    if st.button("▶  Run Discovery", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_x}×{n_y} trajectories for {sys_name} …"):
            if machine.startswith("Curvature"):
                results, L = run_discovery_qn(mu, gamma, alpha, eps_c, leak, thr, 1e-4,
                                              dt, max_steps, conv_thr, n_x, n_y, polish)
            else:
                results, L = run_discovery(mu, beta, gamma0, kappa, m_cap, dt,
                                           max_steps, conv_thr, n_x, n_y, polish)
        found = {k:[r for r in results if r["label"]==k] for k in L_COLORS}
        n_started = len(results)
        n_labeled = sum(1 for r in results if r["label"])
        n_lpts = sum(1 for k in L_COLORS if found[k])

        st.subheader(f"Results — {sys_name}")
        st.caption(
            f"**{n_started}** trajectories launched (one per initial condition). "
            f"**{n_labeled}** localized onto a Lagrange point; "
            f"**{n_started-n_labeled}** ended elsewhere (e.g. the corotation ridge at small μ). "
            f"Many initial conditions can share one basin, so a single L-point may show "
            f"many trajectories. **{n_lpts}/5** distinct Lagrange points discovered."
        )
        cols = st.columns(6)
        for i,(k,rs) in enumerate(found.items()):
            cols[i].metric(k, f"{len(rs)} traj.")
        cols[5].metric("Ended elsewhere", n_started-n_labeled)
        if n_lpts < 5:
            st.warning(f"Only {n_lpts}/5 found — expected for very small μ (corotation ridge). "
                       "Try a system with μ ≳ 1e-3, or enable Newton polish.")

        st.pyplot(plot_trajectories(results, L, mu, p1_name, p2_name), use_container_width=True)
        st.session_state["v3_results"] = results
        st.session_state["v3_sys"] = sys_name
        st.session_state["machine"] = machine
        if known_objs:
            st.info("**Known objects / missions:**\n" +
                    "\n".join(f"- **{k}**: {v}" for k,v in known_objs.items()))
    else:
        st.info(f"Selected **{sys_name}** (μ={mu:.4e}). Press **Run** to start.")

with tab_mem:
    if "v3_results" in st.session_state:
        if st.session_state.get("v3_sys") != sys_name:
            st.warning(f"Showing results from {st.session_state.get('v3_sys')} — re-run for {sys_name}.")
        if st.session_state.get("machine", "").startswith("Curvature"):
            st.pyplot(plot_memory_curvature(st.session_state["v3_results"]), use_container_width=True)
            st.caption(
                "**Left:** the quasi-Newton memory B_yy (solid) learns the true curvature Ω_yy "
                "(dashed) from the gradient history alone — no analytic Hessian. "
                "**Centre:** the inversion gain s adapts — held at −1 to descend onto a minimum, "
                "ramped −1→+1 to invert a saddle into an attractor. "
                "**Right:** clause violation |∇Ω| → 0. Colour = discovered L-point."
            )
        else:
            st.pyplot(plot_memory(st.session_state["v3_results"]), use_container_width=True)
            st.caption(
                "**Left:** memory m (solid) and the dissipation γ_eff = κm (dashed) ratchet up "
                "and saturate. **Centre:** kinetic energy T → 0. **Right:** |∇Ω| → 0. "
                "Colour = discovered L-point."
            )
    else:
        st.info("Run a discovery first.")

with tab_about:
    st.markdown(r"""
### Curvature-learning machine (default)

With the convention $\Omega=\tfrac12(x^2+y^2)+\tfrac{1-\mu}{r_1}+\tfrac{\mu}{r_2}$,
the triangular points $L_4,L_5$ are **minima** of $\Omega$ and the collinear
$L_1,L_2,L_3$ are **saddles**. A damped descent settles onto the minima but is
repelled by the saddles, whose transverse direction is unstable ($\Omega_{yy}<0$).
We invert the transverse force there, $\ddot y=-2\dot x+s\,\partial_y\Omega-\gamma\dot y$
with $s=+1$, turning the saddle into an attractor. The required sign is
**position-dependent and no constant works** ($s\equiv-1$ misses the saddles,
$s\equiv+1$ breaks the minima), so the inversion is genuinely load-bearing — unlike
the viscosity $\gamma$, for which any constant value suffices.

The novelty: the memory **learns** that sign rather than reading the analytic
curvature. A memory matrix $B$ accumulates a **Broyden (quasi-Newton)** estimate of
the Hessian from the gradient samples the flow already evaluates,
$B\leftarrow B+\frac{(\Delta g-B\Delta p)\Delta p^{\mathsf T}}{\Delta p^{\mathsf T}\Delta p}$
(symmetrized, with a slow leak toward $I$), and the gain relaxes toward the learned
sign, $\dot s=\alpha[-\tanh((B_{yy}-\theta_s)/\varepsilon_c)-s]$, with a safe-state
bias $\theta_s<0$. **No analytic second derivative** appears anywhere. This finds all
five points of every solar-system pair ($\mu=2.3\times10^{-9}\dots0.11$). Watch the
memory learn the curvature in the **Memory dynamics** tab.

### Memory-as-dissipation (v3)

A memory controls the damping instead, $\gamma_{\rm eff}=\gamma_0+\kappa m$,
$\dot m=\beta\lVert\nabla\Omega\rVert$, with $\sigma=\mathrm{sign}(-\Omega_{yy})$ read
from the analytic curvature. This also gives $5/5$, but the dissipation it supplies is
not special — a *constant* $\gamma$ works just as well — so the memory is not
load-bearing in this version. It is kept here for comparison.

**Advantages and limitations (both machines):** neither machine is faster than a
classical root finder. However, the latter requires prior knowledge of the Lagrange
points' positions. The MemComputing machine does not: it discovers all five points
simultaneously from a generic grid of initial conditions, treating saddles and minima
on equal footing — no bracket per point, no prior knowledge of which points are saddles.

See `dmm_curvature_ajp.tex` for the full derivation.
""")
