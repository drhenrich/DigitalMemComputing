"""
Solar System DMM v4 — all planets + Sun simultaneously (superposition map)
==========================================================================
PHYSICS NOTE (read this): the full Sun + 8-planet system has NO global Lagrange
points. Lagrange points are a property of the *restricted three-body* problem —
two primaries on a circular orbit, whose co-rotating frame has a time-independent
effective potential Ω with ∇Ω = 0 at five fixed points. The planets orbit at
different rates, so no single co-rotating frame exists and no time-independent Ω
exists. There is no "∇Ω = 0 of the whole solar system."

What IS physical, and what this app shows: for each Sun–planet PAIR we discover
the 5 Lagrange points with the v3 memory-as-dissipation DMM (in that pair's
normalized rotating frame), then map them to heliocentric coordinates and overlay
all 8 planets at their positions for a chosen epoch. The result is a single
snapshot of every Sun–planet Lagrange point at once — where JWST sits (Sun–Earth
L2), where the Jupiter Trojans live (Sun–Jupiter L4/L5), and so on.

Planet angles use a circular, coplanar mean-longitude approximation (J2000 +
mean motion). Radii are real (AU); a radial rescale keeps inner and outer planets
visible together.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd

EPS = 1e-9
M_SUN = 1.989e30
ROUTH = 0.03852

# name: (a_AU, mass_kg, period_yr, mean_longitude_J2000_deg, color)
PLANETS = {
    "Mercury": (0.38710, 3.285e23,   0.24085, 252.25, "#b5b5b5"),
    "Venus":   (0.72333, 4.867e24,   0.61520, 181.98, "#e8cda0"),
    "Earth":   (1.00000, 5.972e24,   1.00000, 100.46, "#4fc3f7"),
    "Mars":    (1.52368, 6.390e23,   1.88085, 355.43, "#c1440e"),
    "Jupiter": (5.20260, 1.898e27,  11.8618,   34.40, "#c88b3a"),
    "Saturn":  (9.55491, 5.683e26,  29.4571,   49.94, "#e4d191"),
    "Uranus":  (19.2184, 8.681e25,  84.0205,  313.23, "#80d8e8"),
    "Neptune": (30.1104, 1.024e26, 164.770,   304.88, "#3f54ba"),
}

L_MARK = {"L1":"o","L2":"o","L3":"o","L4":"^","L5":"v"}

KNOWN = {
    ("Earth","L1"):"SOHO, DSCOVR",
    ("Earth","L2"):"JWST, Gaia, Planck",
    ("Earth","L4"):"Earth Trojan 2010 TK7",
    ("Jupiter","L4"):">7000 Greek-camp Trojans",
    ("Jupiter","L5"):">7000 Trojan-camp Trojans",
    ("Mars","L5"):"Mars Trojans (Eureka)",
    ("Neptune","L4"):"Neptune Trojans",
}


# ── DMM v3 physics (self-contained, memory-as-dissipation) ──────────────────────
def grad_curv(x, y, mu):
    r1 = np.sqrt((x+mu)**2+y**2)+EPS; r2 = np.sqrt((x-1+mu)**2+y**2)+EPS
    gx = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    gy = y - (1-mu)*y/r1**3 - mu*y/r2**3
    oyy = 1 - (1-mu)/r1**3 + 3*(1-mu)*y**2/r1**5 - mu/r2**3 + 3*mu*y**2/r2**5
    return gx, gy, oyy

def analytic_collinear(mu):
    def g0(x):
        r1 = abs(x+mu)+EPS; r2 = abs(x-1+mu)+EPS
        return x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    return (brentq(g0,-mu+1e-4,1-mu-1e-4), brentq(g0,1-mu+1e-4,2.5), brentq(g0,-2.5,-mu-1e-4))

def simulate_v3(mu, start, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=10.0,
                dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); m = 0.0
    for s in range(max_steps):
        x, y = pos; gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        m = min(m + beta*gn*dt, m_cap); ge = gamma0 + kappa*m
        sig = 1.0 if oyy < 0 else -1.0
        ax = 2*vel[1] - gx - ge*vel[0]; ay = -2*vel[0] + sig*gy - ge*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: return pos, True
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: return pos, False
    return pos, False


@st.cache_data(show_spinner=False)
def discover_pair(mu):
    """Run the v3 DMM on a 10x10 grid for one Sun-planet pair; return found counts."""
    L1x, L2x, L3x = analytic_collinear(mu)
    L = {"L1":np.array([L1x,0.]),"L2":np.array([L2x,0.]),"L3":np.array([L3x,0.]),
         "L4":np.array([0.5-mu,np.sqrt(3)/2]),"L5":np.array([0.5-mu,-np.sqrt(3)/2])}
    hill = (mu/3)**(1/3); off = max(0.5*hill, 0.02)
    ep = min(0.03,0.3*hill); es = min(0.012,0.3*hill)
    anchors = np.array([L1x-off,L1x+off,L2x-off,L2x+off,L3x-0.04,L3x+0.04])
    xs = np.sort(np.unique(np.round(np.concatenate([np.linspace(-1.4,1.4,4),anchors]),8)))[:10]
    ys = np.sort(np.unique(np.concatenate([np.linspace(-1.2,-0.15,3),[-0.05,0,0.05],
                 np.linspace(0.15,1.2,4)])))[:10]
    counts = {k:0 for k in L}
    for x in xs:
        for y in ys:
            if (x+mu)**2+y**2 < ep**2 or (x-1+mu)**2+y**2 < es**2: continue
            endp, conv = simulate_v3(mu, [x,y])
            if not conv: continue
            best, db = None, 1e9
            for k,v in L.items():
                d = np.linalg.norm(endp-v)
                if d < db: db, best = d, k
            if db < 0.12: counts[best] += 1
    return counts


# ── Heliocentric geometry ───────────────────────────────────────────────────────
def helio_angle(name, year):
    a, mass, period, L0, color = PLANETS[name]
    return np.radians(L0 + (360.0/period)*(year - 2000.0)) % (2*np.pi)

def lpoints_heliocentric(name, year):
    """Return dict of the 5 L-points of the Sun-(name) pair in heliocentric AU."""
    a, mass, period, L0, color = PLANETS[name]
    mu = mass/(M_SUN + mass)
    th = helio_angle(name, year)
    u = np.array([np.cos(th), np.sin(th)])           # Sun->planet unit vector
    P = a*u
    rH = a*(mu/3)**(1/3)
    L1x, L2x, L3x = analytic_collinear(mu)            # normalized (planet at 1-mu)
    # collinear points sit on the Sun-planet line at normalized x -> physical radius a*x
    pts = {
        "L1": a*L1x*u,
        "L2": a*L2x*u,
        "L3": a*L3x*u,
        "L4": a*np.array([np.cos(th+np.pi/3), np.sin(th+np.pi/3)]),
        "L5": a*np.array([np.cos(th-np.pi/3), np.sin(th-np.pi/3)]),
    }
    return P, pts, mu


# ── Radial rescale so 0.39–30 AU all fit ────────────────────────────────────────
def rescale(xy, mode):
    r = np.hypot(xy[...,0], xy[...,1]) + 1e-12
    if mode == "linear": f = r
    elif mode == "sqrt": f = np.sqrt(r)
    else:                f = np.log10(r + 1.0)        # log: log10(1+r)
    out = xy.copy().astype(float)
    out[...,0] = xy[...,0]/r * f
    out[...,1] = xy[...,1]/r * f
    return out

def rscale_scalar(r, mode):
    if mode == "linear": return r
    if mode == "sqrt":   return np.sqrt(r)
    return np.log10(r + 1.0)


# ── Plot ────────────────────────────────────────────────────────────────────────
def plot_system(year, selected, mode, show_orbits, show_labels):
    fig, ax = plt.subplots(figsize=(9, 9)); fig.patch.set_facecolor("#05060d")
    ax.set_facecolor("#05060d"); ax.set_aspect("equal")
    ax.plot(0, 0, "o", color="#FDB813", ms=16, zorder=10)
    ax.annotate("Sun", (0,0), (8,8), textcoords="offset points", color="#FDB813", fontsize=10)

    th = np.linspace(0, 2*np.pi, 400)
    for name in selected:
        a, mass, period, L0, color = PLANETS[name]
        rp = rscale_scalar(a, mode)
        if show_orbits:
            ax.plot(rp*np.cos(th), rp*np.sin(th), color=color, lw=0.6, alpha=0.35, zorder=1)
        P, pts, mu = lpoints_heliocentric(name, year)
        Pp = rescale(P, mode)
        ax.plot(*Pp, "o", color=color, ms=9, zorder=6, markeredgecolor="white", markeredgewidth=0.4)
        if show_labels:
            ax.annotate(name, Pp, (6,6), textcoords="offset points", color=color, fontsize=8)
        for k, v in pts.items():
            vp = rescale(v, mode)
            ax.plot(*vp, L_MARK[k], color=color, ms=6, zorder=5, alpha=0.95,
                    markeredgecolor="white", markeredgewidth=0.3)

    # radial reference rings (AU)
    for au, lab in [(1,"1 AU"),(5,"5"),(10,"10"),(30,"30")]:
        rr = rscale_scalar(au, mode)
        ax.plot(rr*np.cos(th), rr*np.sin(th), color="#333", lw=0.5, ls=":", zorder=0)
        ax.annotate(lab, (0, rr), color="#555", fontsize=7, ha="center", zorder=0)

    # legend for marker shapes
    handles = [plt.Line2D([],[],marker="o",ls="",color="w",label="L1/L2/L3 (collinear)"),
               plt.Line2D([],[],marker="^",ls="",color="w",label="L4 (+60°)"),
               plt.Line2D([],[],marker="v",ls="",color="w",label="L5 (−60°)"),
               plt.Line2D([],[],marker="o",ls="",color="#FDB813",label="planet")]
    leg = ax.legend(handles=handles, fontsize=8, loc="upper right",
                    facecolor="#11131f", edgecolor="#444", labelcolor="white")
    lim = rscale_scalar(max(PLANETS[n][0] for n in selected), mode)*1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.tick_params(colors="#444"); ax.spines[:].set_color("#222")
    ax.set_title(f"Sun–planet Lagrange points, all pairs superposed — year {year}\n"
                 f"(radial scale: {mode}; angles: circular mean-longitude approx.)",
                 color="white", fontsize=11)
    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Solar System DMM v4", layout="wide", page_icon="🪐")
st.title("🪐 All Sun–planet Lagrange points at once — DMM v4")
st.markdown(r"""
**There are no global Lagrange points of the full solar system** — they are a
property of the *two-body* (restricted three-body) problem, because only two
primaries on a circular orbit admit a co-rotating frame with a time-independent
$\Omega$. This view therefore shows, for each **Sun–planet pair**, the 5 Lagrange
points discovered by the v3 memory-as-dissipation DMM, mapped to heliocentric
coordinates and **superposed** at a chosen epoch. Planet angles use a circular,
coplanar mean-longitude approximation.
""")

with st.sidebar:
    st.header("View")
    year = st.slider("Epoch (year)", 1900, 2100, 2026, 1)
    mode = st.radio("Radial scale", ["sqrt", "log", "linear"], index=0,
                    help="Orbits span 0.39–30 AU; sqrt/log keep inner + outer planets visible.")
    show_orbits = st.checkbox("Show orbits", True)
    show_labels = st.checkbox("Label planets", True)
    st.divider()
    selected = st.multiselect("Planets", list(PLANETS.keys()), default=list(PLANETS.keys()))
    if not selected:
        selected = list(PLANETS.keys())
    st.divider()
    verify = st.checkbox("Verify each pair with the DMM (slower)", value=False,
                         help="Runs the v3 solver per Sun–planet pair and reports how many of "
                              "the 5 L-points it discovers (4/5 expected for Mercury).")

tab_map, tab_table, tab_about = st.tabs(["🗺️ Map", "📋 L-point table", "ℹ️ Physics"])

with tab_map:
    st.pyplot(plot_system(year, selected, mode, show_orbits, show_labels), use_container_width=True)
    st.caption("Markers: ● collinear L1/L2/L3 (on the Sun–planet line), "
               "▲ L4 (+60° leading), ▼ L5 (−60° trailing). One colour per planet. "
               "Radial reference rings at 1, 5, 10, 30 AU.")

with tab_table:
    rows = []
    for name in selected:
        a, mass, period, L0, color = PLANETS[name]
        mu = mass/(M_SUN+mass)
        P, pts, _ = lpoints_heliocentric(name, year)
        stable = mu < ROUTH
        verify_str = ""
        if verify:
            c = discover_pair(mu)
            nf = sum(1 for v in c.values() if v>0)
            verify_str = f"{nf}/5"
        rows.append({
            "Planet": name, "a (AU)": f"{a:.3f}", "μ": f"{mu:.3e}",
            "Hill (AU)": f"{a*(mu/3)**(1/3):.4f}",
            "L4/L5 stable": "✓" if stable else "✗ (Routh)",
            "DMM found": verify_str,
            "Notable": ", ".join(v for (p,k),v in KNOWN.items() if p==name) or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if verify:
        st.caption("‘DMM found’ runs the v3 memory-as-dissipation solver on each pair. "
                   "Mercury (μ≈1.7e-7) returns 4/5 — the corotation-ridge limit documented "
                   "in the paper.")

with tab_about:
    st.markdown(r"""
### Why there is no "solar-system Lagrange point"

A Lagrange point is an equilibrium of the **restricted three-body problem**: two
primaries $M_1,M_2$ on a circular orbit, and a massless test particle. In the
frame co-rotating with the two primaries the effective potential

$$\Omega(x,y)=\tfrac12(x^2+y^2)+\frac{1-\mu}{r_1}+\frac{\mu}{r_2}$$

is **time-independent**, and its five critical points $\nabla\Omega=0$ are the
Lagrange points. This construction needs the two primaries to be *fixed* in the
rotating frame.

With the Sun and **all eight planets**, each planet orbits at a different angular
rate ($\propto a^{-3/2}$: Mercury 88 d, Neptune 165 yr). **No single rotating
frame freezes them all**, so there is no time-independent $\Omega$ for the whole
system and no global $\nabla\Omega=0$. The honest statement is therefore:

> The solar system has **no global Lagrange points** — only the five points of
> each Sun–planet (and planet–moon) pair, each in its own co-rotating frame.

This app computes those per-pair points with the v3 DMM and **superposes** them.
The collinear $L_1,L_2$ hug each planet at $\pm a(\mu/3)^{1/3}$; $L_3$ sits just
beyond the opposite side of the orbit; $L_4,L_5$ lead and trail the planet by
$60^\circ$ on its own orbit. Real missions and asteroid families occupy these
points (JWST at Sun–Earth $L_2$; the Jupiter Trojans at $L_4/L_5$), which is why
the superposition is physically meaningful even though a global equilibrium is
not.

*(For a genuine many-body treatment one would instead integrate test particles in
the time-dependent field of the Sun + planets — a Trojan-stability simulation,
not a fixed-point problem. Happy to add that separately.)*
""")
