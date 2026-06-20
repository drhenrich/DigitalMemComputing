"""
Solar System DMM v5 — N-body Lagrange-point STABILITY simulation
================================================================
Complement to v3/v4. Where v3 *locates* the equilibria (∇Ω = 0) and v4 maps the
per-pair points, v5 forward-integrates clouds of massless test particles in the
time-dependent gravitational field of the **Sun + planets** to see which
Lagrange regions are *dynamically stable*.

- Particles seeded around a host planet's L-point, co-rotating at its mean motion.
- Heliocentric integrator with the indirect term (Sun's reflex acceleration),
  validated: exact L4 is stationary, the 5° libration period matches theory
  (147.7 vs 147.9 yr for Jupiter), Jacobi constant conserved to ~1e-8.
- Toggle between "Sun + host only" (restricted 3-body) and "Sun + all 8 planets"
  (the full time-dependent field) to see how the other planets affect survival.

Physics: L4/L5 are linearly stable when μ < 0.03852 (Routh) — particles librate
on tadpole orbits and survive. The collinear L1/L2/L3 are saddles — particles
drift away. This is the dynamical face of the curvature sign Ω_yy used by v3.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nbody_trojan as N

st.set_page_config(page_title="Solar System DMM v5 — stability", layout="wide", page_icon="🛰️")
st.title("🛰️ Lagrange-point stability in the full planetary field — DMM v5")
st.markdown(r"""
Forward-integrate test particles around a planet's Lagrange point in the
time-dependent field of the **Sun + planets**, and watch which survive.
$L_4/L_5$ (stable for $\mu<0.03852$, Routh) **librate and stay**; the collinear
$L_1/L_2/L_3$ (saddles) **drift away**. Switch on all eight planets to see the
full-field perturbations.
""")

with st.sidebar:
    st.header("Setup")
    host = st.selectbox("Host planet", list(N.PLANETS.keys()), index=4)  # Jupiter
    mu = N.q(host)/(1+N.q(host))
    st.metric("μ (planet/total)", f"{mu:.3e}")
    st.caption("✓ L4/L5 stable (μ<0.03852)" if mu < N.ROUTH else "✗ L4/L5 unstable (Routh violated)")
    which = st.selectbox("Seed around", ["L4","L5","L3","L1","L2"], index=0)
    field = st.radio("Gravitational field",
                     ["Sun + host planet (restricted 3-body)", "Sun + all 8 planets (full field)"])
    perturbers = [host] if field.startswith("Sun + host") else list(N.PLANETS.keys())

    st.divider()
    nP    = st.slider("Test particles", 10, 100, 40, 5)
    dphi  = st.slider("Angular spread (°)", 0.0, 30.0, 5.0, 0.5)
    dr    = st.slider("Radial spread (AU)", 0.0, 0.5, 0.05, 0.01)
    P_orb = N.PLANETS[host][2]
    n_orb = st.slider("Integration time (orbits of host)", 20, 2000, 300, 20)
    t_max = n_orb * P_orb
    dt    = st.select_slider("Time step (yr)", [0.002, 0.005, 0.01, 0.02], value=0.01)
    st.caption(f"= {t_max:,.0f} yr  ({n_orb} × {P_orb:.2f} yr)")

tab_traj, tab_surv, tab_phys = st.tabs(["🌀 Trajectories", "📉 Survival", "ℹ️ Physics"])


@st.cache_data(show_spinner=False)
def run(host, which, perturbers, nP, dphi, dr, t_max, dt):
    R0, V0 = N.seed_cloud(host, which, nP, dr_spread=dr, dphi_spread=dphi, seed=7)
    times, traj, surv, alive = N.integrate_cloud(R0, V0, t_max, dt, tuple(perturbers), host, sample=40)
    co = np.stack([N.to_corotating(times, traj[:,k,:], host) for k in range(traj.shape[1])], axis=1)
    return times, co, surv, alive

with st.spinner(f"Integrating {nP} particles for {t_max:,.0f} yr …"):
    times, co, surv, alive = run(host, which, perturbers, nP, dphi, dr, t_max, dt)

a_host = N.PLANETS[host][0]
col = N.PLANETS[host][4]

with tab_traj:
    c1, c2, c3 = st.columns(3)
    c1.metric("Survived", f"{surv.sum()}/{len(surv)}")
    c2.metric("Survival fraction", f"{100*surv.mean():.0f}%")
    c3.metric("Field", "all 8 planets" if len(perturbers) > 1 else "host only")

    fig, ax = plt.subplots(figsize=(8.5,8)); fig.patch.set_facecolor("#05060d"); ax.set_facecolor("#05060d")
    ax.set_aspect("equal")
    # host orbit circle
    th = np.linspace(0,2*np.pi,300)
    ax.plot(a_host*np.cos(th), a_host*np.sin(th), color="#333", lw=0.6, ls=":")
    ax.plot(0,0,"o",color="#FDB813",ms=14,zorder=8); ax.annotate("Sun",(0,0),(6,6),textcoords="offset points",color="#FDB813",fontsize=9)
    # host fixed on +x in co-rotating frame
    ax.plot(a_host,0,"o",color=col,ms=11,zorder=8,markeredgecolor="white",markeredgewidth=0.5)
    ax.annotate(host,(a_host,0),(6,6),textcoords="offset points",color=col,fontsize=9)
    # L-point markers in co-rot frame
    for nm,(ang,rad) in {"L4":(60,a_host),"L5":(-60,a_host),"L3":(180,a_host)}.items():
        x=rad*np.cos(np.radians(ang)); y=rad*np.sin(np.radians(ang))
        ax.plot(x,y,"*",color="white",ms=12,zorder=7); ax.annotate(nm,(x,y),(5,5),textcoords="offset points",color="white",fontsize=8)
    # trajectories
    for k in range(co.shape[1]):
        c = "#2ca02c" if surv[k] else "#777"
        al = 0.55 if surv[k] else 0.25
        ax.plot(co[:,k,0], co[:,k,1], color=c, lw=0.5, alpha=al)
    ax.plot([],[],color="#2ca02c",lw=1.5,label="survived (librating)")
    ax.plot([],[],color="#777",lw=1.5,label="escaped")
    lim = a_host*1.5
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.tick_params(colors="#444"); ax.spines[:].set_color("#222")
    ax.set_xlabel("x (co-rotating with "+host+")",color="white"); ax.set_ylabel("y",color="white")
    ax.legend(fontsize=8,loc="upper right",facecolor="#11131f",edgecolor="#444",labelcolor="white")
    ax.set_title(f"{host} {which} cloud, frame co-rotating with {host}",color="white",fontsize=11)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)
    st.caption("Green = stayed co-orbital (radius within 25% of the host's); grey = left the "
               "co-orbital zone. Stable L4/L5 trace **tadpole** librations around the point; "
               "collinear seeds drift off.")

with tab_surv:
    fig2, ax2 = plt.subplots(figsize=(9,4)); fig2.patch.set_facecolor("white")
    ax2.plot(times, 100*alive, color="#2ca02c", lw=2.2)
    ax2.set_xlabel("time (yr)"); ax2.set_ylabel("surviving fraction (%)")
    ax2.set_ylim(0,105); ax2.grid(alpha=0.3); ax2.spines[['top','right']].set_visible(False)
    ax2.set_title(f"{host} {which} — {'all 8 planets' if len(perturbers)>1 else 'host only'}")
    st.pyplot(fig2, use_container_width=True)
    st.markdown(f"""
- **Final survival:** {surv.sum()}/{len(surv)} particles ({100*surv.mean():.0f}%) after **{t_max:,.0f} yr**.
- Try switching **L4 → L1** (watch survival collapse — saddles are unstable), or
  toggling **all 8 planets** (outer-planet perturbations slowly erode survival,
  most visibly for inner-planet hosts and over long integrations).
""")

with tab_phys:
    st.markdown(r"""
### What this shows, and how it connects to v3

**v3** locates the five equilibria $\nabla\Omega=0$ of the *restricted* problem
and reads the curvature $\Omega_{yy}$ to tell saddles from extrema. **v5** asks
the *dynamical* question: seed real test particles and integrate them in the
time-dependent field — which equilibria actually hold particles?

- **$L_4,L_5$** are linearly stable when $\mu<\mu_{\rm Routh}=0.03852$. Particles
  execute **tadpole** librations (period $\approx T_{\rm host}/\sqrt{27\mu/4}$ —
  ~148 yr for Jupiter) and survive for very long times. This is why the Jupiter
  ($>$7000), Neptune, Mars and Earth Trojans exist.
- **$L_1,L_2,L_3$** are saddles of $\Omega$ ($\Omega_{yy}<0$). They are
  dynamically **unstable**: a seeded particle drifts away on an instability
  timescale set by the positive Lyapunov exponent. (Real spacecraft at Sun–Earth
  $L_1/L_2$ must station-keep.)

**Full field vs restricted.** With only the host planet this is the classical
restricted three-body problem and L4/L5 are stable essentially forever. Adding
the other planets makes the field genuinely **time-dependent**: secular
resonances and close approaches slowly erode the stable zones. Over the
integration times here the effect is modest for Jupiter, but it is the mechanism
that, over Gyr, sculpts the real Trojan populations and clears unstable regions.

**Numerics.** Heliocentric velocity-Verlet with the indirect term (the Sun's
reflex acceleration from each planet) — without it the Lagrange points are not
equilibria. Planets move on circular, coplanar orbits at their mean motions
(J2000 phases): a clean, honest approximation that captures the dominant
co-orbital dynamics, not a full ephemeris.
""")
