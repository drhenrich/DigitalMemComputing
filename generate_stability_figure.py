"""
Publication figure for the extended paper: dynamical stability cross-check.
White background, no internal titles, PDF. Uses the validated nbody_trojan core.

fig_stability.pdf:
  Left  — Jupiter L4 cloud in the co-rotating frame: green tadpole librations
          (stable). Sun, Jupiter, and L3/L4/L5 marked.
  Right — surviving fraction vs time for clouds seeded at L4, L1, L2, L3
          (Sun+Jupiter): L4 stable, L1/L2 escape fast, L3 erodes slowly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nbody_trojan as N

plt.rcParams.update({
    "font.family":"serif", "font.size":11, "axes.labelsize":12,
    "axes.linewidth":1.4, "xtick.major.width":1.2, "ytick.major.width":1.2,
    "xtick.minor.width":0.8, "ytick.minor.width":0.8,
    "xtick.direction":"in", "ytick.direction":"in",
    "legend.framealpha":0.95, "legend.edgecolor":"0.7", "figure.facecolor":"white",
})

HOST = "Jupiter"
a = N.PLANETS[HOST][0]
COL = {"L4":"#2ca02c","L1":"#d62728","L2":"#ff7f0e","L3":"#9467bd"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))

# ── Left: L4 tadpole librations (co-rotating frame) ─────────────────────────────
print("L4 cloud …")
R0, V0 = N.seed_cloud(HOST, "L4", 30, dr_spread=0.04, dphi_spread=6.0, seed=3)
t, traj, surv, alive = N.integrate_cloud(R0, V0, 250*N.PLANETS[HOST][2], 0.01, (HOST,), HOST, sample=30)
co = np.stack([N.to_corotating(t, traj[:,k,:], HOST) for k in range(traj.shape[1])], axis=1)

th = np.linspace(0, 2*np.pi, 300)
axL.plot(a*np.cos(th), a*np.sin(th), color="0.8", lw=0.7, ls=":")           # Jupiter orbit
for k in range(co.shape[1]):
    c = COL["L4"] if surv[k] else "0.6"
    axL.plot(co[:,k,0], co[:,k,1], color=c, lw=0.4, alpha=0.6)
axL.plot(0, 0, "o", color="#FDB813", ms=14, zorder=8)
axL.annotate("Sun", (0,0), (7,7), textcoords="offset points", fontsize=9)
axL.plot(a, 0, "o", color="#c88b3a", ms=11, zorder=8, markeredgecolor="k", markeredgewidth=0.4)
axL.annotate(HOST, (a,0), (-4,9), textcoords="offset points", fontsize=9)
# true L-point markers
mu = N.q(HOST)/(1+N.q(HOST))
from scipy.optimize import brentq
def g0(x):
    r1=abs(x+mu)+1e-12; r2=abs(x-1+mu)+1e-12
    return x-(1-mu)*(x+mu)/r1**3-mu*(x-1+mu)/r2**3
L3r = abs(brentq(g0,-2.5,-mu-1e-4)+mu)*a
for nm,(ang,rad) in {"L4":(60,a),"L5":(-60,a),"L3":(180,L3r)}.items():
    axL.plot(rad*np.cos(np.radians(ang)), rad*np.sin(np.radians(ang)), "*",
             color="k", ms=13, zorder=9)
    axL.annotate(nm, (rad*np.cos(np.radians(ang)), rad*np.sin(np.radians(ang))),
                 (5,5), textcoords="offset points", fontsize=9, fontweight="bold")
lim = a*1.45
axL.set_xlim(-lim, lim); axL.set_ylim(-lim, lim); axL.set_aspect("equal")
axL.set_xlabel(r"$x$ (frame co-rotating with Jupiter, AU)")
axL.set_ylabel(r"$y$ (AU)")
axL.xaxis.set_minor_locator(ticker.AutoMinorLocator())
axL.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# ── Right: survival fraction vs time for L4 / L1 / L2 / L3 ───────────────────────
for which in ["L4","L1","L2","L3"]:
    print(f"survival {which} …")
    Tmax = 15000.0
    R0,V0 = N.seed_cloud(HOST, which, 40, dr_spread=0.03, dphi_spread=3.0, seed=5)
    tt, tr, sv, al = N.integrate_cloud(R0, V0, Tmax, 0.01, (HOST,), HOST, sample=40)
    axR.plot(tt, 100*al, color=COL[which], lw=2.2, label=f"${which}$")
axR.set_xlabel(r"time (yr)"); axR.set_ylabel(r"surviving fraction (\%)")
axR.set_ylim(-3, 105); axR.grid(alpha=0.25)
axR.xaxis.set_minor_locator(ticker.AutoMinorLocator())
axR.yaxis.set_minor_locator(ticker.AutoMinorLocator())
axR.legend(fontsize=10, loc="center right", title="seeded at")

plt.tight_layout()
fig.savefig("fig_stability.pdf", bbox_inches="tight")
fig.savefig("fig_stability.png", dpi=200, bbox_inches="tight")
print("-> fig_stability.pdf")
