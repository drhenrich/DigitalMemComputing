"""fig_memory_clean.pdf — memory dynamics along converging Earth-Moon trajectories,
one representative per Lagrange point (pulled from the actual run_discovery output).
m & gamma_eff ratchet up, kinetic energy T -> 0, |grad Omega| -> 0."""
import sys, types
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

# import the real v3 discovery (mock streamlit so the module loads headless)
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit")
def _cd(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
_st.cache_data = _cd
sys.modules["streamlit"] = _st
_src = open("solar_system_dmm_v3.py").read()
_ns = {}
exec(_src[:_src.find("# ── UI")], _ns)
simulate_v3, lpoints, L_COLORS = _ns["simulate_v3"], _ns["lpoints"], _ns["L_COLORS"]

plt.rcParams.update({"font.family":"serif","axes.linewidth":1.4,
    "xtick.major.width":1.2,"ytick.major.width":1.2,
    "xtick.direction":"in","ytick.direction":"in","figure.facecolor":"white"})

MU = 0.01215
L = lpoints(MU)
# one dedicated seed per L-point that converges cleanly through the dynamics
seeds = {
    "L1": (L["L1"][0]-0.02, 0.0),     # on-axis, just inside L1
    "L2": (L["L2"][0]+0.02, 0.0),     # on-axis, just outside L2
    "L3": (L["L3"][0], 0.03),         # small y-offset engages the correction current
    "L4": (0.52,  0.80),              # near the equilateral point
    "L5": (0.52, -0.80),
}
reps = {}
for k, s in seeds.items():
    traj, m_h, g_h, gn_h, T_h, conv, pos = simulate_v3(
        MU, s, 0.5, 0.0, 1.0, 10.0, 0.01, 400000, 1e-4)
    if np.linalg.norm(pos - L[k]) < 0.15 and len(gn_h) > 3:
        reps[k] = dict(m_h=m_h, g_h=g_h, gn_h=gn_h, T_h=T_h)
    else:
        print(f"  warn: {k} seed did not localize (|pos-L|={np.linalg.norm(pos-L[k]):.2f})")
print("representatives:", sorted(reps))

fig,(a1,a2,a3) = plt.subplots(1, 3, figsize=(13, 4))
for k in ("L1","L2","L3","L4","L5"):
    if k not in reps: continue
    r = reps[k]; c = L_COLORS[k]
    s  = np.arange(len(r["m_h"]))*20
    sg = np.arange(len(r["gn_h"]))*20
    sT = np.arange(len(r["T_h"]))*20
    a1.plot(s, r["m_h"], color=c, lw=2.0, label=f"${k}$")
    a1.plot(s, r["g_h"], color=c, lw=1.4, ls="--")
    a2.semilogy(sT, np.array(r["T_h"])+1e-14, color=c, lw=2.0)
    a3.semilogy(sg, np.array(r["gn_h"])+1e-14, color=c, lw=2.0)

a1.set_xlabel(r"integration step $n$")
a1.set_ylabel(r"$m$ (solid),  $\gamma_{\rm eff}=\kappa m$ (dashed)")
a1.legend(fontsize=9, loc="lower right", ncol=2)
a2.set_xlabel(r"integration step $n$"); a2.set_ylabel(r"kinetic energy $T$"); a2.set_ylim(1e-12,1e2)
a3.axhline(1e-4, color="0.4", lw=1.0, ls="--")
a3.set_xlabel(r"integration step $n$"); a3.set_ylabel(r"$|\nabla\Omega|$"); a3.set_ylim(1e-6,1e1)
for a in (a1,a2,a3): a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
fig.savefig("fig_memory_clean.pdf", bbox_inches="tight")
fig.savefig("fig_memory_clean.png", dpi=200, bbox_inches="tight")
print("saved fig_memory_clean.pdf with", len(reps), "L-point curves")
