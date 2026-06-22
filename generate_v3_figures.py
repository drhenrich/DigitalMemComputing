"""
Publication figures for dmm_lagrange_v3.tex (memory-as-dissipation).
White background, no internal titles, PDF output, one representative curve / L-point.

fig_v3_discovery.pdf  : instanton paths to all 5 L-points (Earth-Moon)
fig_v3_energy.pdf     : THE centerpiece -- kinetic energy T(t):
                        inert multiplier (gamma=0) never dissipates;
                        memory-as-dissipation (gamma0=0) drives T -> 0.
                        + memory m(t) and gamma_eff(t) ramping up.
fig_v3_curvature.pdf  : Omega_yy correction-current map + effective potential.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import brentq

# Single source of truth for CR3BP geometry (was duplicated here verbatim).
# Omega here is the CR3BP effective potential.
from nbody_trojan import grad_curv, effective_potential as Omega, lpoints

EPS = 1e-9
MU  = 0.01215
L_COLORS = {"L1":"#d62728","L2":"#ff7f0e","L3":"#9467bd","L4":"#2ca02c","L5":"#1f77b4"}

plt.rcParams.update({
    "font.family":"serif", "font.size":11, "axes.labelsize":12,
    "axes.linewidth":1.4, "xtick.major.width":1.2, "ytick.major.width":1.2,
    "xtick.minor.width":0.8, "ytick.minor.width":0.8,
    "xtick.direction":"in", "ytick.direction":"in",
    "legend.framealpha":0.95, "legend.edgecolor":"0.7", "figure.facecolor":"white",
})

def sim_memdiss(mu,start,beta=0.5,gamma0=0.0,kappa=1.0,m_cap=10.0,dt=0.01,
                max_steps=200000,thr=1e-4,record=False):
    pos=np.array(start,float);vel=np.zeros(2);m=0.0
    traj=[pos.copy()];hist={"step":[],"m":[],"geff":[],"gn":[],"T":[]}
    for s in range(max_steps):
        x,y=pos;gx,gy,oyy=grad_curv(x,y,mu);gn=np.hypot(gx,gy)
        m=min(m+beta*gn*dt,m_cap);ge=gamma0+kappa*m;sig=1.0 if oyy<0 else -1.0
        ax=2*vel[1]-gx-ge*vel[0];ay=-2*vel[0]+sig*gy-ge*vel[1]
        vel+=np.array([ax,ay])*dt;pos=pos+vel*dt
        if record and s%20==0:
            hist["step"].append(s);hist["m"].append(m);hist["geff"].append(ge)
            hist["gn"].append(gn);hist["T"].append(0.5*(vel[0]**2+vel[1]**2))
        if s%20==0: traj.append(pos.copy())
        if gn<thr:
            traj.append(pos.copy())
            if record:
                hist["step"].append(s);hist["m"].append(m);hist["geff"].append(ge)
                hist["gn"].append(gn);hist["T"].append(0.5*(vel[0]**2+vel[1]**2))
            return np.array(traj),s,hist
        if not np.isfinite(pos).all() or np.linalg.norm(pos)>12:
            return np.array(traj),None,hist
    return np.array(traj),None,hist

def sim_multiplier(mu,start,beta=0.5,gamma=0.0,dt=0.01,max_steps=40000,record=True):
    """Inert-memory ansatz: w_L multiplies the force. With gamma=0, T does not dissipate."""
    pos=np.array(start,float);vel=np.zeros(2);w=1.0
    hist={"step":[],"T":[],"gn":[]}
    for s in range(max_steps):
        x,y=pos;gx,gy,oyy=grad_curv(x,y,mu);gn=np.hypot(gx,gy)
        w=w+beta*gn*dt;sig=1.0 if oyy<0 else -1.0
        ax=2*vel[1]-w*gx-gamma*vel[0];ay=-2*vel[0]+sig*w*gy-gamma*vel[1]
        vel+=np.array([ax,ay])*dt;pos=pos+vel*dt
        if s%20==0:
            hist["step"].append(s);hist["T"].append(0.5*(vel[0]**2+vel[1]**2));hist["gn"].append(gn)
        if not np.isfinite(pos).all() or np.linalg.norm(pos)>12: break
    return hist

ANA=lpoints(MU)

# ── Discovery: small grid, pick a representative trajectory per L-point ─────────
def grid(mu,n=10):
    L=lpoints(mu);hill=(mu/3)**(1/3);off=max(0.5*hill,0.02)
    ax=np.array([L["L1"][0]-off,L["L1"][0]+off,L["L2"][0]-off,L["L2"][0]+off,
                 L["L3"][0]-0.04,L["L3"][0]+0.04])
    xs=np.sort(np.unique(np.round(np.concatenate([np.linspace(-1.4,1.4,4),ax]),8)))[:n]
    ys=np.sort(np.unique(np.concatenate([np.linspace(-1.2,-0.15,3),[-0.05,0,0.05],
               np.linspace(0.15,1.2,4)])))[:n]
    return xs,ys,hill

print("Running discovery for representative paths ...")
xs,ys,hill=grid(MU);ep=min(0.03,0.3*hill);es=min(0.012,0.3*hill)
reps={}
for x in xs:
    for y in ys:
        if (x+MU)**2+y**2<ep**2 or (x-1+MU)**2+y**2<es**2: continue
        traj,cs,_=sim_memdiss(MU,[x,y])
        if cs is None: continue
        end=traj[-1];best,db=None,1e9
        for k,v in ANA.items():
            d=np.linalg.norm(end-v)
            if d<db: db,best=d,k
        if db<0.05:
            if best not in reps or len(traj)<len(reps[best][0]):
                reps[best]=(traj,[x,y])
print("representatives:",{k:reps[k][1] for k in reps})

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — discovery map
# ════════════════════════════════════════════════════════════════════════════
fig1,ax=plt.subplots(figsize=(6,5.5))
xg=np.linspace(-1.6,1.6,400);yg=np.linspace(-1.45,1.45,400)
XG,YG=np.meshgrid(xg,yg)
ax.contour(XG,YG,np.clip(Omega(XG,YG,MU),0,4),levels=30,colors="0.75",linewidths=0.4)
for k,(traj,st) in reps.items():
    ax.plot(traj[:,0],traj[:,1],color=L_COLORS[k],lw=2.2,zorder=3)
ax.plot([-MU],[0],"o",color="#1f77b4",ms=12,zorder=6,label="Earth")
ax.plot([1-MU],[0],"o",color="0.4",ms=7,zorder=6,label="Moon")
for k,v in ANA.items():
    ax.plot(*v,"*",color=L_COLORS[k],ms=18,zorder=7,markeredgecolor="k",markeredgewidth=0.5)
    ax.annotate(k,xy=v,xytext=(v[0]+0.09,v[1]+0.07),color=L_COLORS[k],fontsize=11,fontweight="bold")
ax.set_xlim(-1.65,1.65);ax.set_ylim(-1.48,1.48)
ax.set_xlabel(r"$x$ (co-rotating frame)");ax.set_ylabel(r"$y$ (co-rotating frame)")
ax.set_aspect("equal")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator());ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend(fontsize=9,loc="upper right")
plt.tight_layout();fig1.savefig("fig_v3_discovery.pdf",bbox_inches="tight");fig1.savefig("fig_v3_discovery.png",dpi=200,bbox_inches="tight")
print("-> fig_v3_discovery.pdf")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — THE centerpiece: kinetic energy + memory dynamics
# ════════════════════════════════════════════════════════════════════════════
start=[0.30,0.60]   # an L4-bound start that converges under memory-as-dissipation
_,cs_md,hm=sim_memdiss(MU,start,record=True)
hmul=sim_multiplier(MU,start,gamma=0.0)   # inert multiplier, no damping

fig2,(a1,a2)=plt.subplots(1,2,figsize=(12,4.3))

# panel A: kinetic energy
a1.semilogy(np.array(hmul["step"]),np.array(hmul["T"])+1e-12,color="#d62728",lw=2.0,
            label=r"multiplier $w_L$, $\gamma=0$ (memory inert)")
a1.semilogy(np.array(hm["step"]),np.array(hm["T"])+1e-12,color="#2ca02c",lw=2.0,
            label=r"memory-as-dissipation, $\gamma_0=0$")
a1.set_xlabel(r"Integration step $n$");a1.set_ylabel(r"kinetic energy $T$")
a1.set_ylim(1e-10,1e2)
a1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
a1.legend(fontsize=9,loc="lower left")

# panel B: memory m(t) and gamma_eff(t)
a2.plot(np.array(hm["step"]),np.array(hm["m"]),color="#9467bd",lw=2.2,label=r"memory $m$")
a2.plot(np.array(hm["step"]),np.array(hm["geff"]),color="#1f77b4",lw=2.2,ls="--",
        label=r"$\gamma_{\rm eff}=\kappa m$")
a2b=a2.twinx()
a2b.semilogy(np.array(hm["step"]),np.array(hm["gn"])+1e-12,color="0.45",lw=1.6,label=r"$|\nabla\Omega|$")
a2b.set_ylabel(r"$|\nabla\Omega|$ (log)",color="0.3")
a2b.tick_params(axis="y",colors="0.3");a2b.set_ylim(1e-5,1e1)
a2.set_xlabel(r"Integration step $n$");a2.set_ylabel(r"$m,\ \gamma_{\rm eff}$")
a2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
l1,la1=a2.get_legend_handles_labels();l2,la2=a2b.get_legend_handles_labels()
a2.legend(l1+l2,la1+la2,fontsize=9,loc="center right")
for a in (a1,a2,a2b): a.spines[:].set_linewidth(1.4)
plt.tight_layout();fig2.savefig("fig_v3_energy.pdf",bbox_inches="tight");fig2.savefig("fig_v3_energy.png",dpi=200,bbox_inches="tight")
print(f"-> fig_v3_energy.pdf  (memory-as-dissipation converged at step {cs_md}; "
      f"multiplier T stays O(1))")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — curvature map + potential
# ════════════════════════════════════════════════════════════════════════════
fig3,(aL,aR)=plt.subplots(1,2,figsize=(11,4.5))
x2=np.linspace(-1.5,1.5,400);y2=np.linspace(-1.3,1.3,400);X2,Y2=np.meshgrid(x2,y2)
_,_,OYY=grad_curv(X2,Y2,MU);OYY=np.clip(OYY,-3,3)
im1=aL.pcolormesh(X2,Y2,OYY,cmap="RdBu",vmin=-3,vmax=3,shading="auto",rasterized=True)
aL.contour(X2,Y2,OYY,levels=[0],colors="#f1c40f",linestyles="--",linewidths=1.8)
cb1=fig3.colorbar(im1,ax=aL,fraction=0.046,pad=0.04);cb1.set_label(r"$\Omega_{yy}$")
for k,v in ANA.items():
    aL.plot(*v,"*",color=L_COLORS[k],ms=13,zorder=5,markeredgecolor="k",markeredgewidth=0.4)
aL.set_xlim(-1.5,1.5);aL.set_ylim(-1.3,1.3);aL.set_xlabel(r"$x$");aL.set_ylabel(r"$y$");aL.set_aspect("equal")
aL.xaxis.set_minor_locator(ticker.AutoMinorLocator());aL.yaxis.set_minor_locator(ticker.AutoMinorLocator())
Om=np.clip(Omega(X2,Y2,MU),0,4)
im2=aR.pcolormesh(X2,Y2,np.clip(Om,1,3),cmap="viridis",shading="auto",rasterized=True)
aR.contour(X2,Y2,Om,levels=25,colors="white",linewidths=0.3,alpha=0.4)
cb2=fig3.colorbar(im2,ax=aR,fraction=0.046,pad=0.04);cb2.set_label(r"$\Omega(x,y)$")
for k,v in ANA.items():
    aR.plot(*v,"*",color=L_COLORS[k],ms=13,zorder=5,markeredgecolor="k",markeredgewidth=0.4)
    aR.annotate(k,xy=v,xytext=(v[0]+0.07,v[1]+0.07),color="white",fontsize=9,fontweight="bold")
aR.set_xlim(-1.5,1.5);aR.set_ylim(-1.3,1.3);aR.set_xlabel(r"$x$");aR.set_ylabel(r"$y$");aR.set_aspect("equal")
aR.xaxis.set_minor_locator(ticker.AutoMinorLocator());aR.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout();fig3.savefig("fig_v3_curvature.pdf",bbox_inches="tight");fig3.savefig("fig_v3_curvature.png",dpi=200,bbox_inches="tight")
print("-> fig_v3_curvature.pdf")
print("done")
