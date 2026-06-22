"""fig_memory_clean.pdf — memory dynamics along converging Earth-Moon trajectories
(no multiplier comparison): m & gamma_eff ratchet up, T -> 0, |grad Omega| -> 0."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
from nbody_trojan import grad_curv, lpoints
plt.rcParams.update({"font.family":"serif","axes.linewidth":1.4,
    "xtick.major.width":1.2,"ytick.major.width":1.2,
    "xtick.direction":"in","ytick.direction":"in","figure.facecolor":"white"})
L_COLORS={"L1":"#d62728","L2":"#ff7f0e","L3":"#9467bd","L4":"#2ca02c","L5":"#1f77b4"}
MU=0.01215

def sim(start, beta=0.5, gamma0=0.0, kappa=1.0, m_cap=10.0, dt=0.01, max_steps=200000, thr=1e-4):
    pos=np.array(start,float); vel=np.zeros(2); m=0.0
    S=[];M=[];G=[];T=[];GN=[]
    for s in range(max_steps):
        x,y=pos; gx,gy,oyy=grad_curv(x,y,MU); gn=np.hypot(gx,gy)
        m=min(m+beta*gn*dt,m_cap); ge=gamma0+kappa*m; sig=1.0 if oyy<0 else -1.0
        ax=2*vel[1]-gx-ge*vel[0]; ay=-2*vel[0]+sig*gy-ge*vel[1]
        vel+=np.array([ax,ay])*dt; pos=pos+vel*dt
        if s%20==0:
            S.append(s);M.append(m);G.append(ge);GN.append(gn);T.append(0.5*(vel[0]**2+vel[1]**2))
        if gn<thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos)>12: break
    return pos,np.array(S),np.array(M),np.array(G),np.array(T),np.array(GN)

L=lpoints(MU)
# one representative converging trajectory per L-point
seeds={"L1":(L["L1"][0]-0.02,0.0),"L2":(L["L2"][0]+0.02,0.0),"L3":(-1.02,0.0),
       "L4":(0.54,0.80),"L5":(0.54,-0.80)}
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(13,4))
for k,s in seeds.items():
    pos,S,M,G,T,GN=sim(s); c=L_COLORS[k]
    # keep only if it converged near the intended L-point
    if np.linalg.norm(pos-L[k])>0.2 or len(S)<3: continue
    a1.plot(S,M,color=c,lw=2.0,label=f"${k}$"); a1.plot(S,G,color=c,lw=1.4,ls="--")
    a2.semilogy(S,T+1e-14,color=c,lw=2.0)
    a3.semilogy(S,GN+1e-14,color=c,lw=2.0)
a1.set_xlabel(r"integration step $n$"); a1.set_ylabel(r"$m$ (solid),  $\gamma_{\rm eff}=\kappa m$ (dashed)")
a1.legend(fontsize=9,loc="lower right")
a2.set_xlabel(r"integration step $n$"); a2.set_ylabel(r"kinetic energy $T$"); a2.set_ylim(1e-12,1e2)
a3.axhline(1e-4,color="0.4",lw=1.0,ls="--")
a3.set_xlabel(r"integration step $n$"); a3.set_ylabel(r"$|\nabla\Omega|$"); a3.set_ylim(1e-6,1e1)
for a in (a1,a2,a3): a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
fig.savefig("fig_memory_clean.pdf",bbox_inches="tight")
fig.savefig("fig_memory_clean.png",dpi=200,bbox_inches="tight")
print("saved fig_memory_clean.pdf")
