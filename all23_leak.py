"""Is the leak (Eq 9) ever strictly necessary? All 23 systems: pure Broyden (λ=0,
Eq 8 alone) vs with the leak (λ=2). Full pipeline, thr=-0.6, polish on."""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
grad_curv=_ns["grad_curv"]; lpoints=_ns["lpoints"]; build_grid=_ns["build_grid"]; newton_polish=_ns["newton_polish"]
B={"Sun":1.989e30,"Mercury":3.285e23,"Venus":4.867e24,"Earth":5.972e24,"Moon":7.342e22,
 "Mars":6.390e23,"Phobos":1.066e16,"Deimos":1.476e15,"Jupiter":1.898e27,"Io":8.932e22,
 "Europa":4.800e22,"Ganymede":1.482e23,"Callisto":1.076e23,"Saturn":5.683e26,"Titan":1.345e23,
 "Enceladus":1.080e20,"Rhea":2.307e21,"Uranus":8.681e25,"Titania":3.527e21,"Oberon":3.014e21,
 "Neptune":1.024e26,"Triton":2.139e22,"Pluto":1.303e22,"Charon":1.586e21}
SYS=[("Sun-Mercury","Sun","Mercury"),("Sun-Venus","Sun","Venus"),("Sun-Earth","Sun","Earth"),
 ("Sun-Mars","Sun","Mars"),("Sun-Jupiter","Sun","Jupiter"),("Sun-Saturn","Sun","Saturn"),
 ("Sun-Uranus","Sun","Uranus"),("Sun-Neptune","Sun","Neptune"),("Sun-Pluto","Sun","Pluto"),
 ("Earth-Moon","Earth","Moon"),("Mars-Phobos","Mars","Phobos"),("Mars-Deimos","Mars","Deimos"),
 ("Jupiter-Io","Jupiter","Io"),("Jupiter-Europa","Jupiter","Europa"),("Jupiter-Ganymede","Jupiter","Ganymede"),
 ("Jupiter-Callisto","Jupiter","Callisto"),("Saturn-Titan","Saturn","Titan"),("Saturn-Enceladus","Saturn","Enceladus"),
 ("Saturn-Rhea","Saturn","Rhea"),("Uranus-Titania","Uranus","Titania"),("Uranus-Oberon","Uranus","Oberon"),
 ("Neptune-Triton","Neptune","Triton"),("Pluto-Charon","Pluto","Charon")]
def sim(mu,start,leak,gamma=0.3,alpha=15.0,eps_c=0.3,thr=-0.6,dpmin=1e-4,dt=0.01,max_steps=120000,conv_thr=1e-4):
    pos=np.array(start,float); vel=np.zeros(2); s=-1.0; Bm=np.eye(2)
    gx,gy,_=grad_curv(pos[0],pos[1],mu); p_prev=pos.copy(); g_prev=np.array([gx,gy])
    for _ in range(max_steps):
        x,y=pos; gx,gy,_=grad_curv(x,y,mu); gn=np.hypot(gx,gy); g=np.array([gx,gy])
        dp=pos-p_prev; dg=g-g_prev; nd=dp@dp
        if nd>dpmin*dpmin:
            Bm=Bm+np.outer(dg-Bm@dp,dp)/nd; Bm=0.5*(Bm+Bm.T); Bm=Bm+leak*(np.eye(2)-Bm)*dt
        s+=alpha*(-np.tanh((Bm[1,1]-thr)/eps_c)-s)*dt; s=max(-1,min(1,s))
        p_prev=pos.copy(); g_prev=g
        ax=2*vel[1]-gx-gamma*vel[0]; ay=-2*vel[0]+s*gy-gamma*vel[1]
        vel+=np.array([ax,ay])*dt; pos=pos+vel*dt
        if gn<conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos)>12: break
    return pos
def n5(mu,leak):
    L=lpoints(mu); f={k:0 for k in L}
    for (x,y) in build_grid(mu,fill=6):
        pt=sim(mu,[x,y],leak)
        if np.isfinite(pt).all():
            pol,ok=newton_polish(pt,mu)
            if ok: pt=pol
        if np.isfinite(pt).all():
            d,lab=1e9,None
            for k,v in L.items():
                dd=np.linalg.norm(pt-v)
                if dd<d: d,lab=dd,k
            if d<0.05: f[lab]+=1
    return sum(1 for v in f.values() if v>0)
print(f"{'system':17}{'mu':>9} | {'λ=0 Broyden':>12} | {'λ=2 leak':>10}")
t0=t2=0
for sname,p1,p2 in SYS:
    mu=B[p2]/(B[p1]+B[p2]); a=n5(mu,0.0); b=n5(mu,2.0); t0+=(a==5); t2+=(b==5)
    print(f"{sname:17}{mu:9.1e} | {a}/5{'':>8} | {b}/5")
print(f"\n5/5 totals:  λ=0 (Eq 8 alone) = {t0}/23   |   λ=2 (Eq 8+9) = {t2}/23")
