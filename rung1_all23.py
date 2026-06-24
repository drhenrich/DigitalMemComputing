"""
Confirm rung-1 (smooth memory-sigma) across all 23 solar-system pairs.
  s_dot = alpha*(-tanh(Omega_yy/eps_c) - s),  y-force = s*gy
Full structural grid + Newton polish, constant gamma. Compare to the discrete switch.
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
build_grid = _ns["build_grid"]; newton_polish = _ns["newton_polish"]
grad_curv = _ns["grad_curv"]; lpoints = _ns["lpoints"]

BODIES = {"Sun":1.989e30,"Mercury":3.285e23,"Venus":4.867e24,"Earth":5.972e24,
 "Moon":7.342e22,"Mars":6.390e23,"Phobos":1.066e16,"Deimos":1.476e15,
 "Jupiter":1.898e27,"Io":8.932e22,"Europa":4.800e22,"Ganymede":1.482e23,
 "Callisto":1.076e23,"Saturn":5.683e26,"Titan":1.345e23,"Enceladus":1.080e20,
 "Rhea":2.307e21,"Uranus":8.681e25,"Titania":3.527e21,"Oberon":3.014e21,
 "Neptune":1.024e26,"Triton":2.139e22,"Pluto":1.303e22,"Charon":1.586e21}
SYSTEMS = [("Sun-Mercury","Sun","Mercury"),("Sun-Venus","Sun","Venus"),
 ("Sun-Earth","Sun","Earth"),("Sun-Mars","Sun","Mars"),("Sun-Jupiter","Sun","Jupiter"),
 ("Sun-Saturn","Sun","Saturn"),("Sun-Uranus","Sun","Uranus"),("Sun-Neptune","Sun","Neptune"),
 ("Sun-Pluto","Sun","Pluto"),("Earth-Moon","Earth","Moon"),("Mars-Phobos","Mars","Phobos"),
 ("Mars-Deimos","Mars","Deimos"),("Jupiter-Io","Jupiter","Io"),("Jupiter-Europa","Jupiter","Europa"),
 ("Jupiter-Ganymede","Jupiter","Ganymede"),("Jupiter-Callisto","Jupiter","Callisto"),
 ("Saturn-Titan","Saturn","Titan"),("Saturn-Enceladus","Saturn","Enceladus"),
 ("Saturn-Rhea","Saturn","Rhea"),("Uranus-Titania","Uranus","Titania"),
 ("Uranus-Oberon","Uranus","Oberon"),("Neptune-Triton","Neptune","Triton"),
 ("Pluto-Charon","Pluto","Charon")]

def sim(mu, start, mode, gamma=0.3, alpha=8.0, eps_c=0.2,
        dt=0.01, max_steps=120000, conv_thr=1e-4):
    pos = np.array(start, float); vel = np.zeros(2); s = -1.0
    for _ in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu); gn = np.hypot(gx, gy)
        if mode == "discrete":
            s = 1.0 if oyy < 0 else -1.0
        else:
            st = -np.tanh(oyy/eps_c); s += alpha*(st - s)*dt; s = max(-1, min(1, s))
        ax = 2*vel[1] - gx - gamma*vel[0]
        ay = -2*vel[0] + s*gy - gamma*vel[1]
        vel += np.array([ax, ay])*dt; pos = pos + vel*dt
        if gn < conv_thr: break
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12: break
    return pos

def discover(mu, mode, **kw):
    L = lpoints(mu); found = {k: 0 for k in L}
    for (x, y) in build_grid(mu, fill=6):
        pt = sim(mu, [x, y], mode, **kw)
        if np.isfinite(pt).all():
            pol, ok = newton_polish(pt, mu)
            if ok: pt = pol
        if np.isfinite(pt).all():
            d, lab = 1e9, None
            for k, v in L.items():
                dd = np.linalg.norm(pt - v)
                if dd < d: d, lab = dd, k
            if d < 0.05: found[lab] += 1
    return sum(1 for v in found.values() if v > 0)

cols = [("discrete", dict(mode="discrete")),
        ("mem eps0.1", dict(mode="mem", eps_c=0.1)),
        ("mem eps0.2", dict(mode="mem", eps_c=0.2))]
print(f"{'system':17}{'mu':>9} | " + " | ".join(f"{n:>10}" for n,_ in cols))
print("-"*70)
tally = {n: 0 for n,_ in cols}
for sname, p1, p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2]); row = []
    for n, kw in cols:
        k = discover(mu, **kw)
        if k == 5: tally[n] += 1
        row.append(f"{k}/5")
    print(f"{sname:17}{mu:9.1e} | " + " | ".join(f"{c:>10}" for c in row))
print("-"*70)
print("at 5/5: " + " | ".join(f"{n}={tally[n]}/23" for n,_ in cols))
