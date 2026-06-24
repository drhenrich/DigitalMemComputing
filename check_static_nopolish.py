"""
Confirm it is the DYNAMICS (damping), not the Newton polish, that finds the
points -- and compare convergence SPEED (memory vs best constant). polish=False,
so labels come from the raw dynamical endpoint (tol 0.12). 'conv' is the step at
which |grad Omega| < conv_thr (None if never).
"""
import sys, types
import numpy as np
class _M(types.ModuleType):
    def __getattr__(s, n): return lambda *a, **k: None
_st = _M("streamlit"); _st.cache_data = lambda **k: (lambda f: f)
sys.modules["streamlit"] = _st
_ns = {}
exec(open("solar_system_dmm_v3.py").read().split("# ── UI")[0], _ns)
run_discovery = _ns["run_discovery"]

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

def probe(mu, gamma0, kappa, max_steps=120000):
    res, L = run_discovery(mu, 0.5, gamma0, kappa, 10.0, 0.01, max_steps, 1e-4, 10, 10, False)
    f = {k: 0 for k in L}
    convs = []
    for r in res:
        if r["label"]:
            f[r["label"]] += 1
            if r["conv"] is not None: convs.append(r["conv"])
    nfound = sum(1 for v in f.values() if v > 0)
    med = int(np.median(convs)) if convs else -1
    return nfound, med

cols = [("memory k1", 0.0, 1.0), ("static g0=0.1", 0.1, 0.0),
        ("static g0=0.2", 0.2, 0.0), ("static g0=0.5", 0.5, 0.0)]
print("polish OFF (raw dynamical endpoint). cell = found/5  (median conv-step)")
print(f"{'system':17}{'mu':>9} | " + " | ".join(f"{n:>20}" for n,_,_ in cols))
print("-"*120)
tally = {n: 0 for n,_,_ in cols}
for sname, p1, p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2])
    row = []
    for n, g0, kap in cols:
        nf, med = probe(mu, g0, kap)
        if nf == 5: tally[n] += 1
        row.append(f"{nf}/5 ({med})")
    print(f"{sname:17}{mu:9.1e} | " + " | ".join(f"{c:>20}" for c in row))
print("-"*120)
print("systems at 5/5: " + " | ".join(f"{n}={tally[n]}/23" for n,_,_ in cols))
