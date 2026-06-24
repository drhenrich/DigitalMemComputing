"""
The RIGHT control (per D. Henrich): does a STATIC constant dissipation find all
five Lagrange points for all pairs, or is the time-varying (memory) dissipation
actually needed?

  memory:  kappa=1, gamma0=0   -> gamma_eff = kappa*m  (accumulating)
  static:  kappa=0, gamma0=c   -> gamma_eff = c        (constant)

Same machine, same seed grid (build_grid), same sigma correction. If some single
constant gamma0 also gives 23/23, the memory is not necessary (just self-tuning).
If no single gamma0 spans mu = 2e-9 ... 0.1 but the memory does, the memory's
adaptivity is the genuine advantage.
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

def found(mu, gamma0, kappa, max_steps=60000):
    res, L = run_discovery(mu, 0.5, gamma0, kappa, 10.0, 0.01, max_steps, 1e-4, 10, 10, True)
    f = {k: 0 for k in L}
    for r in res:
        if r["label"]: f[r["label"]] += 1
    return sum(1 for v in f.values() if v > 0)

# columns: memory baseline, then several constant gamma0 (kappa=0)
cols = [("memory k1 g0", 0.0, 1.0)] + [(f"static g0={g}", g, 0.0)
        for g in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)]
print(f"{'system':17}{'mu':>9} | " + " | ".join(f"{n:>12}" for n,_,_ in cols))
print("-"*120)
tally = {n: 0 for n,_,_ in cols}
for sname, p1, p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2])
    row = []
    for n, g0, kap in cols:
        nf = found(mu, g0, kap)
        if nf == 5: tally[n] += 1
        row.append(f"{nf}/5")
    print(f"{sname:17}{mu:9.1e} | " + " | ".join(f"{c:>12}" for c in row))
print("-"*120)
print("systems at 5/5: " + " | ".join(f"{n}={tally[n]}/23" for n,_,_ in cols))
