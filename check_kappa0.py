"""
Is the memory load-bearing? Run the PAPER's pipeline (run_discovery + build_grid
from solar_system_dmm_v3.py) on all 23 systems for four configurations:

  kappa=1, polish=ON   baseline (paper default; expect 23/23)
  kappa=0, polish=ON   memory OFF (gamma_eff=0, NO dissipation) + Newton refine
  kappa=0, polish=OFF  memory OFF, dynamics ONLY (no Newton)
  kappa=1, polish=OFF  memory ON,  dynamics ONLY (no Newton)

If kappa=0/polish=ON still finds all 5, the Newton refinement is doing the work,
not the memory. The polish=OFF columns isolate what the dynamics alone achieve.
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

def found_count(mu, kappa, polish, max_steps=60000):
    res, L = run_discovery(mu, 0.5, 0.0, kappa, 10.0, 0.01, max_steps, 1e-4, 10, 10, polish)
    f = {k: 0 for k in L}
    for r in res:
        if r["label"]: f[r["label"]] += 1
    return sum(1 for v in f.values() if v > 0)

cfg = [("k=1 polishON", 1.0, True), ("k=0 polishON", 0.0, True),
       ("k=0 polishOFF", 0.0, False), ("k=1 polishOFF", 1.0, False)]
print(f"{'system':18} {'mu':>9} | " + " | ".join(f"{name:>13}" for name,_,_ in cfg))
print("-"*90)
tally = {name: 0 for name,_,_ in cfg}
for sname, p1, p2 in SYSTEMS:
    mu = BODIES[p2]/(BODIES[p1]+BODIES[p2])
    row = []
    for name, kap, pol in cfg:
        nf = found_count(mu, kap, pol)
        if nf == 5: tally[name] += 1
        row.append(f"{nf}/5")
    print(f"{sname:18} {mu:9.1e} | " + " | ".join(f"{c:>13}" for c in row))
print("-"*90)
print("systems with 5/5: " + " | ".join(f"{name}={tally[name]}/23" for name,_,_ in cfg))
