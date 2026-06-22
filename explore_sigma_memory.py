"""
Test the reviewer's hypothesis: fold MEMORY into the correction current sigma
(or into the dissipation it controls) instead of using an inert multiplier w_L.

Decisive questions per variant:
  Q1  Does memory provide the dissipation?  -> converge with NO explicit constant
      damping (gamma0 = 0).  If yes, memory is genuinely computing (Lyapunov role).
  Q2  Does it rescue small-mu failures (Sun-Mercury, mu=1.65e-7)?
  Q3  Is it faster than baseline on Earth-Moon?

Shared, verified physics (one gradient routine for all variants -> no drift).
"""
import numpy as np
from scipy.optimize import brentq

# Single source of truth for CR3BP geometry (was duplicated here verbatim).
from nbody_trojan import grad_curv, lpoints

EPS = 1e-9


def simulate(variant, mu, start, beta, gamma0, dt=0.01, max_steps=120000,
             conv_thr=1e-4, cap=8.0, kappa=4.0, ksharp=6.0):
    """
    variant:
      'baseline'   : ay = -2vx + sigma*w_y*gy - gamma0*vy ; w grows beta|g| (inert mult.)
      'gamma_mem'  : MEMORY IS DAMPING. gamma_eff = gamma0 + kappa*m, m_dot=beta|grad|.
                     ay = -2vx + sigma*gy - gamma_eff*vy ; ax = 2vy - gx - gamma_eff*vx.
                     Tests whether memory can REPLACE the explicit damping.
      'sigma_mem'  : sigma is a per-axis MEMORY current s that relaxes toward the
                     curvature target AND brakes velocity:
                     s_dot = beta*(sigma_tgt*|gy| - lam*s) ; the dissipation is
                     supplied by a memory-controlled friction proportional to |s|.
      'sigma_smooth': continuous correction sigma_c = -tanh(ksharp*oyy) (no memory),
                     keeps gamma0 -- isolates whether a SOFT sign helps speed/rescue.
    Returns (conv_step or None, final_pos, max_mem)
    """
    pos = np.array(start, float); vel = np.zeros(2)
    wx = wy = 1.0          # baseline multipliers
    m  = 0.0               # scalar accumulated-violation memory (gamma_mem)
    sx = sy = 0.0          # signed per-axis sigma-memory (sigma_mem)
    mem_max = 0.0
    lam = 0.5
    for step in range(max_steps):
        x, y = pos
        gx, gy, oyy = grad_curv(x, y, mu)
        gn = np.hypot(gx, gy)
        sigma = 1.0 if oyy < 0 else -1.0

        if variant == 'baseline':
            wx = min(wx + beta*abs(gx)*dt, cap)
            wy = min(wy + beta*abs(gy)*dt, cap)
            ax = 2*vel[1] - wx*gx - gamma0*vel[0]
            ay = -2*vel[0] + sigma*wy*gy - gamma0*vel[1]
            mem_max = max(mem_max, wx, wy)

        elif variant == 'gamma_mem':
            m = m + beta*gn*dt                       # ratchet memory (monotone)
            g_eff = gamma0 + kappa*m
            ax = 2*vel[1] - gx - g_eff*vel[0]
            ay = -2*vel[0] + sigma*gy - g_eff*vel[1]
            mem_max = max(mem_max, m)

        elif variant == 'sigma_mem':
            # signed correction memory; relaxes toward -sign(oyy)*|g|, saturating
            sx = sx + beta*( (-np.sign(oyy) if False else 0.0) )  # x uses plain gradient
            sy = sy + beta*( sigma*abs(gy) - lam*sy )*dt
            # memory-controlled friction (dissipation grows with |s|)
            fric = gamma0 + kappa*abs(sy)
            ax = 2*vel[1] - gx - fric*vel[0]
            ay = -2*vel[0] + sy*gy - fric*vel[1]
            mem_max = max(mem_max, abs(sy))

        elif variant == 'sigma_smooth':
            sigma_c = -np.tanh(ksharp*oyy)            # soft, curvature-scaled sign
            ax = 2*vel[1] - gx - gamma0*vel[0]
            ay = -2*vel[0] + sigma_c*gy - gamma0*vel[1]
            mem_max = 1.0
        else:
            raise ValueError(variant)

        vel += np.array([ax, ay])*dt
        pos = pos + vel*dt
        if gn < conv_thr:
            return step, pos, mem_max
        if not np.isfinite(pos).all() or np.linalg.norm(pos) > 12:
            return None, pos, mem_max
    return None, pos, mem_max


def grid(mu, n=10):
    L = lpoints(mu)
    hill = (mu/3)**(1/3); off = max(0.5*hill, 0.02)
    ax = np.array([L["L1"][0]-off, L["L1"][0]+off, L["L2"][0]-off, L["L2"][0]+off,
                   L["L3"][0]-0.04, L["L3"][0]+0.04])
    fill = np.linspace(-1.4, 1.4, max(n-6,1))
    xs = np.sort(np.unique(np.round(np.concatenate([fill, ax]), 8)))[:n]
    nn = (n-3)//2; npz = n-3-nn
    ys = np.sort(np.unique(np.concatenate([np.linspace(-1.2,-0.15,max(nn,1)),
                 [-0.05,0,0.05], np.linspace(0.15,1.2,max(npz,1))])))[:n]
    return xs, ys, hill

def assess(variant, mu, beta, gamma0, max_steps=120000):
    L = lpoints(mu)
    xs, ys, hill = grid(mu)
    excl_p = min(0.03, 0.3*hill); excl_s = min(0.012, 0.3*hill)
    counts = {k:0 for k in L}; nstart=0; ndiv=0; nspur=0; steps=[]
    for x in xs:
        for y in ys:
            if (x+mu)**2+y**2 < excl_p**2: continue
            if (x-1+mu)**2+y**2 < excl_s**2: continue
            nstart += 1
            cs, final, mm = simulate(variant, mu, [x,y], beta, gamma0, max_steps=max_steps)
            if cs is None: ndiv += 1; continue
            steps.append(cs)
            best,db=None,1e9
            for k,v in L.items():
                d=np.linalg.norm(final-v)
                if d<db: db,best=d,k
            if db<0.12: counts[best]+=1
            else: nspur += 1
    nfound=sum(1 for v in counts.values() if v>0)
    medstep=int(np.median(steps)) if steps else -1
    return dict(counts=counts,nstart=nstart,ndiv=ndiv,nspur=nspur,
                nfound=nfound,medstep=medstep)

def show(tag, r):
    c=r["counts"]
    print(f"{tag:30s} found {r['nfound']}/5 | L1:{c['L1']:2d} L2:{c['L2']:2d} "
          f"L3:{c['L3']:2d} L4:{c['L4']:2d} L5:{c['L5']:2d} | "
          f"med_steps={r['medstep']:6d} div={r['ndiv']:3d} spurious={r['nspur']:3d}")

MU_EM = 7.342e22/(5.972e24+7.342e22)   # Earth-Moon 0.01214
MU_SM = 3.285e23/(1.989e30+3.285e23)   # Sun-Mercury 1.65e-7

print("="*92)
print("Q3  SPEED on Earth-Moon (gamma0=0.6 where explicit damping present)")
print("="*92)
show("baseline beta=0.001",      assess('baseline', MU_EM, 0.001, 0.6, 60000))
show("sigma_smooth (no mem)",    assess('sigma_smooth', MU_EM, 0.0, 0.6, 60000))
show("gamma_mem g0=0.6 b=0.5",   assess('gamma_mem', MU_EM, 0.5, 0.6, 60000))
show("sigma_mem g0=0.6 b=0.5",   assess('sigma_mem', MU_EM, 0.5, 0.6, 60000))

print()
print("="*92)
print("Q1  DECISIVE: NO explicit damping (gamma0=0). Does memory provide the dissipation?")
print("    baseline should FAIL (memory inert). gamma_mem/sigma_mem should SUCCEED if memory computes.")
print("="*92)
show("baseline g0=0 b=0.001",    assess('baseline', MU_EM, 0.001, 0.0, 60000))
show("baseline g0=0 b=1.0",      assess('baseline', MU_EM, 1.0, 0.0, 60000))
for b in [0.2, 0.5, 1.0, 2.0]:
    show(f"gamma_mem g0=0 b={b}",  assess('gamma_mem', MU_EM, b, 0.0, 60000))
for b in [0.5, 1.0, 2.0]:
    show(f"sigma_mem g0=0 b={b}",  assess('sigma_mem', MU_EM, b, 0.0, 60000))

print()
print("="*92)
print("Q2  RESCUE small-mu: Sun-Mercury (mu=1.65e-7), default-style params")
print("="*92)
show("baseline g0=0.6 b=0.001",  assess('baseline', MU_SM, 0.001, 0.6, 120000))
show("gamma_mem g0=0.6 b=0.5",   assess('gamma_mem', MU_SM, 0.5, 0.6, 120000))
show("gamma_mem g0=0 b=1.0",     assess('gamma_mem', MU_SM, 1.0, 0.0, 120000))
show("sigma_smooth",             assess('sigma_smooth', MU_SM, 0.0, 0.6, 120000))
