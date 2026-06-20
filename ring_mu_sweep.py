"""
Where does the memory-as-dissipation DMM actually beat Newton on the ring?
Sweep the ring mass mu and compare ground-truth coverage + robustness.
"""
import numpy as np
from scipy.optimize import fsolve

N = 8; R = 1.0; G = 1.0; M0 = 1.0; EPS = 1e-12
PLANETS = np.array([[R*np.cos(2*np.pi*j/N), R*np.sin(2*np.pi*j/N)] for j in range(N)])

def make(mu):
    m = mu/N
    csc = sum(1.0/np.sin(np.pi*j/N) for j in range(1, N))
    om2 = G*M0/R**3 + G*m/(4*R**3)*csc
    om  = np.sqrt(om2)
    def gh(x, y):
        rs = np.sqrt(x*x+y*y)+EPS
        gx = om2*x - G*M0*x/rs**3; gy = om2*y - G*M0*y/rs**3
        hxx = om2 - G*M0/rs**3 + 3*G*M0*x*x/rs**5
        hyy = om2 - G*M0/rs**3 + 3*G*M0*y*y/rs**5
        hxy = 3*G*M0*x*y/rs**5
        for px, py in PLANETS:
            dx, dy = x-px, y-py; d = np.sqrt(dx*dx+dy*dy)+EPS
            gx -= G*m*dx/d**3; gy -= G*m*dy/d**3
            hxx += -G*m/d**3 + 3*G*m*dx*dx/d**5
            hyy += -G*m/d**3 + 3*G*m*dy*dy/d**5
            hxy += 3*G*m*dx*dy/d**5
        return np.array([gx,gy]), np.array([[hxx,hxy],[hxy,hyy]])
    return gh, om

def sing(p, e1=0.04, e2=0.02):
    return np.hypot(p[0],p[1])<e1 or bool(np.any(np.hypot(PLANETS[:,0]-p[0],PLANETS[:,1]-p[1])<e2))

def newton(gh, s, it=100, tol=1e-9):
    p = np.array(s, float)
    for _ in range(it):
        g,h = gh(p[0],p[1])
        if np.linalg.norm(g)<tol: return p, True
        if sing(p) or np.linalg.norm(p)>8: return p, False
        try: p = p + np.linalg.solve(h,-g)
        except np.linalg.LinAlgError: return p, False
        if not np.isfinite(p).all(): return p, False
    return p, False

def dmm(gh, om, s, beta=0.5, gamma0=0.2, kappa=1.0, m_cap=3.0, dt=0.01, max_steps=20000, thr=1e-4):
    pos=np.array(s,float); vel=np.zeros(2); mem=0.0
    for _ in range(max_steps):
        g,h = gh(pos[0],pos[1]); gn=np.linalg.norm(g)
        if gn<thr:
            try:
                sol,_,ier,_ = fsolve(lambda q: gh(q[0],q[1])[0], pos,
                                     fprime=lambda q: gh(q[0],q[1])[1], full_output=True, xtol=1e-12)
                if ier==1 and np.linalg.norm(gh(sol[0],sol[1])[0])<1e-9 and not sing(sol): return sol, True
            except Exception: pass
            return pos, True
        if sing(pos) or np.linalg.norm(pos)>8: return pos, False
        lam,V = np.linalg.eigh(h)
        F = -(np.sign(lam)*(V.T@g))@V.T
        mem = min(mem+beta*gn*dt, m_cap); ge = gamma0+kappa*mem
        ax = 2*om*vel[1]+F[0]-ge*vel[0]; ay = -2*om*vel[0]+F[1]-ge*vel[1]
        vel += np.array([ax,ay])*dt; pos += vel*dt
    return pos, False

def cluster(pts, tol=2e-3):
    u=[]
    for p in pts:
        if not any(np.hypot(p[0]-q[0],p[1]-q[1])<tol for q in u): u.append(p)
    return u

def ground_truth(gh, n=120, seed=0):
    rng=np.random.default_rng(seed); f=[]
    for _ in range(n*n):
        s=rng.uniform(-1.6,1.6,2)
        if sing(s): continue
        p,ok=newton(gh,s)
        if ok and not sing(p) and np.hypot(*p)<3: f.append(p)
    return cluster(f)

def coverage(truth, found):
    return sum(1 for t in truth if any(np.hypot(t[0]-f[0],t[1]-f[1])<1e-2 for f in found))

print(f"{'mu':>8} | {'#eq':>4} | {'Newton cov':>11} {'div':>4} | {'DMM cov':>9} {'div':>4}")
print("-"*60)
G_=21; xs=np.linspace(-1.5,1.5,G_); ys=np.linspace(-1.5,1.5,G_)
for mu in [2e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 8e-2]:
    gh, om = make(mu)
    truth = ground_truth(gh)
    starts=[(x,y) for x in xs for y in ys if not sing((x,y))]
    nf=[]; nd=0
    for s in starts:
        p,ok=newton(gh,s)
        if ok and not sing(p) and np.hypot(*p)<3: nf.append(p)
        else: nd+=1
    df=[]; dd=0
    for s in starts:
        p,ok=dmm(gh,om,s)
        if ok and not sing(p) and np.hypot(*p)<3: df.append(p)
        else: dd+=1
    ncov=coverage(truth, cluster(nf)); dcov=coverage(truth, cluster(df)); T=len(truth)
    print(f"{mu:8.1e} | {T:4d} | {ncov:3d}/{T} ({100*ncov//T:3d}%) {nd:4d} | "
          f"{dcov:3d}/{T} ({100*dcov//T:3d}%) {dd:4d}")
