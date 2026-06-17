import numpy as np
import matplotlib.pyplot as plt

def simulate_dmm_lpoint():
    # ── 1. System Constants (Earth-Moon Rotating Frame) ───────────────────────
    mu = 0.0121          # Moon/(Earth+Moon) mass ratio
    x1, x2 = -mu, 1 - mu  # Earth and Moon x-positions (normalized units)

    # L4 analytic: equilateral triangle with both primaries (r1=r2=1)
    L4 = np.array([0.5 - mu, np.sqrt(3) / 2])  # ≈ (0.4879, 0.8660)

    # ── 2. Initialization ─────────────────────────────────────────────────────
    # Starting point is perturbed from L4 in both x and y.
    # L4 is only a local attractor via Coriolis (Routh's criterion, μ < 0.0385);
    # the basin of attraction in the rotating frame does NOT include [0.5, 0.5].
    # Any starting point with zero velocity far below L4 falls into Earth.
    pos = L4 + np.array([0.2, -0.3])    # offset from L4
    vel = np.array([0.0, 0.0])

    # ── 3. DMM Memory Variables ───────────────────────────────────────────────
    short_mem = 0.0
    long_mem  = 1.0
    alpha = 0.05
    beta  = 0.001

    # Light damping: must satisfy γ ≪ 2ω_libration ≈ 1.7 (underdamped regime)
    # so the Coriolis effect can complete oscillation cycles before energy is lost
    gamma = 0.1
    dt             = 0.01
    max_steps      = 500_000
    grad_threshold = 1e-6
    eps            = 1e-8

    trajectory = []

    # ── 4. Emulation Loop (Forward Euler) ─────────────────────────────────────
    for step in range(max_steps):
        r1 = np.sqrt((pos[0] - x1)**2 + pos[1]**2) + eps
        r2 = np.sqrt((pos[0] - x2)**2 + pos[1]**2) + eps

        # ∂Ω/∂pos  where  Ω = (x²+y²)/2 + (1-μ)/r1 + μ/r2
        # Zeros of ∇Ω are the Lagrange points (the "clause" in DMM language)
        gx = pos[0] - (1-mu)*(pos[0]-x1)/r1**3 - mu*(pos[0]-x2)/r2**3
        gy = pos[1] - (1-mu)* pos[1]      /r1**3 - mu* pos[1]      /r2**3
        grad_Omega = np.array([gx, gy])
        grad_norm  = np.linalg.norm(grad_Omega)

        # ── 5. DMM Memory Update ──────────────────────────────────────────────
        short_mem = (1 - alpha)*short_mem + alpha*grad_norm
        long_mem  = min(long_mem + beta*short_mem*dt, 5.0)

        # ── 6. Rotating-Frame EOM + DMM Memory + Dissipation ──────────────────
        #
        # True rotating-frame equations of motion:
        #   ẍ =  2ẏ + ∂Ω/∂x
        #   ÿ = −2ẋ + ∂Ω/∂y
        #
        # The Coriolis terms (±2ẋ/ẏ) are essential: they convert L4 from a
        # saddle of the static potential into a stable attractor for μ < 0.0385.
        # long_mem amplifies ∇Ω as "memory pressure" toward the L-point.
        # Dissipation −γv causes the libration spirals to decay to the point.
        # EOM: ẍ = 2ẏ − ∂Ω/∂x,  ÿ = −2ẋ − ∂Ω/∂y  →  force = −∇Ω
        coriolis        = np.array([ 2.0*vel[1], -2.0*vel[0]])
        potential_force = -grad_Omega * long_mem
        damping         = -gamma * vel

        acceleration = coriolis + potential_force + damping

        vel += acceleration * dt
        pos += vel * dt
        trajectory.append(pos.copy())

        if grad_norm < grad_threshold:
            print(f"L-Point resolved at step {step}.")
            break
    else:
        print(f"Did not converge in {max_steps} steps.")

    error = np.linalg.norm(pos - L4)
    print(f"Resolved position : ({pos[0]:.6f}, {pos[1]:.6f})")
    print(f"L4 analytic       : ({L4[0]:.6f}, {L4[1]:.6f})")
    print(f"Position error    : {error:.2e}")
    print(f"Gradient |∇Ω|     : {grad_norm:.2e}")

    # ── Visualization ──────────────────────────────────────────────────────────
    traj = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.plot(x1, 0, 'ro', markersize=14, zorder=5, label='Earth (M₁)')
    ax.plot(x2, 0, 'bo', markersize=10, zorder=5, label='Moon (M₂)')
    ax.plot(*L4, 'y^', markersize=14, zorder=5,
            label=f'L4 analytic ({L4[0]:.4f}, {L4[1]:.4f})')
    ax.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.4, lw=0.8,
            label='DMM Instanton Path (libration spiral)')
    ax.plot(traj[0, 0], traj[0, 1], 'cs', markersize=10, zorder=6,
            label=f'Start ({traj[0,0]:.4f}, {traj[0,1]:.4f})')
    ax.plot(traj[-1, 0], traj[-1, 1], 'k*', markersize=16, zorder=6,
            label=f'Resolved L4 ({traj[-1,0]:.4f}, {traj[-1,1]:.4f})')

    ax.set_title(
        "DMM Emulation: Restricted 3-Body → L4 Lagrange Point\n"
        f"(Earth–Moon rotating frame, μ = {mu},  γ = {gamma})\n"
        "Forces: Coriolis (±2ẋ/ẏ) + memory-amplified ∇Ω + dissipation"
    )
    ax.set_xlabel("x (rotating frame)")
    ax.set_ylabel("y (rotating frame)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_dmm_lpoint()
