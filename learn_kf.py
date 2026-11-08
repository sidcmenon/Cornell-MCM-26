"""
learn_kf.py
Learn the median m of the fire-avoidance weight k_F for a heterogeneous crowd.

Per replicate r (common random numbers: identical across every m):
  - ignition: one random flammable cell, not an exit and not occupied
  - (k_S, k_D) per agent ~ N(x*, c H^-1), redrawn while k_S <= 0, where x* and H come
    from the quadratic fit to results_refined.npz (same fit as refine.py)
  - k_F per agent = m * exp(sigma_F * z), z ~ N(0, 1)   (m = 0 means no fire avoidance)
  - agent dynamics seed, and a separate fire seed (the fire evolves identically for every
    m, so differences between m come from behaviour, not from different fires)

Objective: J = casualty fraction + lam * mean survivor exit time / t_ref (fire_objective).
Raw casualty and exit-time numbers are saved, so J can be recombined with another lam.

    python learn_kf.py
"""

import itertools
import os
from multiprocessing import Pool

import numpy as np

from bailey_grid import build_grid, EXIT, WALL
from fire_sim import FireEvacSim, fire_objective

CONFIG = dict(
    # crowd heterogeneity
    c=1.0,              # mean steps lost per person from (k_S, k_D) spread: Sigma = c H^-1
    sigma_F=0.5,        # lognormal spread of k_F (factor e^0.5 ~ 1.65 per sd)
    # simulation (same settings as the no-fire calibration)
    mu=0.3, alpha=0.3, delta=0.3, door_cap=1, door_q=0.6,
    ell=8,              # fire sensing range, cells
    T_engulf=360.0,     # target median engulf time, seconds (sets fire speed)
    fire_kwargs=dict(ignite_odds=30, burn_odds=60, difficulty=2,
                     spread_scale=8.0),   # from calibrate_s.py: ~95% of ignitions take hold
    burn_time=5.0,      # seconds in fire before becoming a casualty
    dt=0.36,            # seconds per step
    head_start=0,       # fire ticks before evacuation starts
    T=2000,
    # objective and search
    lam=0.1, t_ref=150.0,
    n_reps=50,
    base_seed=7,
)
M_VALUES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])

# calibrate_s.py at s = 8: fire ticking with probability 0.166 per step engulfs in a median
# 180 s. q scales the fire's time linearly, so any target engulf time T needs
Q_AT_180 = 0.166


def fire_tick_prob(cfg):
    q = Q_AT_180 * 180.0 / cfg["T_engulf"]
    if not 0 < q <= 1:
        raise ValueError(f"T_engulf={cfg['T_engulf']} needs q={q:.3f}, outside (0, 1]")
    return q


def burn_steps(cfg):
    return max(1, int(round(cfg["burn_time"] / cfg["dt"])))


def basin_from_refine(path="results_refined.npz"):
    """Quadratic fit of mean exit time over the refined (k_S, k_D) grid -> (x*, H)."""
    r = np.load(path)
    kS, kD, res = r["k_S"], r["k_D"], r["res"][..., 0]
    S, Dg = np.meshgrid(kS, kD, indexing="ij")
    X = np.column_stack([np.ones(S.size), S.ravel(), Dg.ravel(),
                         S.ravel() ** 2, (S * Dg).ravel(), Dg.ravel() ** 2])
    c, *_ = np.linalg.lstsq(X, res.mean(-1).ravel(), rcond=None)
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    if np.any(np.linalg.eigvalsh(H) <= 0):
        raise ValueError("fitted Hessian is not positive definite")
    return np.linalg.solve(H, -c[1:3]), H


def draw_crowd(n, x_star, H, c, sigma_F, rng):
    """Per-agent (k_S, k_D) ~ N(x*, c H^-1) with k_S > 0, and standard normals z_F."""
    cov = c * np.linalg.inv(H)
    k = rng.multivariate_normal(x_star, cov, size=n)
    bad = k[:, 0] <= 0
    while bad.any():
        k[bad] = rng.multivariate_normal(x_star, cov, size=int(bad.sum()))
        bad = k[:, 0] <= 0
    return k[:, 0], k[:, 1], rng.standard_normal(n)


_S = {}


def _init(grid, seats, x_star, H, cfg):
    _S.update(grid=grid, seats=seats, x_star=x_star, H=H, cfg=cfg)


def evaluate(task):
    m, r = task
    cfg, grid, seats = _S["cfg"], _S["grid"], _S["seats"]
    rng_ign, rng_crowd, rng_dyn, rng_fire = (
        np.random.default_rng(s)
        for s in np.random.SeedSequence([cfg["base_seed"], r]).spawn(4))
    occupied = np.zeros(grid.shape, bool)
    occupied[seats[:, 0], seats[:, 1]] = True
    cand = np.argwhere((grid != WALL) & (grid != EXIT) & ~occupied)
    ign = [tuple(cand[rng_ign.integers(len(cand))])]

    k_S, k_D, z_F = draw_crowd(len(seats), _S["x_star"], _S["H"], cfg["c"],
                               cfg["sigma_F"], rng_crowd)
    k_F = m * np.exp(cfg["sigma_F"] * z_F)

    sim = FireEvacSim(grid, mu=cfg["mu"], alpha=cfg["alpha"], delta=cfg["delta"],
                      door_cap=cfg["door_cap"], door_q=cfg["door_q"], ell=cfg["ell"],
                      fire_tick_prob=fire_tick_prob(cfg), fire_kwargs=cfg["fire_kwargs"],
                      burn_steps=burn_steps(cfg))
    res = sim.run(seats, k_S, k_D, k_F, ign, cfg["T"], rng_dyn,
                  head_start=cfg["head_start"], fire_rng=rng_fire)
    o = fire_objective(res, cfg["T"], cfg["lam"], cfg["t_ref"])
    return [o["J"], o["casualty"], o["burned"], o["trapped"], o["mean_exit"]]


def search(m_values, cfg=CONFIG, processes=None):
    g = build_grid()
    grid = g["grid"]
    seats = np.array([(a[0], a[1]) for a in g["agents"]])
    x_star, H = basin_from_refine()
    print("x* =", np.round(x_star, 3), " Sigma = c H^-1 =",
          np.round(cfg["c"] * np.linalg.inv(H), 3).tolist())
    print(f"fire: engulf {cfg['T_engulf']:.0f} s -> tick prob {fire_tick_prob(cfg):.4f};"
          f" casualty after {burn_steps(cfg)} steps in fire ({cfg['burn_time']:.0f} s)")
    tasks = list(itertools.product(range(len(m_values)), range(cfg["n_reps"])))
    with Pool(processes or os.cpu_count(), initializer=_init,
              initargs=(grid, seats, x_star, H, cfg)) as pool:
        out = pool.map(evaluate, [(m_values[i], r) for i, r in tasks])
    return np.array(out).reshape(len(m_values), cfg["n_reps"], 5)


def report(res, m_values):
    names = ["J", "casualty", "burned", "trapped", "mean_exit"]
    mean = res.mean(1)
    se = res.std(1, ddof=1) / np.sqrt(res.shape[1])
    print(f"{'m':>6} " + " ".join(f"{n:>16}" for n in names))
    for i, m in enumerate(m_values):
        print(f"{m:6.2f} " + " ".join(f"{mean[i, k]:9.4f} ±{se[i, k]:6.4f}"
                                      for k in range(len(names))))
    b = int(np.argmin(mean[:, 0]))
    # paired comparison with each other m (same replicates)
    d = res[:, :, 0] - res[b, :, 0]
    dse = d.std(1, ddof=1) / np.sqrt(res.shape[1])
    close = [m_values[i] for i in range(len(m_values)) if i != b and d[i].mean() < 2 * dse[i]]
    print(f"best m = {m_values[b]}  (within 2 paired SE: {close})")
    if b in (0, len(m_values) - 1):
        print("best m is at the edge of the range: extend M_VALUES")
    return b


if __name__ == "__main__":
    res = search(M_VALUES)
    report(res, M_VALUES)
    np.savez("results_kf.npz", m=M_VALUES, res=res,
             columns=np.array(["J", "casualty", "burned", "trapped", "mean_exit"]))