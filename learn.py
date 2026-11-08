"""
learn.py
Grid search for (k_S, k_D) minimizing mean capped exit time in Bailey Hall.

Start positions are the occupied chairs from bailey_grid (all 1010 by default, or a
random subset if FILL < 1). Common random numbers: replicate r uses the same seed at
every grid point, so differences between points are not drowned in seed noise.

    python learn.py          # coarse sweep, prints best point, saves results.npz
"""

import itertools
import os
from multiprocessing import Pool

import numpy as np

from bailey_grid import build_grid
from sim import EvacSim, build_static, objective

CONFIG = dict(
    mu=0.3, alpha=0.3, delta=0.3,
    cut_corners=False, seats_passable=False, winner="weighted", include_seats=False,
    door_cap=1,      # max agents through one door per step (None = unlimited)
    T=2000,          # step cap; keep well above the typical evacuation time
    fill=1.0,        # fraction of chairs occupied (subset redrawn per replicate)
    n_reps=20,       # replicates per grid point
    base_seed=12345,
)
K_S = np.linspace(0.0, 10.0, 11)
K_D = np.linspace(-5.0, 5.0, 11)    # negative k_D = avoid trafficked cells

_S = {}


def _init(grid, seats, static, cfg):
    _S.update(grid=grid, seats=seats, static=static, cfg=cfg)


def evaluate(point):
    k_S, k_D = point
    cfg = _S["cfg"]
    sim = EvacSim(_S["grid"], k_S, k_D, mu=cfg["mu"], alpha=cfg["alpha"], delta=cfg["delta"],
                  cut_corners=cfg["cut_corners"], seats_passable=cfg["seats_passable"],
                  winner=cfg["winner"], include_seats=cfg["include_seats"],
                  door_cap=cfg["door_cap"], static=_S["static"])
    seats = _S["seats"]
    n = int(round(cfg["fill"] * len(seats)))
    out = []
    for r in range(cfg["n_reps"]):
        rng_start, rng_dyn = (np.random.default_rng(s) for s in
                              np.random.SeedSequence([cfg["base_seed"], r]).spawn(2))
        start = seats if n == len(seats) else seats[rng_start.choice(len(seats), n, replace=False)]
        res = objective(sim.run(start, cfg["T"], rng_dyn), cfg["T"])
        out.append([res["mean_exit"], res["frac_out"], res["last_exit"]])
    return np.array(out)                                       # (n_reps, 3)


def grid_search(k_S_vals, k_D_vals, cfg=CONFIG, processes=None):
    g = build_grid()
    grid = g["grid"]
    seats = np.array([(a[0], a[1]) for a in g["agents"]])
    static = build_static(grid, cfg["seats_passable"], cfg["cut_corners"])
    stuck = ~np.isfinite(static[3][seats[:, 0], seats[:, 1]])
    if stuck.any():
        raise ValueError(f"{stuck.sum()} occupants cannot reach any exit")
    pts = list(itertools.product(k_S_vals, k_D_vals))
    with Pool(processes or os.cpu_count(), initializer=_init,
              initargs=(grid, seats, static, cfg)) as pool:
        res = pool.map(evaluate, pts)
    return np.array(res).reshape(len(k_S_vals), len(k_D_vals), cfg["n_reps"], 3)


def summarize(res, k_S_vals, k_D_vals):
    mean = res[..., 0].mean(-1)
    se = res[..., 0].std(-1, ddof=1) / np.sqrt(res.shape[2])
    i, j = np.unravel_index(np.argmin(mean), mean.shape)
    print(f"best k_S={k_S_vals[i]:.3g}  k_D={k_D_vals[j]:.3g}  "
          f"mean exit={mean[i, j]:.2f} +/- {se[i, j]:.2f} steps  "
          f"frac out={res[i, j, :, 1].mean():.3f}")
    if i in (0, len(k_S_vals) - 1) or j in (0, len(k_D_vals) - 1):
        print("optimum is on the grid boundary: widen the range before refining")
    return mean, se, (i, j)


def refine(best_ij, k_S_vals, k_D_vals, n=9):
    """Finer grid spanning one coarse cell on each side of the best point."""
    i, j = best_ij
    s = np.linspace(k_S_vals[max(i - 1, 0)], k_S_vals[min(i + 1, len(k_S_vals) - 1)], n)
    t = np.linspace(k_D_vals[max(j - 1, 0)], k_D_vals[min(j + 1, len(k_D_vals) - 1)], n)
    return s, t


if __name__ == "__main__":
    res = grid_search(K_S, K_D)
    mean, se, best = summarize(res, K_S, K_D)
    np.savez("results.npz", k_S=K_S, k_D=K_D, res=res, mean=mean, se=se)
    s, t = refine(best, K_S, K_D)
    res2 = grid_search(s, t)
    mean2, se2, _ = summarize(res2, s, t)
    np.savez("results_refined.npz", k_S=s, k_D=t, res=res2, mean=mean2, se=se2)