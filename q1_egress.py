"""
q1_egress.py
Egress times for the full house (1010 patrons): how long until everyone is out.

For each scenario, over R replicates with common random numbers (same crowd, ignition
site and random streams across scenarios), reports per-replicate statistics of the exit
times of patrons who got out, averaged over replicates:
  mean, median, 95th percentile, and last out (the egress time of the whole hall),
in time steps and seconds, plus the casualty fraction and the busiest door's load.

Scenarios (edit SCENARIOS below):
  concert            no fire, no ushers, everyone starts moving at once
  concert + delay    no fire, no ushers, pre-movement delay (lognormal, median DELAY_S s)
  fire               fire as in learn_kf.py, no ushers
  fire + ushers      fire, the optimized ushers from usher_search_w0.5.json (w_bar = 0.5)

    python q1_egress.py
"""

import json
import os
from multiprocessing import Pool

import numpy as np

import learn_kf
from bailey_grid import build_grid, EXIT, WALL
from usher_sim import UsherFireSim, ESCAPED

KF = learn_kf.CONFIG
DT = KF["dt"]
R = 50
SEED = 777
M = 1.0                 # learned median fire weight
DELAY_S = 30.0          # median pre-movement delay for the delayed scenario, seconds
DELAY_SIGMA = 0.5       # lognormal spread of the delay
W_BAR, KAPPA = 0.5, 4.0

USHERS = []
if os.path.exists("usher_search_w0.5.json"):
    prog = json.load(open("usher_search_w0.5.json"))
    USHERS = [(tuple(c), d) for c, d in prog.get("local", prog["greedy"])]

SCENARIOS = {
    "concert":         dict(fire=False, ushers=[], delay=0.0),
    "concert + delay": dict(fire=False, ushers=[], delay=DELAY_S),
    "fire":            dict(fire=True, ushers=[], delay=0.0),
    "fire + ushers":   dict(fire=True, ushers=USHERS, delay=0.0),
}

_W = {}


def _init():
    g = build_grid()
    grid = g["grid"]
    seats = np.array([(a[0], a[1]) for a in g["agents"]])
    x_star, H = learn_kf.basin_from_refine()
    sim = UsherFireSim(grid, R=10, mu=KF["mu"], alpha=KF["alpha"], delta=KF["delta"],
                       door_cap=KF["door_cap"], door_q=KF["door_q"], ell=KF["ell"],
                       fire_tick_prob=learn_kf.fire_tick_prob(KF),
                       fire_kwargs=KF["fire_kwargs"], burn_steps=learn_kf.burn_steps(KF))
    occupied = np.zeros(grid.shape, bool)
    occupied[seats[:, 0], seats[:, 1]] = True
    cand = np.argwhere((grid != WALL) & (grid != EXIT) & ~occupied)
    _W.update(sim=sim, seats=seats, x_star=x_star, H=H, cand=cand)


def _one(task):
    name, r = task
    sc = SCENARIOS[name]
    sim, seats = _W["sim"], _W["seats"]
    n = len(seats)
    rng_ign, rng_crowd, rng_dyn, rng_fire, rng_comp, rng_delay = (
        np.random.default_rng(s) for s in np.random.SeedSequence([SEED, r]).spawn(6))
    ign = [tuple(_W["cand"][rng_ign.integers(len(_W["cand"]))])] if sc["fire"] else []
    k_S, k_D, z_F = learn_kf.draw_crowd(n, _W["x_star"], _W["H"], KF["c"],
                                        KF["sigma_F"], rng_crowd)
    k_F = M * np.exp(KF["sigma_F"] * z_F)
    w = rng_comp.beta(KAPPA * W_BAR, KAPPA * (1 - W_BAR), n)
    z = rng_delay.standard_normal(n)
    delay = (np.round(sc["delay"] / DT * np.exp(DELAY_SIGMA * z)).astype(int)
             if sc["delay"] > 0 else None)
    res = sim.run(seats, k_S, k_D, k_F, ign, KF["T"], rng_dyn, fire_rng=rng_fire,
                  ushers=sc["ushers"], compliance=w, start_delay=delay)
    out = res["outcome"] == ESCAPED
    t = res["exit_t"][out]
    doors = np.bincount(res["exit_door"][out], minlength=sim.n_doors)
    stats = [t.mean(), np.median(t), np.percentile(t, 95), t.max(),
             1 - out.mean(), doors.max()] if t.size else [np.nan] * 5 + [0]
    return name, r, stats


if __name__ == "__main__":
    tasks = [(name, r) for name in SCENARIOS for r in range(R)]
    with Pool(os.cpu_count(), initializer=_init) as pool:
        out = pool.map(_one, tasks, chunksize=1)
    res = {name: np.zeros((R, 6)) for name in SCENARIOS}
    for name, r, s in out:
        res[name][r] = s
    if not USHERS:
        print("(no usher_search_w0.5.json found: 'fire + ushers' ran with no ushers)")
    print(f"{R} replicates; times are means over replicates, steps (seconds at {DT} s/step)")
    print(f"{'scenario':>16} {'mean':>14} {'median':>14} {'95th pct':>14} {'last out':>18}"
          f" {'casualties':>11} {'busiest door':>13}")
    for name, a in res.items():
        m = a.mean(0)
        se_last = a[:, 3].std(ddof=1) / np.sqrt(R)
        print(f"{name:>16} {m[0]:6.1f} ({m[0]*DT:5.1f}) {m[1]:6.1f} ({m[1]*DT:5.1f})"
              f" {m[2]:6.1f} ({m[2]*DT:5.1f}) {m[3]:6.1f}±{se_last:4.1f} ({m[3]*DT:5.1f})"
              f" {100*m[4]:10.2f}% {m[5]:13.1f}")
    np.savez("results_q1_egress.npz", names=np.array(list(res)),
             res=np.array([res[k] for k in res]),
             columns=np.array(["mean", "median", "p95", "last", "casualty", "busiest_door"]))
