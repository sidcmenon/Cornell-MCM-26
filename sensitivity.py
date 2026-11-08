"""
sensitivity.py
One-at-a-time sensitivity of the CA crowd model (sim.py) and the fire model (fire_sim.py).
Each setting changes one parameter from the baseline. Replicate r uses the same seeds in
every setting (common random numbers), so differences from the baseline are paired.

    python3 sensitivity.py ca     # ~1.5 min on 10 cores -> sens_ca.npz + markdown table
    python3 sensitivity.py fire   # ~40 min on 10 cores -> sens_fire.npz + markdown tables

Written up in sensitivity_test.md.
"""

import sys
from multiprocessing import Pool

import numpy as np

import learn
import learn_kf
from bailey_grid import build_grid
from sim import EvacSim, build_static, objective

DT = 0.36                     # s per step
K_S, K_D = 2.08, -2.98        # optimum from refine.py (measured-access geometry)
N_REPS = 50

CA_BASE = dict(learn.CONFIG, k_S=K_S, k_D=K_D)
CA_CHANGES = [("mu", 0.0), ("mu", 0.15), ("mu", 0.45), ("mu", 0.6),
              ("alpha", 0.1), ("alpha", 0.5), ("delta", 0.1), ("delta", 0.5),
              ("door_q", 0.45), ("door_q", 0.75),
              ("k_S", 0.75 * K_S), ("k_S", 1.25 * K_S), ("k_D", 0.75 * K_D), ("k_D", 1.25 * K_D),
              ("fill", 0.5), ("fill", 0.75),
              ("winner", "uniform"), ("cut_corners", True)]

FIRE_CHANGES = [("c", 0.5), ("c", 2.0), ("c", 4.0), ("sigma_F", 0.25), ("sigma_F", 1.0),
                ("ell", 4), ("ell", 12), ("T_engulf", 180.0), ("T_engulf", 720.0),
                ("burn_time", 2.5), ("burn_time", 10.0)]
LAMS = [0.0, 0.05, 0.1, 0.2, 0.5]   # objective weight on exit time (recombined, no rerun)

G = build_grid()
GRID, SEATS = G["grid"], np.array([(a[0], a[1]) for a in G["agents"]])
_STATIC = {}


def ca_run(task):
    key, val, r = task
    cfg = {**CA_BASE, **({key: val} if key else {})}
    sk = (cfg["seats_passable"], cfg["cut_corners"])
    if sk not in _STATIC:
        _STATIC[sk] = build_static(GRID, *sk)
    sim = EvacSim(GRID, cfg["k_S"], cfg["k_D"], mu=cfg["mu"], alpha=cfg["alpha"],
                  delta=cfg["delta"], cut_corners=cfg["cut_corners"],
                  seats_passable=cfg["seats_passable"], winner=cfg["winner"],
                  include_seats=cfg["include_seats"], door_cap=cfg["door_cap"],
                  door_q=cfg["door_q"], static=_STATIC[sk])
    rng_start, rng_dyn = (np.random.default_rng(s) for s in
                          np.random.SeedSequence([cfg["base_seed"], r]).spawn(2))
    n = int(round(cfg["fill"] * len(SEATS)))
    start = SEATS if n == len(SEATS) else SEATS[rng_start.choice(len(SEATS), n, replace=False)]
    o = objective(sim.run(start, cfg["T"], rng_dyn), cfg["T"])
    return o["mean_exit"], o["last_exit"]


def ca():
    settings = [(None, None)] + CA_CHANGES
    tasks = [(k, v, r) for k, v in settings for r in range(N_REPS)]
    with Pool() as pool:
        res = np.array(pool.map(ca_run, tasks)).reshape(len(settings), N_REPS, 2)
    np.savez("sens_ca.npz", settings=np.array([f"{k}={v}" for k, v in settings]), res=res)

    base = res[0]
    print(f"Baseline: mean exit {base[:, 0].mean():.1f} steps ({base[:, 0].mean() * DT:.1f} s), "
          f"last exit {base[:, 1].mean():.0f} steps ({base[:, 1].mean() * DT:.0f} s)\n")
    print("| Change | Mean exit (s) | Δ mean exit (s) | Δ % | Last exit (s) |")
    print("|---|---|---|---|---|")
    for (k, v), x in zip(settings[1:], res[1:]):
        d = x[:, 0] - base[:, 0]
        se = d.std(ddof=1) / np.sqrt(N_REPS)
        vs = f"{v:.3g}" if isinstance(v, float) else v
        print(f"| {k} = {vs} | {x[:, 0].mean() * DT:.1f} | {d.mean() * DT:+.1f} ± {se * DT:.1f} "
              f"| {100 * d.mean() / base[:, 0].mean():+.1f}% | {x[:, 1].mean() * DT:.0f} |")


def fire():
    x_star, H = learn_kf.basin_from_refine()
    m = learn_kf.M_VALUES
    settings = [(None, None)] + FIRE_CHANGES
    out = []
    for k, v in settings:
        cfg = {**learn_kf.CONFIG, "n_reps": N_REPS, **({k: v} if k else {})}
        out.append(learn_kf.search(m, cfg))          # (n_m, reps, 5): J, cas, burned, trapped, exit
    res = np.array(out)
    np.savez("sens_fire.npz", settings=np.array([f"{k}={v}" for k, v in settings]), m=m, res=res)

    def row(name, r, J):
        b = int(np.argmin(J.mean(1)))
        d = J - J[b]
        close = [m[i] for i in range(len(m)) if i != b
                 and d[i].mean() < 2 * d[i].std(ddof=1) / np.sqrt(N_REPS)]
        i1 = int(np.where(m == 1.0)[0][0])
        print(f"| {name} | {m[b]:g} | {', '.join(f'{c:g}' for c in close) or '-'} "
              f"| {100 * r[b, :, 1].mean():.2f}% | {100 * r[0, :, 1].mean():.2f}% "
              f"| {100 * r[i1, :, 1].mean():.2f}% | {100 * r[-1, :, 1].mean():.2f}% "
              f"| {r[b, :, 4].mean() * DT:.0f} |")

    hdr = ("| {} | Best m | Also within 2 SE | Casualties at best m | at m = 0 | at m = 1 "
           "| at m = 20 | Mean survivor exit (s) |\n|---|---|---|---|---|---|---|---|")
    print(hdr.format("Change"))
    for (k, v), r in zip(settings, res):
        J = r[..., 1] + learn_kf.CONFIG["lam"] * r[..., 4] / learn_kf.CONFIG["t_ref"]
        row("baseline" if k is None else f"{k} = {v:g}", r, J)
    print("\n" + hdr.format("Objective weight lam"))
    for lam in LAMS:
        r = res[0]
        row(f"lam = {lam:g}", r, r[..., 1] + lam * r[..., 4] / learn_kf.CONFIG["t_ref"])


if __name__ == "__main__":
    {"ca": ca, "fire": fire}[sys.argv[1]]()
