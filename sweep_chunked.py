"""
sweep_chunked.py
Resumable version of learn.py's sweep: same grid, config and seeds, so results match
`python learn.py` exactly. Each call works for at most BUDGET seconds, saves finished
(point, replicate) results to a progress file, and picks up where it left off.

    python3 sweep_chunked.py coarse     # repeat until it prints DONE
    python3 sweep_chunked.py refine     # needs the finished coarse sweep
"""
import itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np

import learn
from learn import CONFIG, K_S, K_D, summarize, refine
from bailey_grid import build_grid
from sim import EvacSim, build_static, objective

BUDGET = 140.0


def _init(grid, seats, static, cfg):
    learn._S.update(grid=grid, seats=seats, static=static, cfg=cfg)


def one_rep(task):
    i, j, k_S, k_D, r = task
    S = learn._S
    cfg = S["cfg"]
    sim = EvacSim(S["grid"], k_S, k_D, mu=cfg["mu"], alpha=cfg["alpha"], delta=cfg["delta"],
                  cut_corners=cfg["cut_corners"], seats_passable=cfg["seats_passable"],
                  winner=cfg["winner"], include_seats=cfg["include_seats"],
                  door_cap=cfg["door_cap"], door_q=cfg["door_q"], static=S["static"])
    seats = S["seats"]
    n = int(round(cfg["fill"] * len(seats)))
    rng_start, rng_dyn = (np.random.default_rng(s) for s in
                          np.random.SeedSequence([cfg["base_seed"], r]).spawn(2))
    start = seats if n == len(seats) else seats[rng_start.choice(len(seats), n, replace=False)]
    res = objective(sim.run(start, cfg["T"], rng_dyn), cfg["T"])
    return i, j, r, [float(res["mean_exit"]), float(res["frac_out"]), float(res["last_exit"])]


def run(stage):
    t0 = time.time()
    if stage == "coarse":
        ks, kd = K_S, K_D
    else:
        c = np.load("results.npz")
        ks, kd = refine(np.unravel_index(np.argmin(c["mean"]), c["mean"].shape), c["k_S"], c["k_D"])
    prog = f"sweep_{stage}_progress.json"
    done = json.load(open(prog)) if os.path.exists(prog) else {}
    R = CONFIG["n_reps"]
    tasks = [(i, j, float(ks[i]), float(kd[j]), r)
             for i, j, r in itertools.product(range(len(ks)), range(len(kd)), range(R))
             if f"{i},{j},{r}" not in done]
    if tasks:
        g = build_grid()
        grid = g["grid"]
        seats = np.array([(a[0], a[1]) for a in g["agents"]])
        static = build_static(grid, CONFIG["seats_passable"], CONFIG["cut_corners"])
        # slow (low k_S) tasks first so the tail of each chunk is short
        tasks.sort(key=lambda t: t[2])
        with Pool(os.cpu_count(), initializer=_init, initargs=(grid, seats, static, CONFIG)) as pool:
            it = pool.imap_unordered(one_rep, tasks, chunksize=1)
            for i, j, r, v in it:
                done[f"{i},{j},{r}"] = v
                if time.time() - t0 > BUDGET:
                    pool.terminate()
                    break
        json.dump(done, open(prog, "w"))
    total = len(ks) * len(kd) * R
    print(f"{stage}: {len(done)}/{total} runs done ({time.time() - t0:.0f} s this call)")
    if len(done) < total:
        return
    res = np.zeros((len(ks), len(kd), R, 3))
    for key, v in done.items():
        i, j, r = map(int, key.split(","))
        res[i, j, r] = v
    mean, se, best = summarize(res, ks, kd)
    out = "results.npz" if stage == "coarse" else "results_refined.npz"
    np.savez(out, k_S=ks, k_D=kd, res=res, mean=mean, se=se)
    print("DONE, saved", out)


if __name__ == "__main__":
    run(sys.argv[1])
