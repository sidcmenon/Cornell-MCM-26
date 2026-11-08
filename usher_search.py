"""
usher_search.py
Where to put the ushers.

Uses the same fire scenarios and crowd as learn_kf.py (same seeds: ignition, crowd,
patron and fire streams), with the learned median fire weight m = 1, plus a per-patron
compliance w_a ~ Beta(kappa * w_bar, kappa * (1 - w_bar)) from its own random stream.
Objective: J = casualty fraction + lam * mean survivor exit / t_ref, as in learn_kf.py.

    python usher_search.py baseline          # no ushers: load on each door and access point
    python usher_search.py search [w_bar]    # greedy + local search, then final comparison
    python usher_search.py final [w_bar]     # redo only the final comparison from saved search
    python usher_search.py curve [w_bar]     # value of each additional usher (greedy order)

Search: candidate slots are walkable cells along the outer wall, thinned to `spacing`
cells apart; each slot may direct to its `door_options` nearest doors. Greedy adds the
best usher one at a time (screening replicates), then local search tries moving each
usher to the two slots on either side along the wall or changing its door, accepting a
move only if it beats the current configuration by more than one paired standard error.
Final: the optimized, greedy-only, no-usher and random configurations re-evaluated on
fresh seeds with more replicates.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import learn_kf
from bailey_grid import build_grid, EXIT, WALL
from fire_sim import fire_objective
from usher_sim import UsherFireSim, usher_candidates

KF = learn_kf.CONFIG
USHER = dict(
    n_ushers=6,
    R=10,               # usher range, cells of walking distance
    spacing=4,          # minimum distance between candidate slots, cells
    door_options=2,     # each usher may direct to one of its nearest doors
    kappa=4.0,          # Beta concentration of compliance
    m=1.0,              # median fire weight (learned)
    R_screen=12,        # replicates per configuration during the search
    R_final=100,        # replicates per configuration in the final comparison
    final_seed=4242,    # seeds for the final comparison (fresh, not used in search)
    max_passes=3,       # local search passes
    n_random=10,        # random configurations in the final comparison
)

_W = {}


def _init(cfg, ucfg, w_bar):
    g = build_grid()
    grid = g["grid"]
    seats = np.array([(a[0], a[1]) for a in g["agents"]])
    x_star, H = learn_kf.basin_from_refine()
    sim = UsherFireSim(grid, R=ucfg["R"], mu=cfg["mu"], alpha=cfg["alpha"],
                       delta=cfg["delta"], door_cap=cfg["door_cap"], door_q=cfg["door_q"],
                       ell=cfg["ell"], fire_tick_prob=learn_kf.fire_tick_prob(cfg),
                       fire_kwargs=cfg["fire_kwargs"], burn_steps=learn_kf.burn_steps(cfg))
    occupied = np.zeros(grid.shape, bool)
    occupied[seats[:, 0], seats[:, 1]] = True
    cand = np.argwhere((grid != WALL) & (grid != EXIT) & ~occupied)
    _W.update(cfg=cfg, ucfg=ucfg, w_bar=w_bar, grid=grid, seats=seats, x_star=x_star,
              H=H, sim=sim, ign_cand=cand)


def _one(task):
    cid, config, r, seed = task
    cfg, ucfg, w_bar, sim = _W["cfg"], _W["ucfg"], _W["w_bar"], _W["sim"]
    seats = _W["seats"]
    n = len(seats)
    rng_ign, rng_crowd, rng_dyn, rng_fire, rng_comp = (
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, r]).spawn(5))
    ign = [tuple(_W["ign_cand"][rng_ign.integers(len(_W["ign_cand"]))])]
    k_S, k_D, z_F = learn_kf.draw_crowd(n, _W["x_star"], _W["H"], cfg["c"],
                                        cfg["sigma_F"], rng_crowd)
    k_F = ucfg["m"] * np.exp(cfg["sigma_F"] * z_F)
    if w_bar <= 0:
        w = np.zeros(n)
    elif w_bar >= 1:
        w = np.ones(n)
    else:
        w = rng_comp.beta(ucfg["kappa"] * w_bar, ucfg["kappa"] * (1 - w_bar), n)
    res = sim.run(seats, k_S, k_D, k_F, ign, cfg["T"], rng_dyn, fire_rng=rng_fire,
                  ushers=[(tuple(c), d) for c, d in config], compliance=w,
                  head_start=cfg["head_start"])
    o = fire_objective(res, cfg["T"], cfg["lam"], cfg["t_ref"])
    doors = np.bincount(res["exit_door"][res["exit_door"] >= 0], minlength=sim.n_doors)
    n_acc = int(sim.access_id.max()) + 1
    acc = np.bincount(res["access_used"][res["access_used"] >= 0], minlength=n_acc)
    return cid, r, [o["J"], o["casualty"], o["burned"], o["trapped"], o["mean_exit"]], \
        doors.tolist(), acc.tolist()


def evaluate(pool, configs, R, seed):
    """Returns metrics (n_configs, R, 5), door counts (n_configs, R, n_doors),
    access counts (n_configs, R, n_access)."""
    tasks = [(i, cfg, r, seed) for i, cfg in enumerate(configs) for r in range(R)]
    out = pool.map(_one, tasks, chunksize=1)
    met = np.zeros((len(configs), R, 5))
    doors, acc = {}, {}
    for cid, r, m, d, a in out:
        met[cid, r] = m
        doors[cid, r], acc[cid, r] = d, a
    D = np.array([[doors[i, r] for r in range(R)] for i in range(len(configs))])
    A = np.array([[acc[i, r] for r in range(R)] for i in range(len(configs))])
    return met, D, A


def setup():
    """Candidate slots and their door options (computed in the main process)."""
    g = build_grid()
    sim = UsherFireSim(g["grid"], R=USHER["R"])
    slots = usher_candidates(g, sim.walk, USHER["spacing"])
    options = [sim.nearest_doors(s, USHER["door_options"]) for s in slots]
    return slots, options


def describe(config, slots):
    return [{"slot": slots.index(tuple(c)), "cell": list(c), "door": int(d)} for c, d in config]


def paired(a, b):
    """mean and standard error of a - b over replicates (common random numbers)."""
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def search(w_bar):
    slots, options = setup()
    n_cand = sum(len(o) for o in options)
    U = USHER
    print(f"{len(slots)} candidate slots, {n_cand} (slot, door) options; "
          f"greedy needs about {U['n_ushers'] * n_cand * U['R_screen']} runs")
    prog_path = f"usher_search_w{w_bar}.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    seed = KF["base_seed"]
    with Pool(os.cpu_count(), initializer=_init, initargs=(KF, U, w_bar)) as pool:
        chosen = [(tuple(c), d) for c, d in prog.get("greedy", [])]
        while len(chosen) < U["n_ushers"]:
            t0 = time.time()
            used = {c for c, _ in chosen}
            cands = [chosen + [(s, d)] for s, opts in zip(slots, options) if s not in used
                     for d in opts]
            met, _, _ = evaluate(pool, cands, U["R_screen"], seed)
            J = met[:, :, 0].mean(1)
            b = int(np.argmin(J))
            chosen = cands[b]
            print(f"usher {len(chosen)}: slot {slots.index(chosen[-1][0])} cell {chosen[-1][0]}"
                  f" door {chosen[-1][1]}  J={J[b]:.4f}  casualties={met[b, :, 1].mean():.4f}"
                  f"  ({time.time() - t0:.0f} s)")
            prog["greedy"] = [[list(c), d] for c, d in chosen]
            json.dump(prog, open(prog_path, "w"))

        current = [(tuple(c), d) for c, d in prog.get("local", prog["greedy"])]
        cur, _, _ = evaluate(pool, [current], U["R_screen"], seed)
        cur = cur[0]
        for p in range(prog.get("passes_done", 0), U["max_passes"]):
            improved = False
            for i in range(U["n_ushers"]):
                si = slots.index(current[i][0])
                others = {c for j, (c, _) in enumerate(current) if j != i}
                alts = []
                for ds in (-2, -1, 0, 1, 2):
                    s = slots[(si + ds) % len(slots)]
                    if s in others:
                        continue
                    for d in options[slots.index(s)]:
                        if (s, d) != current[i]:
                            alts.append(current[:i] + [(s, d)] + current[i + 1:])
                if not alts:
                    continue
                met, _, _ = evaluate(pool, alts, U["R_screen"], seed)
                diffs = [paired(met[k, :, 0], cur[:, 0]) for k in range(len(alts))]
                k = int(np.argmin([m for m, _ in diffs]))
                m, se = diffs[k]
                if m < -se:
                    current, cur = alts[k], met[k]
                    improved = True
                    print(f"pass {p + 1}, usher {i + 1}: moved to slot "
                          f"{slots.index(current[i][0])} door {current[i][1]}  "
                          f"dJ={m:.4f} +/- {se:.4f}")
            prog["local"] = [[list(c), d] for c, d in current]
            prog["passes_done"] = p + 1
            json.dump(prog, open(prog_path, "w"))
            if not improved:
                break
    print("search done:", describe(current, slots))
    final(w_bar)


def final(w_bar):
    slots, options = setup()
    U = USHER
    prog = json.load(open(f"usher_search_w{w_bar}.json"))
    optimized = [(tuple(c), d) for c, d in prog.get("local", prog["greedy"])]
    greedy = [(tuple(c), d) for c, d in prog["greedy"]]
    rng = np.random.default_rng(U["final_seed"])
    randoms = []
    for _ in range(U["n_random"]):
        pick = rng.choice(len(slots), U["n_ushers"], replace=False)
        randoms.append([(slots[i], int(rng.choice(options[i]))) for i in pick])
    names = ["none", "optimized", "greedy"] + [f"random{i}" for i in range(len(randoms))]
    configs = [[], optimized, greedy] + randoms
    with Pool(os.cpu_count(), initializer=_init, initargs=(KF, U, w_bar)) as pool:
        met, D, A = evaluate(pool, configs, U["R_final"], U["final_seed"])
    print(f"\nfinal comparison, w_bar={w_bar}, {U['R_final']} fresh replicates")
    print(f"{'config':>10} {'J':>16} {'casualty':>16} {'mean_exit':>16} {'dJ vs none':>18}")
    for i, nm in enumerate(names):
        J = met[i, :, 0]
        dm, dse = paired(J, met[0, :, 0])
        print(f"{nm:>10} {J.mean():9.4f} ±{J.std(ddof=1)/np.sqrt(len(J)):.4f}"
              f" {met[i, :, 1].mean():9.4f} ±{met[i, :, 1].std(ddof=1)/np.sqrt(len(J)):.4f}"
              f" {met[i, :, 4].mean():9.2f} ±{met[i, :, 4].std(ddof=1)/np.sqrt(len(J)):.2f}"
              f" {dm:+9.4f} ±{dse:.4f}")
    rJ = met[3:, :, 0].mean(1)
    print(f"random configurations: mean J {rJ.mean():.4f}, best {rJ.min():.4f}, worst {rJ.max():.4f}")
    print("optimized ushers:", describe(optimized, slots))
    print("door use (mean patrons per door), none vs optimized:")
    print("  none     ", np.round(D[0].mean(0), 1).tolist())
    print("  optimized", np.round(D[1].mean(0), 1).tolist())
    print("access point use, none vs optimized:")
    print("  none     ", np.round(A[0].mean(0), 1).tolist())
    print("  optimized", np.round(A[1].mean(0), 1).tolist())
    np.savez(f"results_ushers_w{w_bar}.npz", names=np.array(names), met=met, doors=D,
             access=A, optimized=np.array([list(c) + [d] for c, d in optimized]),
             slots=np.array(slots))


def curve(w_bar):
    """Evaluate the first k greedy ushers, k = 0..n, on fresh seeds: the marginal value of
    each additional crowd manager."""
    slots, _ = setup()
    U = USHER
    prog = json.load(open(f"usher_search_w{w_bar}.json"))
    greedy = [(tuple(c), d) for c, d in prog["greedy"]]
    configs = [greedy[:k] for k in range(len(greedy) + 1)]
    with Pool(os.cpu_count(), initializer=_init, initargs=(KF, U, w_bar)) as pool:
        met, _, _ = evaluate(pool, configs, U["R_final"], U["final_seed"])
    print(f"\nvalue of each usher (greedy order), w_bar={w_bar}, {U['R_final']} fresh replicates")
    print(f"{'ushers':>6} {'J':>16} {'casualty':>16} {'mean_exit':>14} {'dJ vs one fewer':>20}")
    for k in range(len(configs)):
        J = met[k, :, 0]
        line = (f"{k:6d} {J.mean():9.4f} ±{J.std(ddof=1)/np.sqrt(len(J)):.4f}"
                f" {met[k, :, 1].mean():9.4f} ±{met[k, :, 1].std(ddof=1)/np.sqrt(len(J)):.4f}"
                f" {met[k, :, 4].mean():9.2f}")
        if k:
            dm, dse = paired(J, met[k - 1, :, 0])
            line += f" {dm:+10.4f} ±{dse:.4f}"
        print(line)
    np.savez(f"results_usher_curve_w{w_bar}.npz", met=met)


def baseline(R=50):
    slots, options = setup()
    print(f"{len(slots)} candidate usher slots")
    with Pool(os.cpu_count(), initializer=_init, initargs=(KF, USHER, 0.0)) as pool:
        met, D, A = evaluate(pool, [[]], R, KF["base_seed"])
    print(f"no ushers, fire scenarios, {R} replicates: J={met[0, :, 0].mean():.4f} "
          f"casualties={met[0, :, 1].mean():.4f} mean exit={met[0, :, 4].mean():.1f}")
    d, a = D[0].mean(0), A[0].mean(0)
    print("patrons per door (mean):        ", np.round(d, 1).tolist(),
          f" max/min = {d.max() / max(d.min(), 1e-9):.2f}")
    print("patrons per access point (mean):", np.round(a, 1).tolist(),
          f" max/min = {a.max() / max(a.min(), 1e-9):.2f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    w_bar = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    if mode == "baseline":
        baseline()
    elif mode == "search":
        search(w_bar)
    elif mode == "final":
        final(w_bar)
    elif mode == "curve":
        curve(w_bar)
    else:
        raise SystemExit("mode must be baseline, search or final")
