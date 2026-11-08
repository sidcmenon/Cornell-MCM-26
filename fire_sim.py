"""
fire_sim.py
Evacuation with fire and a heterogeneous crowd.

Each agent a has its own (k_S, k_D, k_F) and scores its 9 Moore moves by
    log w = k_D[a] * D - k_S[a] * d - k_F[a] * F
  d  BFS step-distance to the nearest exit, recomputed as fire and casualties block cells
  D  dynamic trail field (base_dynamic_field)
  F  fire proximity, max(0, ell - r) with r the step-distance to the nearest burning cell,
     so within ell cells each step away from the fire lowers F by exactly 1 (same unit as d)

Per time step:
  1. fire tick (every fire_every steps); agents on newly burning cells become casualties
     and stay in their cell as obstacles
  2. recompute d and F if the burning set or casualties changed
  3. agents choose moves; blocked: walls, seats (entering), occupied cells (agents or
     casualties), burning cells, wall corner-cuts
  4. conflicts with friction mu; door capacity door_q * door_cap per door per step
  5. deposits, exits, dynamic field update

Outcome per agent: 0 = escaped, 1 = burned, 2 = trapped (still inside at T).
"""

import numpy as np

from bailey_grid import PASSABLE, SEAT, EXIT
from base_dynamic_field import DynamicField, move_probs
from fire import Fire, shift
from sim import MOVES, door_labels

ESCAPED, BURNED, TRAPPED = 0, 1, 2


def static_moves(grid, cut_corners=False):
    """ok[m, y, x]: move m from (y, x) is geometrically allowed (target walkable,
    no wall corner-cut). Ignores fire, agents and casualties. ok[0] (stay) is all True."""
    passable = np.isin(grid, PASSABLE)
    wall = ~passable
    walk = passable & (grid != SEAT)
    H, W = grid.shape
    ok = np.zeros((9, H, W), bool)
    ok[0] = True
    for m, (dy, dx) in enumerate(MOVES[1:], start=1):
        v = shift(walk, dy, dx, False)
        if dy and dx and not cut_corners:
            v &= ~shift(wall, dy, 0, True) & ~shift(wall, 0, dx, True)
        ok[m] = v
    return ok, walk


def exit_distance(ok, walk, exits, open_cells, seats):
    """Step-distance to the nearest exit through walkable open cells (Bellman-Ford on
    the grid); seats get 1 + best walkable neighbour. inf where no exit is reachable."""
    allowed = walk & open_cells
    src = exits & allowed
    d = np.where(src, 0.0, np.inf)
    while True:
        nd = d.copy()
        for m, (dy, dx) in enumerate(MOVES[1:], start=1):
            nd = np.minimum(nd, np.where(ok[m], shift(d, dy, dx, np.inf) + 1, np.inf))
        nd = np.where(allowed, nd, np.inf)
        nd[src] = 0.0
        if np.array_equal(nd, d):
            break
        d = nd
    best = np.full(d.shape, np.inf)
    for m, (dy, dx) in enumerate(MOVES[1:], start=1):
        best = np.minimum(best, np.where(ok[m], shift(d, dy, dx, np.inf) + 1, np.inf))
    s = seats & open_cells
    d[s] = best[s]
    return d


def fire_proximity(burning, passable, ell):
    """F = max(0, ell - r), r = Moore step-distance from the nearest burning cell through
    non-wall cells (only needed out to ell)."""
    r = np.where(burning, 0.0, np.inf)
    for _ in range(ell):
        nr = r.copy()
        for dy, dx in MOVES[1:]:
            nr = np.minimum(nr, shift(r, dy, dx, np.inf) + 1)
        r = np.where(passable, nr, np.inf)
        r[burning] = 0.0
    return np.maximum(0.0, ell - r)


class FireEvacSim:
    def __init__(self, grid, mu=0.3, alpha=0.3, delta=0.3, door_cap=1, door_q=0.6,
                 winner="weighted", cut_corners=False, include_seats=False,
                 ell=8, fire_every=1, fire_kwargs=None):
        """ell: fire sensing range in cells. fire_every: steps per fire tick.
        fire_kwargs: passed to Fire (ignite_odds, burn_odds, difficulty, spread_scale...)."""
        self.grid = grid
        self.passable = np.isin(grid, PASSABLE)
        self.exits = grid == EXIT
        self.seats = grid == SEAT
        self.ok, self.walk = static_moves(grid, cut_corners)
        self.door_id = door_labels(self.exits)
        self.mu, self.door_cap, self.door_q, self.winner = mu, door_cap, door_q, winner
        self.ell, self.fire_every = ell, fire_every
        self.field = DynamicField(grid, alpha, delta, include_seats)
        self.fire = Fire(grid, **(fire_kwargs or {}))

    def run(self, start, k_S, k_D, k_F, ignition, T, rng, head_start=0, record=False):
        """start: (N, 2) initial cells. k_S, k_D, k_F: (N,) per-agent weights.
        ignition: iterable of (iy, ix) cells set alight at t = 0.
        head_start: fire ticks before anyone moves.
        Returns dict with exit_t (N,), outcome (N,), and optional history."""
        H, W = self.grid.shape
        pos = np.array(start, dtype=int)
        N = len(pos)
        k_S, k_D, k_F = (np.asarray(k, float) for k in (k_S, k_D, k_F))
        exit_t = np.full(N, T + 1)
        outcome = np.full(N, TRAPPED)
        alive = np.ones(N, bool)
        occ = np.zeros((H, W), bool)
        occ[pos[:, 0], pos[:, 1]] = True
        body = np.zeros((H, W), bool)
        self.field.reset()
        fire = self.fire
        fire.reset()
        fire.ignite(ignition)
        for _ in range(head_start):
            fire.step(rng)
        hist = [] if record else None

        def burn_check():
            idx = np.nonzero(alive)[0]
            hit = idx[fire.burning[pos[idx, 0], pos[idx, 1]]]
            if hit.size:
                alive[hit] = False
                outcome[hit] = BURNED
                body[pos[hit, 0], pos[hit, 1]] = True      # stays occupied
            return hit.size > 0

        def fields():
            open_cells = ~fire.burning & ~body
            d = exit_distance(self.ok, self.walk, self.exits, open_cells, self.seats)
            fin = np.isfinite(d)
            d_safe = np.where(fin, d, (d[fin].max() + 1) if fin.any() else 0.0)
            F = fire_proximity(fire.burning, self.passable, self.ell)
            return d_safe, F

        burn_check()
        d_safe, F = fields()

        for t in range(T):
            changed = False
            if t % self.fire_every == 0:
                changed = fire.step(rng)
            changed |= burn_check()
            if changed:
                d_safe, F = fields()

            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            y, x = pos[idx, 0], pos[idx, 1]
            ty = np.clip(y[:, None] + MOVES[:, 0], 0, H - 1)
            tx = np.clip(x[:, None] + MOVES[:, 1], 0, W - 1)

            valid = self.ok[:, y, x].T.copy()                  # (n, 9)
            blocked = occ[ty, tx] | fire.burning[ty, tx]
            blocked[:, 0] = False
            valid &= ~blocked

            logw = (k_D[idx, None] * self.field.D[ty, tx]
                    - k_S[idx, None] * d_safe[ty, tx]
                    - k_F[idx, None] * F[ty, tx])
            p = move_probs(logw, ~valid)

            u = rng.random(idx.size)[:, None]
            choice = np.argmax(np.cumsum(p, axis=1) >= u, axis=1)
            p_chosen = p[np.arange(idx.size), choice]

            mv = np.nonzero(choice != 0)[0]
            if mv.size:
                lin = ty[mv, choice[mv]] * W + tx[mv, choice[mv]]
                _, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
                win = np.ones(mv.size, bool)
                for c in np.nonzero(counts > 1)[0]:
                    members = np.nonzero(inv == c)[0]
                    win[members] = False
                    if rng.random() >= self.mu:
                        if self.winner == "weighted":
                            w = p_chosen[mv[members]]
                            k = rng.choice(members, p=w / w.sum())
                        else:
                            k = rng.choice(members)
                        win[k] = True
                mv = mv[win]

                if self.door_cap is not None and mv.size:
                    dn = self.door_id[ty[mv, choice[mv]], tx[mv, choice[mv]]]
                    keep = np.ones(mv.size, bool)
                    for door in np.unique(dn[dn >= 0]):
                        members = np.nonzero(dn == door)[0]
                        if self.door_q < 1.0 and rng.random() >= self.door_q:
                            keep[members] = False
                        elif members.size > self.door_cap:
                            keep[members] = False
                            if self.winner == "weighted":
                                w = p_chosen[mv[members]]
                                sel = rng.choice(members, self.door_cap, replace=False,
                                                 p=w / w.sum())
                            else:
                                sel = rng.choice(members, self.door_cap, replace=False)
                            keep[sel] = True
                    mv = mv[keep]

                a = idx[mv]
                oy, ox = pos[a, 0].copy(), pos[a, 1].copy()
                ny, nx = ty[mv, choice[mv]], tx[mv, choice[mv]]
                occ[oy, ox] = False
                occ[ny, nx] = True
                pos[a, 0], pos[a, 1] = ny, nx
                self.field.deposit(list(zip(oy, ox)))

                out = a[self.exits[ny, nx]]
                exit_t[out] = t + 1
                outcome[out] = ESCAPED
                alive[out] = False
                occ[pos[out, 0], pos[out, 1]] = False

            self.field.update()
            if record:
                hist.append((t, int(fire.burning.sum()), int(fire.burnt.sum()),
                             int((outcome == ESCAPED).sum()), int((outcome == BURNED).sum())))
        return dict(exit_t=exit_t, outcome=outcome, history=hist)


def fire_objective(res, T, lam=0.1, t_ref=150.0):
    """casualty fraction (burned + trapped) + lam * mean survivor exit time / t_ref."""
    o, et = res["outcome"], res["exit_t"]
    cas = np.mean(o != ESCAPED)
    surv = et[o == ESCAPED]
    mean_exit = surv.mean() if surv.size else float(T + 1)
    return dict(J=cas + lam * mean_exit / t_ref, casualty=cas,
                burned=np.mean(o == BURNED), trapped=np.mean(o == TRAPPED),
                mean_exit=mean_exit)