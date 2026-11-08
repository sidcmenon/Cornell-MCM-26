"""
usher_sim.py
Fire evacuation with ushers (extends fire_sim.FireEvacSim).

Ushers stand at walkable cells next to the outer wall (they occupy no cell). Usher u
directs patrons to an assigned exit door E_u. Its influence region is the cells within
R steps of it (walking distance through walkable cells, seats included); where regions
overlap the nearer usher wins. A patron becomes instructed the first time it stands in
a region and keeps that instruction for the rest of the run. An instructed patron a with
compliance w_a scores moves with the blended distance
    SD_a = (1 - w_a) * SD + w_a * SD_door
in place of SD, where SD_door is the step distance to the assigned door only (recomputed,
like SD, whenever fire or casualties block cells). Everything else is as in fire_sim.

run() also returns, per patron, the door it left through and the first access point it
used, for load diagnostics.
"""

import numpy as np

from bailey_grid import ACCESS, EXIT, WALL
from fire_sim import (FireEvacSim, exit_distance, fire_proximity, ESCAPED, BURNED,
                      TRAPPED, fire_objective)
from base_dynamic_field import move_probs
from sim import MOVES, door_labels


def _safe(d):
    fin = np.isfinite(d)
    return np.where(fin, d, (d[fin].max() + 1) if fin.any() else 0.0)


def usher_candidates(g, walk, spacing=4):
    """Candidate usher cells: walkable non-seat, non-exit, non-access cells touching an
    outer wall cell (a wall cell that is not part of the orchestra/mezzanine barrier),
    thinned so kept cells are at least `spacing` cells apart (Chebyshev), ordered by
    angle around the hall so neighbouring slots are adjacent in the list."""
    grid = g["grid"]
    H, W = grid.shape
    wall = grid == WALL

    def touches(m):
        out = np.zeros_like(m)
        for dy, dx in MOVES[1:]:
            s = np.zeros_like(m)
            s[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
                m[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
            out |= s
        return out

    barrier = wall & touches(g["orch"]) & touches(g["mezz"])
    outer = wall & ~barrier
    cand = walk & touches(outer) & (grid != ACCESS) & (grid != EXIT)
    cells = np.argwhere(cand)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    order = np.argsort(np.arctan2(cells[:, 1] - cx, cells[:, 0] - cy))
    kept = []
    for c in cells[order]:
        if all(max(abs(c[0] - k[0]), abs(c[1] - k[1])) >= spacing for k in kept):
            kept.append(tuple(int(v) for v in c))
    return kept


class UsherFireSim(FireEvacSim):
    def __init__(self, grid, R=10, **kw):
        super().__init__(grid, **kw)
        self.R = R
        self.n_doors = int(self.door_id.max()) + 1
        self.access_id = door_labels(grid == ACCESS)
        everything = np.ones(grid.shape, bool)
        # static (no fire) distance to each door, for choosing ushers' candidate doors
        self.door_dist_static = np.array([
            exit_distance(self.ok, self.walk, self.door_id == k, everything, self.seats)
            for k in range(self.n_doors)])

    def nearest_doors(self, cell, n=2):
        d = self.door_dist_static[:, cell[0], cell[1]]
        return [int(k) for k in np.argsort(d)[:n]]

    def regions(self, ushers):
        """region[y, x] = index of the usher whose region contains the cell, or -1."""
        everything = np.ones(self.grid.shape, bool)
        best = np.full(self.grid.shape, np.inf)
        region = np.full(self.grid.shape, -1)
        for u, (cell, _) in enumerate(ushers):
            src = np.zeros(self.grid.shape, bool)
            src[cell] = True
            d = exit_distance(self.ok, self.walk, src, everything, self.seats)
            take = (d <= self.R) & (d < best)
            best[take] = d[take]
            region[take] = u
        return region

    def run(self, start, k_S, k_D, k_F, ignition, T, rng, fire_rng=None,
            ushers=(), compliance=None, head_start=0, start_delay=None):
        """ushers: sequence of ((iy, ix), door) pairs. compliance: (N,) in [0, 1].
        start_delay: optional (N,) steps each patron waits in place before moving
        (pre-movement time); None means everyone moves from step 0.
        Returns dict with exit_t, outcome, exit_door (-1 if not escaped) and
        access_used (-1 if never on an access cell)."""
        if fire_rng is None:
            fire_rng = rng
        H, W = self.grid.shape
        pos = np.array(start, dtype=int)
        N = len(pos)
        k_S, k_D, k_F = (np.asarray(k, float) for k in (k_S, k_D, k_F))
        w = np.zeros(N) if compliance is None else np.asarray(compliance, float)
        exit_t = np.full(N, T + 1)
        outcome = np.full(N, TRAPPED)
        exit_door = np.full(N, -1)
        access_used = np.full(N, -1)
        alive = np.ones(N, bool)
        occ = np.zeros((H, W), bool)
        occ[pos[:, 0], pos[:, 1]] = True
        body = np.zeros((H, W), bool)
        exposure = np.zeros(N, int)
        self.field.reset()
        fire = self.fire
        fire.reset()
        fire.ignite(ignition)
        for _ in range(head_start):
            fire.step(fire_rng)

        ushers = list(ushers)
        doors_used = sorted({d for _, d in ushers})
        field_of_door = {d: i for i, d in enumerate(doors_used)}
        usher_field = np.array([field_of_door[d] for _, d in ushers], dtype=int)
        region = self.regions(ushers) if ushers else np.full((H, W), -1)
        instr = np.full(N, -1)        # index into doors_used, or -1

        def burn_check():
            idx = np.nonzero(alive)[0]
            in_fire = idx[fire.burning[pos[idx, 0], pos[idx, 1]]]
            exposure[in_fire] += 1
            hit = in_fire[exposure[in_fire] >= self.burn_steps]
            if hit.size:
                alive[hit] = False
                outcome[hit] = BURNED
                body[pos[hit, 0], pos[hit, 1]] = True
            return hit.size > 0

        def fields():
            open_cells = ~fire.burning & ~body
            d = _safe(exit_distance(self.ok, self.walk, self.exits, open_cells, self.seats))
            dd = np.array([_safe(exit_distance(self.ok, self.walk, self.door_id == k,
                                               open_cells, self.seats))
                           for k in doors_used]) if doors_used else np.zeros((0, H, W))
            F = fire_proximity(fire.burning, self.passable, self.ell)
            return d, dd, F

        burn_check()
        d_safe, dd_safe, F = fields()

        for t in range(T):
            changed = False
            if fire_rng.random() < self.fire_tick_prob:
                changed = fire.step(fire_rng)
            changed |= burn_check()
            if changed:
                d_safe, dd_safe, F = fields()

            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            y, x = pos[idx, 0], pos[idx, 1]

            if ushers:
                new = idx[(instr[idx] < 0) & (region[y, x] >= 0)]
                if new.size:
                    instr[new] = usher_field[region[pos[new, 0], pos[new, 1]]]
            acc = self.access_id[y, x]
            first = (access_used[idx] < 0) & (acc >= 0)
            access_used[idx[first]] = acc[first]

            ty = np.clip(y[:, None] + MOVES[:, 0], 0, H - 1)
            tx = np.clip(x[:, None] + MOVES[:, 1], 0, W - 1)
            valid = self.ok[:, y, x].T.copy()
            blocked = occ[ty, tx] | fire.burning[ty, tx]
            blocked[:, 0] = False
            valid &= ~blocked

            sd = d_safe[ty, tx]
            ins = np.nonzero(instr[idx] >= 0)[0]
            if ins.size:
                a = idx[ins]
                wa = w[a][:, None]
                sd = sd.copy()
                sd[ins] = (1 - wa) * sd[ins] + wa * dd_safe[instr[a][:, None], ty[ins], tx[ins]]

            logw = (k_D[idx, None] * self.field.D[ty, tx]
                    - k_S[idx, None] * sd
                    - k_F[idx, None] * F[ty, tx])
            p = move_probs(logw, ~valid)

            u = rng.random(idx.size)[:, None]
            choice = np.argmax(np.cumsum(p, axis=1) >= u, axis=1)
            if start_delay is not None:
                choice[t < start_delay[idx]] = 0          # still getting up: stay put
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
                            ww = p_chosen[mv[members]]
                            k = rng.choice(members, p=ww / ww.sum())
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
                                ww = p_chosen[mv[members]]
                                sel = rng.choice(members, self.door_cap, replace=False,
                                                 p=ww / ww.sum())
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

                at_exit = self.exits[ny, nx]
                out = a[at_exit]
                exit_t[out] = t + 1
                outcome[out] = ESCAPED
                exit_door[out] = self.door_id[ny[at_exit], nx[at_exit]]
                alive[out] = False
                occ[pos[out, 0], pos[out, 1]] = False

            self.field.update()
        return dict(exit_t=exit_t, outcome=outcome, exit_door=exit_door,
                    access_used=access_used)
