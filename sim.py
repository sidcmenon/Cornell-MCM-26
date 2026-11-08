"""
sim.py
Floor-field cellular automaton for evacuation (Kirchner-Schadschneider style).

Per time step (parallel update):
  1. each agent scores its 9 Moore moves (including staying put):
         log w = k_D * D[target] - k_S * d[target]
     where d = BFS step-distance to the nearest exit (diagonal step = 1) and D is
     the dynamic field. (distance_field.py instead counts a diagonal as sqrt 2 and is
     used only for the bailey_fields.png figure; the simulator and all saved results
     use this diagonal = 1 distance, which matches one diagonal move = one time step.)
     Blocked: walls, seat cells (unless seats_passable), cells
     occupied at the start of the step, cells that cannot reach an exit, and
     diagonal steps that cut a wall corner (unless cut_corners).
  2. each agent samples a move from the softmax of those scores
  3. conflicts (2+ agents targeting the same cell): with probability mu nobody
     moves; otherwise one winner moves (chosen with weight p, or uniformly)
  3b. door capacity: each door (connected group of exit cells) admits anyone this
     step with probability door_q, and then at most door_cap agents; extra agents
     stay put, survivors chosen with weight p or uniformly
  4. movers deposit +1 dynamic field on the cell they left
  5. agents that stepped onto an exit cell are removed
  6. dynamic field diffuses and decays

Seats: by default agents can leave a seat but cannot enter one, so nobody climbs
over rows. Seat cells get d = 1 + min(d of walkable neighbours) so a seated agent's
"stay" option has a finite score.

Exit times are recorded per agent; agents not out by step T get T + 1.
"""

from collections import deque

import numpy as np

from bailey_grid import PASSABLE, SEAT, EXIT
from base_dynamic_field import DynamicField, move_probs

# index 0 = stay; 1-4 orthogonal; 5-8 diagonal
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)])


def static_field(walk, exits, wall, cut_corners=False):
    """Multi-source BFS distance (in steps) from every exit cell over `walk` cells,
    Moore neighbourhood, every step costs 1. Unreachable cells get inf.
    With cut_corners=False a diagonal step is blocked if either adjacent orthogonal
    cell is a wall."""
    H, W = walk.shape
    d = np.full((H, W), np.inf)
    q = deque()
    for y, x in zip(*np.nonzero(exits & walk)):
        d[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        nd = d[y, x] + 1
        for dy, dx in MOVES[1:]:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W) or not walk[ny, nx] or d[ny, nx] <= nd:
                continue
            if dy and dx and not cut_corners and (wall[ny, x] or wall[y, nx]):
                continue
            d[ny, nx] = nd
            q.append((ny, nx))
    return d


def door_labels(exits):
    """Label 8-connected groups of exit cells: 0, 1, ... per door, -1 elsewhere."""
    H, W = exits.shape
    lab = np.full((H, W), -1)
    n = 0
    for y0, x0 in np.argwhere(exits):
        if lab[y0, x0] >= 0:
            continue
        lab[y0, x0] = n
        q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in MOVES[1:]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and exits[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = n
                    q.append((ny, nx))
        n += 1
    return lab


def build_static(grid, seats_passable=False, cut_corners=False):
    """Return (walk, wall, exits, d) for a bailey_grid-style grid."""
    passable = np.isin(grid, PASSABLE)
    wall = ~passable
    walk = passable if seats_passable else passable & (grid != SEAT)
    exits = grid == EXIT
    d = static_field(walk, exits, wall, cut_corners)
    if not seats_passable:                       # seats: one step more than best neighbour
        H, W = grid.shape
        seats = np.argwhere(grid == SEAT)
        for y, x in seats:
            best = np.inf
            for dy, dx in MOVES[1:]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W) or not walk[ny, nx]:
                    continue
                if dy and dx and not cut_corners and (wall[ny, x] or wall[y, nx]):
                    continue
                best = min(best, d[ny, nx])
            d[y, x] = best + 1
    return walk, wall, exits, d


class EvacSim:
    def __init__(self, grid, k_S, k_D, mu=0.0, alpha=0.3, delta=0.3,
                 cut_corners=False, seats_passable=False, winner="weighted",
                 include_seats=False, door_cap=1, door_q=0.6, static=None):
        """grid: 2D array of bailey_grid cell types.
        k_S, k_D: static / dynamic field weights (the parameters being learned).
        mu: friction. alpha, delta: dynamic-field diffusion and decay.
        winner: "weighted" (by move probability) or "uniform" conflict resolution.
        door_cap: max agents entering one door per step (None = unlimited).
        door_q: probability a door admits anyone in a given step, so mean door
            throughput is door_q * door_cap per step. With 0.432 m cells and a
            0.36 s step, door_q = 0.6, door_cap = 1 gives ~1.7 people/s through a
            1.3 m door (~1.3 people/m/s), in the measured bottleneck range.
        static: precomputed build_static(...) output (pass it when sweeping k)."""
        if winner not in ("weighted", "uniform"):
            raise ValueError("winner must be 'weighted' or 'uniform'")
        self.grid = grid
        walk, wall, exits, d = static or build_static(grid, seats_passable, cut_corners)
        self.exits, self.d = exits, d
        self.k_S, self.k_D, self.mu = k_S, k_D, mu
        self.winner = winner
        self.door_cap = door_cap
        if not 0.0 < door_q <= 1.0:
            raise ValueError("door_q must lie in (0, 1]")
        self.door_q = door_q
        self.door_id = door_labels(exits)
        self.field = DynamicField(grid, alpha, delta, include_seats)
        self._d_safe = np.where(np.isfinite(d), d, 0.0)

        # static validity of each move from each cell: (H, W, 9)
        H, W = grid.shape
        yy, xx = np.mgrid[0:H, 0:W]
        ok = np.zeros((H, W, 9), bool)
        for m, (dy, dx) in enumerate(MOVES[1:], start=1):
            ty, tx = yy + dy, xx + dx
            inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
            tyc, txc = np.clip(ty, 0, H - 1), np.clip(tx, 0, W - 1)
            v = inb & walk[tyc, txc] & np.isfinite(d[tyc, txc])
            if dy and dx and not cut_corners:
                v &= ~wall[tyc, xx] & ~wall[yy, txc]
            ok[..., m] = v
        ok[..., 0] = True                   # staying is always allowed
        self._ok = ok

    def run(self, start, T, rng):
        """start: (N, 2) int array of initial (y, x), all distinct.
        Returns exit_t: (N,) step at which each agent left (T + 1 if it never did)."""
        H, W = self.grid.shape
        pos = np.array(start, dtype=int)
        N = len(pos)
        exit_t = np.full(N, T + 1)
        alive = np.ones(N, bool)
        occ = np.zeros((H, W), bool)
        occ[pos[:, 0], pos[:, 1]] = True
        if occ.sum() != N:
            raise ValueError("start positions must be distinct")
        self.field.reset()

        on_exit = self.exits[pos[:, 0], pos[:, 1]]
        exit_t[on_exit] = 0
        alive[on_exit] = False
        occ[pos[on_exit, 0], pos[on_exit, 1]] = False

        for t in range(T):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            y, x = pos[idx, 0], pos[idx, 1]
            ty = np.clip(y[:, None] + MOVES[:, 0], 0, H - 1)
            tx = np.clip(x[:, None] + MOVES[:, 1], 0, W - 1)

            valid = self._ok[y, x].copy()
            occupied = occ[ty, tx]
            occupied[:, 0] = False
            valid &= ~occupied

            logw = self.field.log_weight(self.k_D)[ty, tx] - self.k_S * self._d_safe[ty, tx]
            p = move_probs(logw, ~valid)

            u = rng.random(idx.size)[:, None]
            choice = np.argmax(np.cumsum(p, axis=1) >= u, axis=1)   # rounding miss -> stay
            rows = np.arange(idx.size)
            p_chosen = p[rows, choice]

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
                            keep[members] = False          # door admits nobody this step
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
                alive[out] = False
                occ[pos[out, 0], pos[out, 1]] = False

            self.field.update()
        return exit_t


def objective(exit_t, T):
    """Mean capped exit time (lower is better); equivalent to maximizing the area
    under the egress curve. Also returns fraction evacuated and last exit time."""
    capped = np.minimum(exit_t, T + 1)
    done = exit_t <= T
    return {"mean_exit": capped.mean(),
            "frac_out": done.mean(),
            "last_exit": exit_t[done].max() if done.any() else np.inf}