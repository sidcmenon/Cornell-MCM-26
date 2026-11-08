"""
base_dynamic_field.py
Dynamic field D: a trail that grows where people have walked, so agents follow the crowd.

Each time step:
  1. deposit   every agent that moves leaves +1 on the cell it just left
               (only on walkable cells; deposits elsewhere are discarded)
  2. diffuse   a fraction alpha of each cell's D spreads equally to its walkable
               4-neighbours (D is conserved by diffusion; nothing leaks into walls)
  3. decay     D <- (1 - delta) D

Parameter convention follows Burstedde et al. / Kirchner-Schadschneider:
alpha = diffusion, delta = decay. (Yang, Wang & Qin 2015 swap these labels.)
Diffusion and decay are linear and decay is a scalar, so their order does not matter.

This is the deterministic (mean-field) form of the Kirchner-Schadschneider dynamic
floor field; the original moves integer "bosons" stochastically with the same
rates alpha and delta.

Move probabilities are formed in log space to avoid overflow:

    logw = D.log_weight(k_D)[nbrs] - k_S * S[nbrs]   # S = BFS distance to exit
    p = move_probs(logw, blocked)

Typical use inside the simulation loop:

    field = DynamicField(grid)
    for t in range(n_steps):
        ... choose moves with move_probs(field.log_weight(k_D)[...] - k_S * S[...], ...)
        field.deposit(cells_vacated_this_step)   # list of (iy, ix) or boolean mask
        field.update()
"""

import numpy as np

from bailey_grid import NEI4, PASSABLE, SEAT


class DynamicField:
    def __init__(self, grid, alpha=0.3, delta=0.3, include_seats=False):
        """alpha: diffusion fraction per step, 0..1.  delta: decay fraction per step, 0..1.
        include_seats: whether seat cells carry trail (off by default, so the crowd
        signal stays in aisles and legroom). Cells that do not carry trail always
        have D = 0, i.e. a neutral log-weight of 0."""
        if not (0.0 <= alpha <= 1.0 and 0.0 <= delta <= 1.0):
            raise ValueError("alpha and delta must lie in [0, 1]")
        types = [t for t in PASSABLE if include_seats or t != SEAT]
        self.walkable = np.isin(grid, types)
        self.alpha = alpha
        self.delta = delta
        self.D = np.zeros(grid.shape)

        # number of walkable 4-neighbours of each cell
        H, W = grid.shape
        self._nbr = []
        n = np.zeros(grid.shape)
        for dy, dx in NEI4:
            m = np.zeros(grid.shape, bool)
            m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
                self.walkable[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
            self._nbr.append((dy, dx, m))       # m[y, x]: (y+dy, x+dx) is walkable
            n += m
        self._n = n
        self._spreads = self.walkable & (n > 0)

    def deposit(self, cells, amount=1.0):
        """Add `amount` to each vacated walkable cell; non-walkable cells are skipped.
        cells: boolean mask or iterable of (iy, ix). A cell listed twice gets two deposits."""
        if isinstance(cells, np.ndarray) and cells.dtype == bool:
            self.D[cells & self.walkable] += amount
        else:
            for iy, ix in cells:
                if self.walkable[iy, ix]:
                    self.D[iy, ix] += amount

    def update(self):
        D, H, W = self.D, *self.D.shape
        share = np.where(self._spreads, self.alpha * D / np.maximum(self._n, 1), 0.0)
        new = D - np.where(self._spreads, self.alpha * D, 0.0)
        for dy, dx, m in self._nbr:
            give = np.where(m, share, 0.0)      # from (y, x) to (y+dy, x+dx)
            new[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] += \
                give[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        self.D = (1.0 - self.delta) * new

    def log_weight(self, k_D):
        """Log of the multiplicative move factor: k_D * D. Combine with other log terms
        and exponentiate once via move_probs (never call np.exp on this directly)."""
        return k_D * self.D

    def reset(self):
        self.D[:] = 0.0


def move_probs(logw, blocked=None):
    """Normalized move probabilities from log-weights over candidate cells.

    logw: array (..., K) of log-weights, e.g. k_D * D - k_S * S for the K = 9 Moore moves.
    blocked: optional boolean array of the same shape; True means the move is
             impossible (wall, occupied, corner-cut) and gets probability 0.
    Rows with every move blocked return all zeros (agent waits).
    """
    logw = np.array(logw, dtype=float, copy=True)
    if blocked is not None:
        logw[blocked] = -np.inf
    mx = logw.max(axis=-1, keepdims=True)
    allblocked = ~np.isfinite(mx)
    mx = np.where(allblocked, 0.0, mx)
    w = np.exp(logw - mx)
    s = w.sum(axis=-1, keepdims=True)
    return np.where(allblocked, 0.0, w / np.where(s > 0, s, 1.0))