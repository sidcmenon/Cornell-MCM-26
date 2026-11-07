"""
dynamic_field.py
Dynamic field D: a trail that grows where people have walked, so agents follow the crowd.

Each time step:
  1. deposit   every agent that moves leaves +1 on the cell it just left
  2. diffuse   a fraction alpha of each cell's D spreads equally to its walkable
               4-neighbours (D is conserved by diffusion; nothing leaks into walls)
  3. decay     D <- (1 - delta) D

So a cell's value rises with the number of people who have recently passed through it
and fades once the traffic stops. Agents weight a move into cell j by exp(k_D * D_j).

This is the deterministic (mean-field) form of the Kirchner-Schadschneider dynamic
floor field; the original moves integer "bosons" stochastically with the same
rates alpha and delta.

Typical use inside the simulation loop:

    D = DynamicField(grid)
    for t in range(n_steps):
        ... choose moves using  exp(k_S * -S) * T**k_T * D.weight(k_D) ...
        D.deposit(cells_vacated_this_step)   # list of (iy, ix) or boolean mask
        D.update()
"""

import numpy as np

from bailey_grid import NEI4, PASSABLE, SEAT


class DynamicField:
    def __init__(self, grid, alpha=0.3, delta=0.3, include_seats=False):
        """alpha: diffusion fraction per step, 0..1.  delta: decay fraction per step, 0..1.
        include_seats: whether trail can spread onto seat cells (off by default, so the
        crowd signal stays in aisles and legroom)."""
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

    def deposit(self, cells, amount=1.0):
        """Add `amount` to each vacated cell. cells: boolean mask or iterable of (iy, ix).
        A cell listed twice gets two deposits."""
        if isinstance(cells, np.ndarray) and cells.dtype == bool:
            self.D[cells] += amount
        else:
            for iy, ix in cells:
                self.D[iy, ix] += amount

    def update(self):
        D, H, W = self.D, *self.D.shape
        spreads = self.walkable & (self._n > 0)
        share = np.where(spreads, self.alpha * D / np.maximum(self._n, 1), 0.0)
        new = D - np.where(spreads, self.alpha * D, 0.0)
        for dy, dx, m in self._nbr:
            give = np.where(m, share, 0.0)      # from (y, x) to (y+dy, x+dx)
            new[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] += \
                give[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        self.D = (1.0 - self.delta) * new

    def weight(self, k_D):
        """Multiplicative factor for move probabilities: exp(k_D * D)."""
        return np.exp(k_D * self.D)

    def reset(self):
        self.D[:] = 0.0