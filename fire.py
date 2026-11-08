"""
fire.py
Fire spread on the 2D grid, adapted from Minecraft's (Java Edition) fire mechanics.

Every non-wall cell is fuel with the same ignite and burn odds; walls (stone) never burn.
State per cell: age (-1 = not burning, 0..15 = burning) and burnt (fuel used up).

Each fire tick, for every burning cell:
  1. spread   each flammable Moore neighbour ignites with probability
                  p = spread_scale * (ignite_odds + 40 + 7 * difficulty) / ((age + 30) * 100)
              (independent attempts from every burning neighbour)
  2. consume  the cell's own fuel is used up with probability burn_odds / burn_div;
              the fire there goes out and the cell becomes burnt (ash, passable, not fuel)
  3. age      age += 1 with probability 1/3, capped at 15
  4. burnout  age > 3 and no flammable neighbour -> goes out;
              age 15 -> goes out with probability 1/4
New fires start at age 0.

Differences from Minecraft, all forced by going 2D: Minecraft's fire is a block sitting
next to separate fuel blocks and reaches 4 blocks up; here the fire occupies the fuel
cell itself and spreads within the 3x3 neighbourhood. The spread and burn formulas
follow the Java source as best recalled; check the constants against FireBlock before
relying on absolute spread speeds. spread_scale rescales spread to the time step.
"""

import numpy as np

from bailey_grid import PASSABLE

NEI8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def shift(a, dy, dx, fill):
    """out[y, x] = a[y + dy, x + dx], filled with `fill` outside the grid."""
    H, W = a.shape
    out = np.full_like(a, fill)
    out[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
        a[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
    return out


class Fire:
    def __init__(self, grid, ignite_odds=30, burn_odds=60, difficulty=2,
                 burn_div=300, spread_scale=1.0):
        """Defaults: wool-like fuel (ignite 30, burn 60), normal difficulty."""
        self.fuel = np.isin(grid, PASSABLE)
        self.age = np.full(grid.shape, -1, dtype=int)
        self.burnt = np.zeros(grid.shape, bool)
        self.ignite_odds, self.burn_odds = ignite_odds, burn_odds
        self.difficulty, self.burn_div = difficulty, burn_div
        self.spread_scale = spread_scale

    @property
    def burning(self):
        return self.age >= 0

    def reset(self):
        self.age[:] = -1
        self.burnt[:] = False

    def ignite(self, cells):
        """cells: boolean mask or iterable of (iy, ix). Only unburnt fuel ignites."""
        mask = np.zeros(self.age.shape, bool)
        if isinstance(cells, np.ndarray) and cells.dtype == bool:
            mask |= cells
        else:
            for iy, ix in cells:
                mask[iy, ix] = True
        mask &= self.fuel & ~self.burnt & ~self.burning
        self.age[mask] = 0

    def step(self, rng):
        """One fire tick. Returns True if the burning set changed."""
        burning = self.burning
        if not burning.any():
            return False
        shape = self.age.shape
        flam = self.fuel & ~self.burnt & ~burning

        # 1. spread: P(cell stays unlit) = prod over burning neighbours of (1 - p)
        p = np.where(burning,
                     self.spread_scale * (self.ignite_odds + 40 + 7 * self.difficulty)
                     / ((self.age + 30) * 100.0), 0.0)
        p = np.clip(p, 0.0, 1.0)
        log_unlit = np.zeros(shape)
        has_flam = np.zeros(shape, bool)
        for dy, dx in NEI8:
            log_unlit += np.log1p(-np.minimum(shift(p, dy, dx, 0.0), 1 - 1e-12))
            has_flam |= shift(flam, dy, dx, False)
        new = flam & (rng.random(shape) < -np.expm1(log_unlit))

        # 2. consume own fuel
        consumed = burning & (rng.random(shape) < self.burn_odds / self.burn_div)

        # 3. age
        self.age[burning & (rng.random(shape) < 1 / 3)] += 1
        np.minimum(self.age, 15, out=self.age)

        # 4. burnout
        out = burning & (((self.age > 3) & ~has_flam) |
                         ((self.age >= 15) & (rng.random(shape) < 0.25)))

        ends = consumed | out
        self.age[ends] = -1
        self.burnt[ends] = True
        self.age[new] = 0
        return bool(new.any() or ends.any())