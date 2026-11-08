"""
calibrate_spread.py
Calibrate the fire in two separate parts.

  spread_scale s     controls the fire's shape: whether an ignition takes hold and how
                     completely the front burns the hall. Low s sits near a percolation
                     threshold (most ignitions fizzle, fronts leave large unburnt patches),
                     so s cannot be lowered to slow the fire down.
  fire_tick_prob q   the fire updates in a step with probability q (Minecraft-style random
                     tick). This rescales the fire's time without changing its shape.

For each s in S_VALUES, with the fire ticking every step, we record per run the coverage
C(t) (fraction of non-wall cells that have caught fire) and its final value C_final.
  established   C_final >= MIN_COVER
  engulf ticks  first tick with C >= FRAC * C_final
Then q = (median engulf ticks over established fires) * DT / T_TARGET, which makes the
median engulf time T_TARGET seconds when the fire ticks with probability q per step.

    python calibrate_s.py
Pick an s with high establishment and coverage, and put s and its q into learn_kf.py.
"""

import numpy as np

from bailey_grid import build_grid, WALL, EXIT
from fire import Fire

DT = 0.36            # seconds per time step (match the paper's conversion)
T_TARGET = 180.0     # seconds to engulf the hall
FRAC = 0.95          # engulfed = reached 95% of the area the fire will eventually reach
MIN_COVER = 0.5      # fires covering less than this fizzled
S_VALUES = [4.0, 6.0, 8.0, 12.0, 16.0]
N_RUNS = 40
BASE_SEED = 11
MAX_TICKS = 2000

g = build_grid()
grid = g["grid"]
occupied = np.zeros(grid.shape, bool)
for a in g["agents"]:
    occupied[a[0], a[1]] = True
cand = np.argwhere((grid != WALL) & (grid != EXIT) & ~occupied)
n_fuel = int((grid != WALL).sum())
ignitions = [tuple(cand[np.random.default_rng([BASE_SEED, k]).integers(len(cand))])
             for k in range(N_RUNS)]


def run_once(s, k):
    rng = np.random.default_rng([BASE_SEED, k, 1])
    f = Fire(grid, spread_scale=s)
    f.ignite([ignitions[k]])
    cover = []
    for _ in range(MAX_TICKS):
        f.step(rng)
        cover.append((f.burning | f.burnt).sum() / n_fuel)
        if not f.burning.any():
            break
    return np.array(cover)


print(f"{'s':>6} {'established':>12} {'coverage':>9} {'engulf ticks':>13} "
      f"{'q':>7} {'IQR of engulf time (s) at q':>30}")
for s in S_VALUES:
    ticks, finals = [], []
    for k in range(N_RUNS):
        c = run_once(s, k)
        finals.append(c[-1])
        if c[-1] >= MIN_COVER:
            ticks.append(int(np.argmax(c >= FRAC * c[-1])) + 1)
    finals = np.array(finals)
    est = finals >= MIN_COVER
    if not ticks:
        print(f"{s:6.1f} {est.mean():12.0%}   (no established fires)")
        continue
    n = np.median(ticks)
    q = n * DT / T_TARGET
    lo, hi = np.percentile(ticks, [25, 75]) * DT / q
    flag = "" if q <= 1 else "   q > 1: fire too slow at this s even ticking every step"
    print(f"{s:6.1f} {est.mean():12.0%} {np.median(finals[est]):9.2f} {n:13.0f} "
          f"{q:7.3f} {lo:14.0f} - {hi:.0f}{flag}")