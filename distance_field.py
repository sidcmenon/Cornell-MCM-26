"""
distance_field.py
Static field S: straight-line distance to the exits, in cells.

    Mezzanine and access cells:  S(x) = min_e |x - x_e| / CELL
    Orchestra cells:             S(x) = min_q ( |x - x_q| / CELL + S(x_q) )

e runs over exit cells, q over access cells. Routing the orchestra through the access
points keeps S decreasing all the way out, so no walkable cell is a local minimum in
front of the barrier. Straight lines ignore walls; wall cells get inf.
"""

import numpy as np

from bailey_grid import ACCESS, CELL, EXIT, WALL


def distance_field(g):
    grid, xs, ys, orch = g["grid"], g["xs"], g["ys"], g["orch"]
    X, Y = np.meshgrid(xs, ys)

    S = np.full(grid.shape, np.inf)
    ey, ex = np.nonzero(grid == EXIT)
    for px, py in zip(xs[ex], ys[ey]):
        S = np.minimum(S, np.hypot(X - px, Y - py) / CELL)

    S_orch = np.full(grid.shape, np.inf)
    ay, ax = np.nonzero(grid == ACCESS)
    for iy, ix in zip(ay, ax):
        S_orch = np.minimum(S_orch, np.hypot(X - xs[ix], Y - ys[iy]) / CELL + S[iy, ix])
    S[orch] = S_orch[orch]

    S[grid == WALL] = np.inf
    return S