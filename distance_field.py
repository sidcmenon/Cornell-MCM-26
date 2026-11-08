"""
distance_field.py
Static field S: shortest walking distance to the nearest exit, in cells.

S is found by Dijkstra's algorithm outward from the exit cells (S = 0), moving to any
of the 8 neighbours (up/down/left/right cost 1, diagonals cost sqrt 2) and only through
traversable cells (T = 1: access, aisle, legroom, exit).

Seats are not traversable, so a path never passes through one, but the people sitting
in them still need a value: a seat gets the smallest S + step cost over its traversable
neighbours (the step out of the chair). Walls, and anything with no route out, are inf.

Because the barrier is a wall, orchestra paths automatically go through the access points.

Note: sim.py and fire_sim.py compute their own distance with every step (diagonal too)
costing 1, because one diagonal move takes one time step there. This file's field is
the true walking distance, used for build_fields.py / bailey_fields.png.
"""

import heapq

import numpy as np

from bailey_grid import EXIT, NEI8, SEAT
from traversability_field import traversability_field

STEP = [(dy, dx, float(np.hypot(dy, dx))) for dy, dx in NEI8]


def distance_field(g):
    grid = g["grid"]
    walk = traversability_field(grid) == 1
    H, W = grid.shape

    S = np.full(grid.shape, np.inf)
    S[grid == EXIT] = 0.0
    heap = [(0.0, int(y), int(x)) for y, x in np.argwhere(grid == EXIT)]
    while heap:
        s, y, x = heapq.heappop(heap)
        if s > S[y, x]:
            continue
        for dy, dx, c in STEP:
            v, u = y + dy, x + dx
            if 0 <= v < H and 0 <= u < W and walk[v, u] and s + c < S[v, u]:
                S[v, u] = s + c
                heapq.heappush(heap, (s + c, v, u))

    for y, x in np.argwhere(grid == SEAT):
        S[y, x] = min((S[y + dy, x + dx] + c for dy, dx, c in STEP
                       if 0 <= y + dy < H and 0 <= x + dx < W and walk[y + dy, x + dx]),
                      default=np.inf)
    return S
