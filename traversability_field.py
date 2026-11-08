"""
traversability_field.py
Static field T: whether a cell can be walked through (1) or not (0), by cell type.
Access, aisle, legroom and exit cells are 1; seats and walls are 0.
"""

import numpy as np

from bailey_grid import ACCESS, AISLE, EXIT, LEGROOM, SEAT, TYPE_NAMES, WALL

TRAVERSABILITY = {WALL: 0, AISLE: 1, LEGROOM: 1, SEAT: 0, ACCESS: 1, EXIT: 1}


def traversability_field(grid, values=TRAVERSABILITY):
    lut = np.array([values[t] for t in range(len(TYPE_NAMES))], dtype=int)
    return lut[grid]
