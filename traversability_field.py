"""
traversability_field.py
Static field T: how easy each cell is to move through, by cell type (0 = impassable).
"""

import numpy as np

from bailey_grid import ACCESS, AISLE, EXIT, LEGROOM, SEAT, TYPE_NAMES, WALL

TRAVERSABILITY = {WALL: 0.0, AISLE: 1.0, LEGROOM: 0.8, SEAT: 0.1,
                  ACCESS: 0.6, EXIT: 1.0}


def traversability_field(grid, values=TRAVERSABILITY):
    lut = np.array([values[t] for t in range(len(TYPE_NAMES))])
    return lut[grid]