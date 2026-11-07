"""
building.py
First floor of Bailey Hall out to the street: the auditorium from ../bailey_grid.py plus
the semicircular corridor, the foyer, and the doors to the outside.

    seats -> (orchestra: 6 access points) -> 9 mezzanine doors -> corridor
          -> 3 doors to the foyer -> 5 front doors -> outside
          -> 4 side doors to the portico                -> outside

Every opening (access point or door) is listed in `openings` with its real width,
because the model caps the flow through each one (see continuum.py).

All corridor, foyer and door dimensions below are rough estimates read off the
first-floor plan (scale from the 31 m auditorium width); replace with measurements.

funnels=True adds V-shaped barriers on the orchestra side of each access point.
"""

import numpy as np
from scipy import ndimage

from bailey_grid import (ACCESS, ACCESS_WIDTH, AISLE, DOOR_WIDTH, EXIT, HALF_WIDTH,
                         LOGE_CENTER_Y, OUTER_R, STAGE_Y, WALL, build_grid)

DOOR, OUT = 6, 7               # new cell types (bailey_grid uses 0..5)

CORR_W = 2.7                   # m, corridor width
FOYER_DEPTH = 6.0              # m
FOYER_HALF = 10.5              # m, half the foyer's width
FOYER_DOOR_X, FOYER_DOOR_W = [-6.0, 0.0, 6.0], 1.8      # corridor -> foyer
FRONT_DOOR_X, FRONT_DOOR_W = [-8.0, -4.0, 0.0, 4.0, 8.0], 1.5   # foyer -> outside
PORTICO_PHI, PORTICO_DOOR_W = [-75.0, -40.0, 40.0, 75.0], 1.2   # corridor -> portico

FUNNEL_LEN = 1.5               # m, length of each funnel wall
FUNNEL_SPREAD = 35.0           # deg, how far each wall opens away from the opening's axis


def build_building(funnels=False):
    g = build_grid()
    cell = g["cell"]

    # pad the auditorium grid with wall so the corridor and foyer fit
    px = int(np.ceil((CORR_W + 3 * cell + HALF_WIDTH) / cell)) - len(g["xs"]) // 2 + 2
    y_back = LOGE_CENTER_Y + OUTER_R + cell + CORR_W + cell + FOYER_DEPTH
    py = int(np.ceil((y_back + 3 * cell - g["ys"][-1]) / cell))
    pad = ((0, py), (px, px))
    grid = np.pad(g["grid"], pad, constant_values=WALL)
    occupied = np.pad(g["occupied"], pad)
    orch = np.pad(g["orch"], pad)
    xs = np.concatenate([g["xs"][0] - cell * np.arange(px, 0, -1), g["xs"],
                         g["xs"][-1] + cell * np.arange(1, px + 1)])
    ys = np.concatenate([g["ys"], g["ys"][-1] + cell * np.arange(1, py + 1)])
    X, Y = np.meshgrid(xs, ys)
    phi = np.degrees(np.arctan2(X, Y - LOGE_CENTER_Y))
    r = np.hypot(X, Y - LOGE_CENTER_Y)

    hall = (Y >= STAGE_Y) & (np.abs(X) < HALF_WIDTH) & (r < OUTER_R)
    d = ndimage.distance_transform_edt(~hall) * cell        # m from the auditorium
    near_hall = ndimage.binary_dilation(hall, np.ones((3, 3)))
    corridor = ~near_hall & (d <= cell + CORR_W) & (Y >= STAGE_Y)
    outer_wall = (d > cell + CORR_W) & (d <= 2 * cell + CORR_W)
    foyer = (np.abs(X) < FOYER_HALF) & (d > 2 * cell + CORR_W) & (Y <= y_back)

    openings = []                                          # (mask, width in m)

    def add(mask, width, kind=DOOR):
        grid[mask] = kind
        openings.append((mask, width))

    for k in range(1, 10):                                 # 9 mezzanine doors
        lab, _ = ndimage.label(g["grid"] == EXIT, np.ones((3, 3)))
        add(np.pad(lab == k, pad), DOOR_WIDTH)
    lab, n = ndimage.label(g["grid"] == ACCESS, np.ones((3, 3)))
    access = [np.pad(lab == k, pad) for k in range(1, n + 1)]
    for m in access:                                       # 6 access points
        add(m, ACCESS_WIDTH, ACCESS)

    grid[corridor | foyer] = AISLE
    for x0 in FOYER_DOOR_X:
        add(outer_wall & (np.abs(X - x0) <= FOYER_DOOR_W / 2) & (np.abs(X) < FOYER_HALF),
            FOYER_DOOR_W)
    for x0 in FRONT_DOOR_X:
        across = np.abs(X - x0) <= FRONT_DOOR_W / 2
        add(across & (Y > y_back) & (Y <= y_back + cell), FRONT_DOOR_W)
        grid[across & (Y > y_back + cell) & (Y <= y_back + 2 * cell)] = OUT
    for p0 in PORTICO_PHI:
        across = np.abs(np.radians(phi - p0)) * (OUTER_R + CORR_W) <= PORTICO_DOOR_W / 2
        add(across & outer_wall & ~foyer, PORTICO_DOOR_W)
        grid[across & (d > 2 * cell + CORR_W) & (d <= 3 * cell + CORR_W)] = OUT

    if funnels:
        for m in access:
            ax_, ay_ = X[m].mean(), Y[m].mean()
            n_ = -np.array([ax_, ay_ - LOGE_CENTER_Y]); n_ /= np.hypot(*n_)  # into orchestra
            t_ = np.array([n_[1], -n_[0]])
            s = np.radians(FUNNEL_SPREAD)
            for sign in (-1, 1):
                p0 = np.array([ax_, ay_]) + sign * t_ * (ACCESS_WIDTH / 2 + cell / 2)
                u = np.cos(s) * n_ + sign * np.sin(s) * t_
                along = np.clip((X - p0[0]) * u[0] + (Y - p0[1]) * u[1], 0, FUNNEL_LEN)
                dist = np.hypot(X - p0[0] - along * u[0], Y - p0[1] - along * u[1])
                grid[(dist <= 0.75 * cell) & orch & ~occupied & (grid != ACCESS)] = WALL

    return dict(grid=grid, occupied=occupied, orch=orch, xs=xs, ys=ys, cell=cell,
                openings=openings)


def plot_building(b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ["#2b2b2b", "#f4f4f4", "#cfe0f0", "#a7adb4", "#f39c12", "#27ae60",
              "#e74c3c", "#27ae60"]       # wall aisle legroom seat access (exit) door out
    xs, ys, c = b["xs"], b["ys"], b["cell"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(b["grid"], origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=7.5,
              extent=[xs[0] - c / 2, xs[-1] + c / 2, ys[0] - c / 2, ys[-1] + c / 2],
              interpolation="nearest")
    ax.set_title("first floor: red = doors, orange = access points, green = outside")
    ax.set_aspect("equal"); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
