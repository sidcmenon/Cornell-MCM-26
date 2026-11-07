"""
build_fields.py
Builds the grid and static fields, runs the grid checks, saves everything to
bailey_grid.npz and writes bailey_grid.png and bailey_fields.png.
The dynamic field is created inside the simulation (see dynamic_field.py).
"""

import numpy as np

from bailey_grid import build_grid, check, extent, plot_grid
from distance_field import distance_field
from traversability_field import traversability_field


def plot_fields(g, S, T, path="bailey_fields.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = extent(g)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    im = axs[0].imshow(np.ma.masked_invalid(S), origin="lower", extent=ext, cmap="viridis_r")
    axs[0].set_title("distance field S (cells; orchestra via access points)")
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    im = axs[1].imshow(T, origin="lower", extent=ext, cmap="magma")
    axs[1].set_title("traversability field T")
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    for a in axs:
        a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    g = build_grid()
    check(g)
    S = distance_field(g)
    T = traversability_field(g["grid"])
    np.savez("bailey_grid.npz", grid=g["grid"], occupied=g["occupied"], orch=g["orch"],
             S_distance=S, traversability=T, xs=g["xs"], ys=g["ys"])
    plot_grid(g)
    plot_fields(g, S, T)