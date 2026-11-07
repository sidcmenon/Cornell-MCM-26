"""
bailey_grid.py
Cellular-automaton grid (geometry and cell types only; fields live in their own files) for the first floor of Bailey Hall:
orchestra, barrier (wall cells) with 6 access points, mezzanine/loge, side boxes, 9 exit doors.

Coordinates (metres): x is lateral (0 on the centreline), y is depth measured from
the stage toward the back wall. Arrays are indexed grid[iy, ix], iy = 0 at the stage.

The geometry is a parametric reconstruction of the seating chart. Dimensions printed
on the chart are used where they exist (row pitch 2'-10", 27'-0" centre rows); the rest
are estimates collected in PARAMETERS. Door and access-point positions and widths are
placeholders until real plans are available. Seat counts per row are estimates from the
chart, set so that orchestra + mezzanine = 525 + 485 = 1010 chairs.
"""

from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
IN, FT = 0.0254, 0.3048
ROW_PITCH = 34 * IN            # 2'-10" = 0.864 m
CELL = ROW_PITCH / 2           # 0.432 m, so one row = legroom band + seat band

STAGE_Y = -0.5                 # stage front (treated as wall)
HALF_WIDTH = 15.6              # inside face of the side walls

# Orchestra: rows are arcs about (0, ORCH_CENTER_Y)
ORCH_CENTER_Y = -14.0
ORCH_RHO0 = 14.5               # radius where row A's legroom begins (y = 0.5 on the axis)
CENTER_HALF = 27 * FT / 2      # centre section 27'-0" wide
INNER_AISLE = 0.9
SIDE_X0 = CENTER_HALF + INNER_AISLE
SIDE_SEAT_W = 21 * IN

# Barrier and mezzanine: arcs about (0, LOGE_CENTER_Y)
LOGE_CENTER_Y = 4.6
BARRIER_R = 12.0               # orchestra edge of the barrier, arc part
BARRIER_X = 10.8               # orchestra edge of the barrier, straight part along the boxes
LOGE_R0 = 13.3                 # radius where row AA's legroom begins
OUTER_R = 18.8                 # back wall of the rear walkway (doors are here)
LOGE_CENTER_PHI = 18.0         # centre loge section |phi| <= 18 deg
LOGE_SIDE_PHI0 = 22.0          # side loge sections start at 22 deg
LOGE_SIDE_YMIN = 11.6          # side loge rows stop here (box walkway in front)
LOGE_SIDE_XMAX = 14.7          # side loge rows stop here (side aisle along the wall)

# Boxes: straight rows parallel to the centreline, between barrier and side wall
BOX_X0 = 11.232                # first box row's legroom edge (aligned to a cell edge)
BOX_ROWS = 5
BOX_TOP = 9.8                  # boxes occupy STAGE_Y <= y < BOX_TOP; walkway behind

# Exits and access points: placeholders, no evacuation plans yet
DOOR_PHI = np.linspace(-50, 50, 9)          # 9 doors in the back wall, deg from the axis
DOOR_WIDTH = 1.3
ACCESS_X = [-10.3, -7.6, -4.6, 4.6, 7.6, 10.3]
ACCESS_WIDTH = 1.0

# Seat counts (estimates from the chart)
ORCH_ROWS = list("ABCDEFGHJKLMNPQRST")
ORCH_CENTER_SEATS = dict(A=13, B=14, S=15, T=13, **{r: 16 for r in "CDEFGHJKLMNPQR"})
ORCH_SIDE_SEATS = dict(A=6, B=7, M=7, N=7, P=6, Q=5, R=3, S=1, T=0,
                       **{r: 9 for r in "CDEFGHJKL"})          # per side
LOGE_ROWS = ["AA", "BB", "CC", "DD", "EE"]
LOGE_CENTER_SEATS = [15, 16, 16, 16, 16]
LOGE_SIDE_SEATS = [16, 18, 20, 21, 23]                        # per side
BOX_SEATS_PER_ROW = 21

# ----------------------------------------------------------------------------
# Cell types
# ----------------------------------------------------------------------------
WALL, AISLE, LEGROOM, SEAT, ACCESS, EXIT = range(6)   # the barrier is WALL
TYPE_NAMES = ["wall", "aisle", "legroom", "seat", "access", "exit"]
PASSABLE = [AISLE, LEGROOM, SEAT, ACCESS, EXIT]

NEI4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
NEI8 = NEI4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _shift(m, dy, dx):
    out = np.zeros_like(m)
    H, W = m.shape
    out[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] = \
        m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return out


def _dilate8(m):
    out = m.copy()
    for dy, dx in NEI8:
        out |= _shift(m, dy, dx)
    return out


# ----------------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------------
def build_grid():
    nx = 2 * int(np.ceil((HALF_WIDTH + 2 * CELL) / CELL))
    x_min = -nx * CELL / 2
    y_min = STAGE_Y - 2 * CELL
    ny = int(np.ceil((LOGE_CENTER_Y + OUTER_R + 2 * CELL - y_min) / CELL))
    xs = x_min + (np.arange(nx) + 0.5) * CELL
    ys = y_min + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)

    r_orch = np.hypot(X, Y - ORCH_CENTER_Y)
    r_loge = np.hypot(X, Y - LOGE_CENTER_Y)
    phi = np.degrees(np.arctan2(X, Y - LOGE_CENTER_Y))
    ax, aphi = np.abs(X), np.abs(phi)

    hall = (Y >= STAGE_Y) & (ax < HALF_WIDTH) & (r_loge < OUTER_R)
    orch = hall & (ax < BARRIER_X) & (r_loge < BARRIER_R)
    barrier = _dilate8(orch) & hall & ~orch      # one cell thick, no diagonal leaks
    mezz = hall & ~orch & ~barrier

    grid = np.full((ny, nx), WALL, np.int8)
    grid[orch | mezz] = AISLE
    # barrier cells stay WALL; access points are cut into it below

    occupied = np.zeros((ny, nx), bool)
    agents = []                                   # (iy, ix, section, row)

    def add_row(band, inner_half, span, points, section, row):
        """Rasterise one row. The outer half-band is the chair band, the inner half is
        legroom. Each chair goes to the nearest free cell, preferring the chair band;
        chair-band cells left without an occupant stay SEAT (empty chairs)."""
        row_cells = band & span
        grid[row_cells & inner_half] = LEGROOM
        grid[row_cells & ~inner_half] = SEAT
        free = np.argwhere(row_cells & ~occupied)
        if len(free) < len(points):
            raise ValueError(f"{section} row {row}: {len(points)} chairs, "
                             f"only {len(free)} cells")
        cx, cy = xs[free[:, 1]], ys[free[:, 0]]
        penalty = np.where(inner_half[free[:, 0], free[:, 1]], 4 * CELL ** 2, 0.0)
        taken = np.zeros(len(free), bool)
        for px, py in points:
            d = (cx - px) ** 2 + (cy - py) ** 2 + penalty
            d[taken] = np.inf
            j = int(np.argmin(d))
            taken[j] = True
            iy, ix = free[j]
            occupied[iy, ix] = True
            grid[iy, ix] = SEAT
            agents.append((int(iy), int(ix), section, row))

    # Orchestra
    for k, row in enumerate(ORCH_ROWS):
        rho = ORCH_RHO0 + k * ROW_PITCH
        band = orch & (r_orch >= rho) & (r_orch < rho + ROW_PITCH)
        inner = r_orch < rho + ROW_PITCH / 2
        rs = rho + 0.75 * ROW_PITCH

        def on_arc(x, rs=rs):
            return x, np.sqrt(rs ** 2 - x ** 2) + ORCH_CENTER_Y

        n = ORCH_CENTER_SEATS[row]
        if n:
            pts = [on_arc(-CENTER_HALF + (j + 0.5) * 2 * CENTER_HALF / n) for j in range(n)]
            add_row(band, inner, ax <= CENTER_HALF, pts, "orchestra", row)
        n = ORCH_SIDE_SEATS[row]
        if n:
            x1 = SIDE_X0 + n * SIDE_SEAT_W
            for s in (-1, 1):
                span = (s * X >= SIDE_X0) & (s * X <= x1)
                pts = [on_arc(s * (SIDE_X0 + (j + 0.5) * SIDE_SEAT_W)) for j in range(n)]
                add_row(band, inner, span, pts, "orchestra", row)

    # Loge
    for k, row in enumerate(LOGE_ROWS):
        r0 = LOGE_R0 + k * ROW_PITCH
        band = mezz & (r_loge >= r0) & (r_loge < r0 + ROW_PITCH)
        inner = r_loge < r0 + ROW_PITCH / 2
        rs = r0 + 0.75 * ROW_PITCH

        def polar(p, rs=rs):
            t = np.radians(p)
            return rs * np.sin(t), LOGE_CENTER_Y + rs * np.cos(t)

        n = LOGE_CENTER_SEATS[k]
        pts = [polar(-LOGE_CENTER_PHI + (j + 0.5) * 2 * LOGE_CENTER_PHI / n) for j in range(n)]
        add_row(band, inner, aphi <= LOGE_CENTER_PHI, pts, "mezzanine", row)

        n = LOGE_SIDE_SEATS[k]
        phi_end = min(np.degrees(np.arccos((LOGE_SIDE_YMIN - LOGE_CENTER_Y) / rs)),
                      np.degrees(np.arcsin(min(1.0, LOGE_SIDE_XMAX / rs))))
        side = (aphi >= LOGE_SIDE_PHI0) & (Y >= LOGE_SIDE_YMIN) & (ax <= LOGE_SIDE_XMAX)
        for s in (-1, 1):
            pts = [polar(s * (LOGE_SIDE_PHI0 + (j + 0.5) * (phi_end - LOGE_SIDE_PHI0) / n))
                   for j in range(n)]
            add_row(band, inner, side & (s * X > 0), pts, "mezzanine", row)

    # Boxes (part of the mezzanine level)
    for i in range(BOX_ROWS):
        a = BOX_X0 + i * ROW_PITCH
        for s in (-1, 1):
            band = mezz & (s * X >= a) & (s * X < a + ROW_PITCH) & (Y < BOX_TOP)
            inner = s * X < a + ROW_PITCH / 2
            n = BOX_SEATS_PER_ROW
            xc = s * (a + 0.75 * ROW_PITCH)
            pts = [(xc, STAGE_Y + (j + 0.5) * (BOX_TOP - STAGE_Y) / n) for j in range(n)]
            add_row(band, inner, np.ones_like(band), pts, "mezzanine", f"box{i + 1}")

    # Exit doors in the back wall
    ring = _dilate8(mezz) & ~hall & (r_loge >= OUTER_R - CELL)
    for p in DOOR_PHI:
        grid[ring & (np.abs(np.radians(phi - p)) * OUTER_R <= DOOR_WIDTH / 2)] = EXIT

    # Access points: openings in the barrier
    rb = BARRIER_R + CELL / 2
    for xa in ACCESS_X:
        ya = LOGE_CENTER_Y + np.sqrt(rb ** 2 - xa ** 2)
        grid[barrier & (np.hypot(X - xa, Y - ya) <= ACCESS_WIDTH / 2)] = ACCESS

    return dict(grid=grid, occupied=occupied, agents=agents, xs=xs, ys=ys,
                orch=orch, mezz=mezz, cell=CELL)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def _bfs8(start_mask, allowed):
    seen = start_mask.copy()
    q = deque(map(tuple, np.argwhere(start_mask)))
    H, W = allowed.shape
    while q:
        y, x = q.popleft()
        for dy, dx in NEI8:
            v, u = y + dy, x + dx
            if 0 <= v < H and 0 <= u < W and allowed[v, u] and not seen[v, u]:
                seen[v, u] = True
                q.append((v, u))
    return seen


def check(g):
    grid, agents = g["grid"], g["agents"]
    counts = {}
    for _, _, sec, _ in agents:
        counts[sec] = counts.get(sec, 0) + 1
    print("chairs:", counts, "total", len(agents))
    assert len(agents) == 1010

    # every occupant can step out of its seat into space connected to an exit
    walk = np.isin(grid, [AISLE, LEGROOM, ACCESS, EXIT])
    reach = _bfs8(grid == EXIT, walk)
    stuck = [a for a in agents
             if not any(0 <= a[0] + dy < grid.shape[0] and 0 <= a[1] + dx < grid.shape[1]
                        and reach[a[0] + dy, a[1] + dx] for dy, dx in NEI8)]
    print("occupants with no walkable route out:", len(stuck))

    # orchestra must not reach the exits except through the access points
    no_access = np.isin(grid, [AISLE, LEGROOM, SEAT, EXIT])
    leak = _bfs8(grid == EXIT, no_access) & g["orch"]
    print("orchestra cells reachable without an access point:", int(leak.sum()))

    print("exit cells:", int((grid == EXIT).sum()),
          " access cells:", int((grid == ACCESS).sum()),
          " grid shape:", grid.shape, f" cell = {CELL:.3f} m")


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
CELL_COLORS = ["#2b2b2b", "#f4f4f4", "#cfe0f0", "#a7adb4", "#f39c12", "#27ae60"]


def extent(g):
    xs, ys, c = g["xs"], g["ys"], g["cell"]
    return [xs[0] - c / 2, xs[-1] + c / 2, ys[0] - c / 2, ys[-1] + c / 2]


def plot_grid(g, path="bailey_grid.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    xs, ys = g["xs"], g["ys"]
    fig, axp = plt.subplots(figsize=(11, 8.5))
    axp.imshow(g["grid"], origin="lower", extent=extent(g), interpolation="nearest",
               cmap=ListedColormap(CELL_COLORS), vmin=-0.5, vmax=len(CELL_COLORS) - 0.5)
    oy, ox = np.nonzero(g["occupied"])
    axp.scatter(xs[ox], ys[oy], s=4, c="#1f3a93")
    axp.legend(handles=[Patch(color=col, label=n) for col, n in zip(CELL_COLORS, TYPE_NAMES)]
               + [plt.Line2D([], [], marker="o", ls="", color="#1f3a93", ms=3,
                             label="occupant")], loc="lower right", fontsize=8)
    axp.set_title("Bailey Hall first floor, CA grid (stage at bottom)")
    axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)"); axp.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    g = build_grid()
    check(g)
    plot_grid(g)