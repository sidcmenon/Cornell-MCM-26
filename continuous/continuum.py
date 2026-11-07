"""
continuum.py
Continuous (macroscopic) crowd model of the Bailey Hall first floor, after Hughes (2002),
from the seats all the way out of the building (geometry in building.py).

The crowd is a density rho(x, t) in people/m^2 that obeys

    d(rho)/dt + div( rho * v * d ) = 0          (people are conserved)
    |grad phi| = 1 / v_eik(rho, x),  phi = 0 outside   (phi = time to get out)
    d = -grad phi / |grad phi|                  (everyone walks the fastest route)

Crowding enters in three places:
  1. Speed falls with density, separately for each cell type (Greenshields):
         q(rho) = vf * rho * (1 - rho/RHO_MAX),   vf = V_FREE * T(type)
     T is the traversability field from the CA (aisle 1.0, legroom 0.8, ...).
  2. Flow between two cells is min(demand of the sender, supply of the receiver)
     (Godunov / cell-transmission form). A packed aisle has little supply, so it is
     hard to enter; a packed legroom row has its demand capped, so it empties slowly.
  3. Every opening (door or access point) passes at most DOOR_FLOW * width people/s.
     That capacity drops when the crowd in front of it is packed AND has to squeeze in
     from a wider area (clogging):
         capacity = DOOR_FLOW * width * (1 - JAM_DROP * jam * squeeze)
         jam     = how far the density in front is above critical, 0..1
         squeeze = 1 - width / (width of open floor 2 cells upstream), 0..1
     Funnel barriers lower `squeeze`. This clogging law is an assumption; JAM_DROP is
     varied in the output because the benefit of funnels depends on it.

The grid is only the numerical mesh: halving DT should barely change results.

Run:  python continuum.py            -> table of egress times, writes egress.png
      python continuum.py --dt-check -> also repeats the baseline at DT/2
"""

import sys
from pathlib import Path

import numpy as np
import skfmm
from scipy import ndimage

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bailey_grid import AISLE, EXIT, LEGROOM, SEAT, WALL      # noqa: E402
from traversability_field import traversability_field        # noqa: E402
from building import OUT, build_building                      # noqa: E402

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
V_FREE = 1.3        # m/s, free walking speed (Weidmann 1993: 1.34)
RHO_MAX = 5.4       # people/m^2, jam density (Weidmann 1993)
DOOR_FLOW = 1.3     # people/(m s), specific flow through an opening (SFPE handbook)
JAM_DROP = 0.3      # max fraction of capacity lost to clogging
V_EIK_MIN = 0.1     # route planning: a jammed cell still costs finite time
PHI_EVERY = 1.0     # s, how often people re-plan (recompute phi)
T_END = 900.0       # s, give up after this
DONE = 0.5          # people left inside when we call it evacuated

RHO_C = RHO_MAX / 2                       # critical density (max flow)
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (dy, dx): east, west, north, south
SQ = np.ones((3, 3), bool)


def nb(a, dy, dx):
    """Value of a at the neighbour (y+dy, x+dx). The grid border is all wall."""
    return np.roll(a, (-dy, -dx), axis=(0, 1))


def initial_density(b):
    """Each seated person is spread evenly over their seat and the seat/legroom/aisle
    cells around it (they stand up), conserving the head count."""
    walk = np.isin(b["grid"], [AISLE, LEGROOM, SEAT])
    occ = b["occupied"].astype(float)
    n = sum(nb(walk, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    share = occ / np.maximum(n, 1)
    people = sum(nb(share, -dy, -dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) * walk
    return people / b["cell"] ** 2


def travel_time(speed, b):
    grid = b["grid"]
    phi0 = np.ma.MaskedArray(np.where(grid == OUT, -1.0, 1.0), mask=(grid == WALL))
    phi = skfmm.travel_time(phi0, np.maximum(speed, 1e-6), dx=b["cell"]).filled(np.inf)
    phi[grid == OUT] = 0.0
    return phi


def directions(phi):
    """Unit direction of steepest descent of phi, split into the 4 face components."""
    with np.errstate(invalid="ignore"):                # inf - inf inside walls
        g = [np.nan_to_num(np.maximum(phi - nb(phi, dy, dx), 0), nan=0, posinf=0)
             for dy, dx in DIRS]
    gx = np.where(g[0] >= g[1], g[0], -g[1])
    gy = np.where(g[2] >= g[3], g[2], -g[3])
    norm = np.hypot(gx, gy) + 1e-12
    return [np.maximum(gx, 0) / norm, np.maximum(-gx, 0) / norm,
            np.maximum(gy, 0) / norm, np.maximum(-gy, 0) / norm]


def opening_fronts(b, vf):
    """For each opening: the floor in front of it (upstream, within 2 cells) and the
    squeeze factor 1 - width / width of open floor 2 cells upstream."""
    phi = travel_time(vf, b)
    floor = np.isin(b["grid"], [AISLE, LEGROOM, SEAT])
    out = []
    for mask, width in b["openings"]:
        up = floor & (phi > phi[mask].max())
        d1 = ndimage.binary_dilation(mask, SQ)
        d2 = ndimage.binary_dilation(d1, SQ)
        ring2 = d2 & ~d1 & up
        squeeze = np.clip(1 - width / max(ring2.sum() * b["cell"], 1e-9), 0, 1)
        out.append((mask, width, d2 & up, squeeze))
    return out


def simulate(funnels=False, dt=None, jam_drop=JAM_DROP):
    b = build_building(funnels)
    grid, cell = b["grid"], b["cell"]
    assert (grid[[0, -1], :] == WALL).all() and (grid[:, [0, -1]] == WALL).all()
    dt = dt or 0.5 * cell / V_FREE                     # CFL condition

    vf = V_FREE * traversability_field(np.minimum(grid, EXIT))   # doors/outside = 1.0
    out_ = grid == OUT
    fronts = opening_fronts(b, vf)

    rho = initial_density(b)
    total = rho.sum() * cell**2
    assert np.isfinite(travel_time(vf, b)[rho > 0]).all(), "someone is walled in"
    t, out, t_phi, history, snaps = 0.0, 0.0, -np.inf, [], {}

    while t < T_END and total - out > DONE:
        if t - t_phi >= PHI_EVERY - 1e-9:              # re-plan routes
            dirs = directions(travel_time(vf * np.maximum(1 - rho / RHO_MAX, V_EIK_MIN), b))
            t_phi = t

        demand = vf * np.minimum(rho, RHO_C) * (1 - np.minimum(rho, RHO_C) / RHO_MAX)
        rs = np.clip(np.maximum(rho, RHO_C), None, RHO_MAX)
        supply = np.where(out_, np.inf, vf * rs * (1 - rs / RHO_MAX))

        # people moving across each face this step: min(demand of sender, supply of receiver)
        flows = [np.minimum(demand * c, nb(supply, dy, dx)) * cell * dt
                 for c, (dy, dx) in zip(dirs, DIRS)]

        # capacity of each opening, reduced by clogging
        for mask, width, front, squeeze in fronts:
            jam = np.clip((rho[front].mean() - RHO_C) / (RHO_MAX - RHO_C), 0, 1)
            cap = DOOR_FLOW * width * (1 - jam_drop * jam * squeeze) * dt
            entering = [nb(mask, dy, dx) & ~mask for dy, dx in DIRS]
            wanted = sum(f[e].sum() for f, e in zip(flows, entering))
            if wanted > cap:
                for f, e in zip(flows, entering):
                    f[e] *= cap / wanted

        # never send out more people than a cell holds
        sent = sum(flows) / cell**2
        scale = np.where(sent > rho, rho / np.maximum(sent, 1e-12), 1.0)
        flows = [f * scale for f in flows]

        for f, (dy, dx) in zip(flows, DIRS):
            rho -= f / cell**2
            rho += nb(f, -dy, -dx) / cell**2
        out += rho[out_].sum() * cell**2
        rho[out_] = 0.0
        t += dt

        history.append((t, total - out, rho.max()))
        for ts in (20, 60, 120):
            if ts not in snaps and t >= ts:
                snaps[ts] = rho.copy()

    return dict(b=b, history=np.array(history), snaps=snaps, total=total, dt=dt,
                squeeze=[f[3] for f in fronts])


def summary(r):
    t, left, rmax = r["history"].T
    def when(frac):
        i = np.argmax(left <= (1 - frac) * r["total"])
        return t[i] if left[i] <= (1 - frac) * r["total"] else np.nan
    return when(0.5), when(0.9), t[-1], rmax.max()


def plot(runs, path=Path(__file__).with_name("egress.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0 = runs["baseline"]
    b, c = r0["b"], r0["b"]["cell"]
    ext = [b["xs"][0] - c / 2, b["xs"][-1] + c / 2, b["ys"][0] - c / 2, b["ys"][-1] + c / 2]
    fig, axs = plt.subplots(1, len(r0["snaps"]) + 1, figsize=(5 * (len(r0["snaps"]) + 1), 5))
    for ax, (ts, rho) in zip(axs, r0["snaps"].items()):
        im = ax.imshow(np.ma.masked_where(b["grid"] == WALL, rho), origin="lower",
                       extent=ext, cmap="inferno", vmin=0, vmax=RHO_MAX)
        ax.set_title(f"baseline density, t = {ts} s"); ax.set_aspect("equal")
    fig.colorbar(im, ax=axs[:-1], shrink=0.7, label="people / m$^2$")
    for name, r in runs.items():
        axs[-1].plot(r["history"][:, 0], r["history"][:, 1], label=name)
    axs[-1].axvline(180, ls="--", c="gray")
    axs[-1].set_xlabel("time (s)"); axs[-1].set_ylabel("people inside")
    axs[-1].set_title("people still inside"); axs[-1].legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    runs = {}
    print(f"{'run':28s} {'50% out':>8s} {'90% out':>8s} {'all out':>8s} {'peak rho':>9s}")
    for jd in (0.0, JAM_DROP, 0.6):
        for funnels in (False, True):
            name = f"{'funnels' if funnels else 'baseline'}, JAM_DROP={jd}"
            r = simulate(funnels, jam_drop=jd)
            print(f"{name:28s} " + "{:7.0f}s {:7.0f}s {:7.0f}s {:9.2f}".format(*summary(r)))
            if jd == JAM_DROP:
                runs["funnels" if funnels else "baseline"] = r
    print("access-point squeeze, baseline:", np.round(runs["baseline"]["squeeze"][9:15], 2))
    print("access-point squeeze, funnels: ", np.round(runs["funnels"]["squeeze"][9:15], 2))
    plot(runs)
    if "--dt-check" in sys.argv:
        r = simulate(dt=runs["baseline"]["dt"] / 2)
        print(f"{'baseline at dt/2':28s} " + "{:7.0f}s {:7.0f}s {:7.0f}s {:9.2f}".format(*summary(r)))
