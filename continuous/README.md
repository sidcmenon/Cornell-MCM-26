# Continuous crowd model (Hughes 2002)

The crowd is a density ρ(x, t) (people/m²) on the Bailey Hall first floor, from the seats
to the street. People are conserved, and everyone walks the fastest route out given the current
crowding (an eikonal equation, solved with fast marching). Equations: top of `continuum.py`.

| File | What |
|---|---|
| `building.py` | First floor out to the street: reuses `../bailey_grid.py` and adds the corridor, foyer, 3 foyer doors, 5 front doors, 4 portico doors, the 6 orchestra access points measured from the seating chart, and the barricade layouts (`BARRICADES`) |
| `continuum.py` | The model; runs every barricade layout at three clogging / merge-loss settings |

```
pip install -r requirements.txt
python continuum.py             # table of egress times, writes egress.png (~45 s)
python continuum.py --dt-check  # also repeats the baseline at half the time step
```

## How crowding is modeled
| Effect | How |
|---|---|
| Packed space is slower | v = V_FREE · T(cell type) · (1 − ρ/RHO_MAX), T = per-type speed factor in `continuum.py` (aisle 1, legroom 0.8, seat 0.1; not the CA's binary traversability field) |
| Packed aisle hard to enter / packed legroom hard to leave | Flow between cells = min(sender's demand, receiver's supply) |
| Doors and access-point steps | Capacity = DOOR_FLOW (1.3) or STAIR_FLOW (1.0) people/(m·s) × width, reduced by up to JAM_DROP when the crowd in front is packed and squeezing in (clogging) |
| Merging / crossing streams | Walking speed drops by up to MERGE_LOSS where streams converge from different directions |
| Re-planning around crowds | Routes recomputed every PHI_EVERY s from current density |

## Access points (measured on the seating chart; all are steps)
| Where | x, y (m) | Width |
|---|---|---|
| Tops of the two main aisles | ±4.4, 16.4 | 1.35 m (4'-5" on chart) |
| Stairs where the barrier turns straight | ±10.9, 9.6 | 1.1 m |
| Steps by the stage up to the boxes | ±10.9, 0.0 | 1.05 m |

## Results (first floor only, 1,010 people; corridor, foyer and portico doors are estimates)
All-out times in seconds:

| Barricades | no clogging/merging | JAM_DROP = MERGE_LOSS = 0.3 | both 0.6 |
|---|---|---|---|
| none | 187 | 186 | 201 |
| corridor_45 (outer mezz doors → outer portico doors) | 182 | 188 | 194 |
| corridor_32 (outer 4 mezz doors → portico) | 185 | 191 | 198 |
| orchestra_split (front of side sections → stage steps) | 182 | 193 | 198 |
| corridor_45 + split | 182 | 186 | 194 |

- Numerical error is about 3 s (baseline at half the time step: 183 s vs 186 s), so most
  differences above are within noise. corridor_45 has the best 90%-out time in every setting
  and helps most when clogging is strong (201 → 194 s).
- Load is very uneven: the outer mezzanine doors (±50°) and the portico doors at ±40° run at
  capacity until the end (≈267 people each through the ±40° doors), while the centre mezzanine,
  foyer and front doors are idle after ≈40 s. The result depends heavily on the estimated
  portico door positions and widths; measure them next.
