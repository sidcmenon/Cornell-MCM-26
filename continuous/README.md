# Continuous crowd model (Hughes 2002)

The crowd is a density ρ(x, t) (people/m²) on the Bailey Hall first floor, from the seats
to the street. People are conserved, and everyone walks the fastest route out given the current
crowding (an eikonal equation, solved with fast marching). Equations: top of `continuum.py`.

| File | What |
|---|---|
| `building.py` | First floor out to the street: reuses `../bailey_grid.py` and adds the corridor, foyer, 3 foyer doors, 5 front doors, 4 portico doors, and optional funnel barriers at the 6 orchestra access points |
| `continuum.py` | The model; runs baseline vs funnels for three clogging strengths |

```
pip install -r requirements.txt
python continuum.py             # table of egress times, writes egress.png (~20 s)
python continuum.py --dt-check  # also repeats the baseline at half the time step
```

## How crowding is modeled
| Effect | How |
|---|---|
| Packed space is slower | v = V_FREE · T(cell type) · (1 − ρ/RHO_MAX), T from `../traversability_field.py` |
| Packed aisle hard to enter / packed legroom hard to leave | Flow between cells = min(sender's demand, receiver's supply) |
| Doors and access points | Capacity DOOR_FLOW · width, reduced by up to JAM_DROP when the crowd in front is packed and has to squeeze in from a wider area (assumed clogging law) |
| Re-planning around crowds | Routes recomputed every PHI_EVERY s from current density |

## Results so far (first floor only, 1,010 people, door/corridor sizes are estimates)
| Run | 50% out | 90% out | All out |
|---|---|---|---|
| Baseline | 77 s | 139 s | 160 s |
| Funnels at access points | 79 s | 144 s | 167 s |
| Baseline at dt/2 (convergence) | 76 s | 135 s | 155 s |

- Corridor and foyer stay nearly empty: the bottlenecks are the 6 orchestra access points and
  the route to the side portico doors.
- Funnels made it 5–8 s slower at every clogging strength: they take floor space, and in this
  model the access points are limited by walking speed (T = 0.6), not by clogging.
