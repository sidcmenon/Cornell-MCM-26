# Sensitivity test: CA crowd model and fire model

**Method.** One parameter at a time is changed from the baseline, all others held fixed.
Each setting uses 50 replicates with the *same* random seeds as the baseline (common random
numbers), so every difference below is a paired comparison; ± is one standard error of the
paired difference. Geometry: first floor with the measured access points; "exit" = the 9
mezzanine doors. 1 step = 0.36 s. Code: `sensitivity.py` (raw results in `sens_ca.npz`,
`sens_fire.npz`).

## 1. CA crowd model (`sim.py`)

Baseline: k_S = 2.08, k_D = −2.98 (fitted optimum), mu = 0.3, alpha = delta = 0.3,
door_q = 0.6, door_cap = 1, all 1,010 seats full.
**Mean exit 63.4 s; last person reaches the corridor at 170 s.**

| Change | Mean exit (s) | Δ mean exit (s) | Δ % | Last exit (s) |
|---|---|---|---|---|
| **door_q = 0.45** (−25% door flow) | 72.7 | +9.3 ± 0.3 | **+14.7%** | 198 |
| **door_q = 0.75** (+25% door flow) | 57.0 | −6.4 ± 0.2 | **−10.1%** | 151 |
| **mu = 0** (no friction) | 58.4 | −5.1 ± 0.2 | **−8.0%** | 156 |
| mu = 0.15 | 60.6 | −2.8 ± 0.2 | −4.4% | 162 |
| mu = 0.45 | 66.5 | +3.1 ± 0.2 | +4.9% | 178 |
| **mu = 0.6** | 70.3 | +6.8 ± 0.2 | **+10.8%** | 189 |
| delta = 0.1 (trail fades slower) | 62.0 | −1.4 ± 0.2 | −2.3% | 164 |
| delta = 0.5 | 64.3 | +0.9 ± 0.2 | +1.3% | 174 |
| alpha = 0.1 (trail spreads less) | 64.2 | +0.8 ± 0.2 | +1.2% | 172 |
| alpha = 0.5 | 63.1 | −0.3 ± 0.2 | −0.5% | 168 |
| k_S = 1.56 (−25%) | 63.2 | −0.2 ± 0.2 | −0.3% | 168 |
| k_S = 2.60 (+25%) | 64.7 | +1.3 ± 0.2 | +2.0% | 176 |
| k_D = −2.23 (−25%) | 63.7 | +0.3 ± 0.2 | +0.4% | 171 |
| k_D = −3.73 (+25%) | 63.2 | −0.3 ± 0.2 | −0.4% | 169 |
| winner = uniform | 63.2 | −0.2 ± 0.3 | −0.3% | 169 |
| cut_corners = True | 63.3 | −0.2 ± 0.2 | −0.3% | 168 |
| fill = 0.75 (scenario) | 50.0 | −13.5 ± 0.2 | −21.2% | 135 |
| fill = 0.5 (scenario) | 35.7 | −27.8 ± 0.2 | −43.8% | 95 |

**Findings**
- **Door throughput and friction drive the result.** A ±25% change in door flow moves egress
  time by +15% / −10%; friction mu from 0 to 0.6 spans −8% to +11%. These are the two
  physical bottleneck parameters, and the egress time is roughly proportional to them, so
  their calibration (SFPE ~1.3 people/m/s for doors) is what the absolute numbers rest on.
- **Behaviour weights are robust.** ±25% on k_S or k_D changes egress by ≤ 2%: the optimum
  is flat, so the fitted values do not need to be exact.
- **Trail parameters (alpha, delta) and modelling choices** (conflict winner, corner
  cutting) change egress by ≤ 2.3%; conclusions do not depend on them.
- **Occupancy matters most** (half full: −44%), as expected; it is a scenario, not an
  uncertain parameter.
- The last person reaches the corridor at ~170 s at baseline (151–198 s across the door and
  friction range). Since the corridor and foyer are not yet in the CA, full-building egress
  is longer, so the **180 s target is at risk** at full capacity.

## 2. Fire model (`fire_sim.py`, `learn_kf.py`)

Baseline: random ignition, fire engulfs the hall in a median 360 s, casualty after 5 s in
fire, fire sensed within ell = 8 cells (3.5 m), crowd spread c = 1, fear spread
sigma_F = 0.5, objective weight lam = 0.1. Fear median m searched over
{0, 0.5, 1, 2, 3, 5, 8, 12, 20}.

| Change | Best m | Also within 2 SE | Casualties at best m | at m = 0 (ignore fire) | at m = 1 | at m = 20 (panic) | Mean survivor exit (s) |
|---|---|---|---|---|---|---|---|
| **baseline** | **1** | 0.5 | 1.61% | 2.49% | 1.61% | 3.53% | 65 |
| c = 0.5 (less execution error) | 1 | 0.5 | 1.47% | 2.65% | 1.47% | 3.57% | 65 |
| c = 2 | 1 | – | 1.54% | 2.67% | 1.54% | 3.58% | 65 |
| c = 4 (much more execution error) | 1 | – | 1.57% | 2.75% | 1.57% | 3.57% | 66 |
| sigma_F = 0.25 | 1 | 0.5, 2 | 1.61% | 2.49% | 1.61% | 3.63% | 65 |
| sigma_F = 1 | 0.5 | 1 | 1.77% | 2.49% | 1.78% | 3.46% | 65 |
| ell = 4 (senses fire at 1.7 m) | 1 | 0.5 | 1.97% | 2.49% | 1.97% | 2.82% | 64 |
| ell = 12 (5.2 m) | 1 | 0.5 | 1.29% | 2.49% | 1.29% | 4.80% | 66 |
| **T_engulf = 180 s** (fast fire) | 1 | 0.5, 2 | **5.26%** | 6.88% | 5.26% | 7.80% | 63 |
| T_engulf = 720 s (slow fire) | 0.5 | 1 | 0.53% | 0.84% | 0.45% | 2.87% | 64 |
| burn_time = 2.5 s | 1 | – | 2.12% | 3.68% | 2.12% | 4.25% | 65 |
| burn_time = 10 s | 0.5 | 1 | 1.11% | 1.64% | 1.06% | 2.79% | 65 |

| Objective weight lam | Best m | Also within 2 SE | Casualties at best m |
|---|---|---|---|
| 0 (casualties only) | 1 | 2 | 1.61% |
| 0.05 – 0.2 | 1 | 0.5 at 0.1, 0.2 | 1.61% |
| 0.5 (time weighted heavily) | 0.5 | 0, 1 | 1.90% |

**Findings**
- **"Moderate fear is best" is robust.** The best m is 1 in 9 of 12 settings and 0.5 in the
  other 3, where m = 1 is still within 2 SE. In *every* setting, moderate fear beats both
  ignoring the fire (m = 0) and panic (m = 20), and panic is the worst option in all of them.
- **Fire speed dominates the casualty level.** Halving the engulf time to 180 s triples
  casualties (1.6% → 5.3%); doubling it to 720 s cuts them to 0.5%. Absolute casualty
  numbers should be reported as a range over fire speed, not a single value.
- **Earlier sensing helps calm crowds but hurts panicked ones.** With ell = 12, casualties at
  m = 1 fall to 1.3%, but at m = 20 they rise to 4.8%: a panicked crowd that senses the fire
  from farther away overreacts more.
- **Execution error barely matters.** Raising the crowd's spread around optimal behaviour
  (c from 0.5 to 4) changes casualties by ≤ 0.1 percentage points and survivor exit time by
  ≤ 1 s. Consistent with section 1: the (k_S, k_D) optimum is flat.
- **The objective weighting does not change the conclusion** unless exit time is weighted
  very heavily (lam = 0.5), where m = 0.5 is best and m = 1 is still within 2 SE.

## Caveats
- One-at-a-time: interactions (e.g. fast fire *and* low door flow) are not tested.
- Absolute times are to the mezzanine doors, not out of the building.
- door_q is a calibration of door flow; the CA's time step (0.36 s) only converts steps to
  seconds and was not varied separately.

## Reproduce
```
python3 sensitivity.py ca     # ~1.5 min on 10 cores
python3 sensitivity.py fire   # ~40 min on 10 cores
```
