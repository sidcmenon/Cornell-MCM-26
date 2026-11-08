import numpy as np
from learn import grid_search, CONFIG

if __name__ == "__main__":
    cfg = dict(CONFIG, n_reps=200, base_seed=2024)
    a = grid_search([1.9], [-3.0], cfg=cfg)[0, 0, :, 0]
    b = grid_search([1.9], [0.0], cfg=cfg)[0, 0, :, 0]
    d = a - b                                    # paired: same seeds for both
    print("k_D=-3:", round(a.mean(), 2), "  k_D=0:", round(b.mean(), 2))
    print("mean diff", round(d.mean(), 2), "+/-", round(d.std(ddof=1) / np.sqrt(len(d)), 2))