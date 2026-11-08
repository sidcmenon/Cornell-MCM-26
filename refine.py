import numpy as np

r = np.load("results_refined.npz")
kS, kD, res = r["k_S"], r["k_D"], r["res"][..., 0]      # res: (nS, nD, reps)
S, Dg = np.meshgrid(kS, kD, indexing="ij")
X = np.column_stack([np.ones(S.size), S.ravel(), Dg.ravel(),
                     S.ravel()**2, (S*Dg).ravel(), Dg.ravel()**2])

def fit(J):
    c, *_ = np.linalg.lstsq(X, J.ravel(), rcond=None)
    H = np.array([[2*c[3], c[4]], [c[4], 2*c[5]]])
    return np.linalg.solve(H, -c[1:3]), H, c

(x, H, c) = fit(res.mean(-1))
print("minimizer k_S, k_D:", np.round(x, 3), " Hessian eigenvalues:", np.round(np.linalg.eigvalsh(H), 3))
print("predicted J at minimizer:", round(c @ [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2], 2))

# bootstrap over replicates (same replicate draw at every grid point, preserving common random numbers)
rng = np.random.default_rng(0)
R = res.shape[-1]
boots = np.array([fit(res[..., rng.integers(0, R, R)].mean(-1))[0] for _ in range(2000)])
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
print("95% CI  k_S:", np.round([lo[0], hi[0]], 2), "  k_D:", np.round([lo[1], hi[1]], 2))