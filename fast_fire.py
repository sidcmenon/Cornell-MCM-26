import numpy as np
from learn_kf import search, report, CONFIG

if __name__ == "__main__":
    # 3-minute fire; casualty on the first step in a burning cell
    cfg = dict(CONFIG, T_engulf=180.0, burn_time=0.36)
    m_values = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    res = search(m_values, cfg=cfg)
    report(res, m_values)
    np.savez("results_kf_fast.npz", m=m_values, res=res,
             columns=np.array(["J", "casualty", "burned", "trapped", "mean_exit"]))