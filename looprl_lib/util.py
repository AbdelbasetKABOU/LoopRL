import numpy as np


def apply_temperature(policy: np.ndarray, tau: float) -> np.ndarray:
    assert tau >= 0
    if tau == 0:
        res = np.zeros(len(policy), dtype=policy.dtype)
        max = np.max(policy)
        res[policy == max] = 1
    else:
        invtau = 1/tau
        res = policy ** invtau
    res /= res.sum()
    return res
