import numpy as np

from fift.fast_hankel_c import CNUFHTPlan


def _j0_scalar(x: float, max_terms: int = 80) -> float:
    term = 1.0
    acc = 1.0
    if x == 0.0:
        return 1.0
    factor = - (x * x) / 4.0
    for m in range(1, max_terms):
        term *= factor / (m * m)
        acc += term
        if abs(term) < 1e-15:
            break
    return acc


_j0_vec = np.vectorize(_j0_scalar)


def _direct_sum(rs: np.ndarray, ws: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    out = np.empty((coeffs.shape[0], ws.size), dtype=np.float64)
    for i in range(coeffs.shape[0]):
        for j, wval in enumerate(ws):
            out[i, j] = np.sum(coeffs[i] * _j0_vec(wval * rs))
    return out


def test_cfastfht_matches_direct_sum():
    rng = np.random.default_rng(42)
    rs = np.linspace(0.05, 3.0, 24)
    ws = np.linspace(0.1, 4.0, 12)
    coeffs = rng.standard_normal((4, rs.size))

    plan = CNUFHTPlan(
        0.0,
        rs,
        ws,
        tol=1e-10,
        min_dim_prod=10**6,
        K_asy=-1,
        K_loc=-1,
    )
    try:
        approx = plan.execute_batch(coeffs)
    finally:
        plan.close()

    reference = _direct_sum(rs, ws, coeffs)
    assert np.allclose(approx, reference, rtol=5e-6, atol=5e-6)
