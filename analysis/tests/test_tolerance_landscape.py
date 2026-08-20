import numpy as np

from analysis.tolerance_landscape import (
    GeneratorEstimate,
    conditioned_tv_distance,
    marginal_kernel,
    marginal_tv_distance,
    viable_future_summary,
)


def gen_from_theta(theta):
    p = 0.60
    kernels = {}
    row_counts = {}
    regime_counts = {(0,): 100, (1,): 100}
    for z in [(0,), (1,)]:
        K = np.eye(3)
        pv = p + theta if z == (0,) else p - theta
        K[0] = [0.0, pv, 1 - pv]
        kernels[z] = K
        for i in range(3):
            row_counts[(z, i)] = 100
    return GeneratorEstimate(
        kernels=kernels,
        row_counts=row_counts,
        regime_counts=regime_counts,
        state_edges=np.array([-np.inf, 0.5, 1.5, np.inf]),
        condition_edges={"z": np.array([-np.inf, 0.0, np.inf])},
        n_transitions=200,
    )


def test_hidden_architecture_aliasing():
    a = gen_from_theta(-0.30)
    b = gen_from_theta(+0.30)
    weights = {((0,), 0): 0.5, ((1,), 0): 0.5}
    d_cond, cov = conditioned_tv_distance(a, b, evaluation_weights=weights)
    assert d_cond > 0.5
    assert cov == 1.0

    nu = {(0,): 0.5, (1,): 0.5}
    ma = marginal_kernel(a, nu)
    mb = marginal_kernel(b, nu)
    d_marg, _ = marginal_tv_distance(ma, mb, row_weights=[1.0, 0.0, 0.0])
    assert np.isclose(d_marg, 0.0)


def test_viable_mass():
    K = np.array([
        [0.0, 0.8, 0.2],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
    ])
    init = np.array([1.0, 0.0, 0.0])
    out = viable_future_summary(K, init, [True, True, False], horizon_steps=2)
    assert np.isclose(out["viable_mass"], 0.8 * 0.9)
    assert out["effective_terminal_states"] >= 1.0
