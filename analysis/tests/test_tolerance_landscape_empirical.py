import numpy as np

from analysis.tolerance_landscape import GeneratorEstimate
from analysis.tolerance_landscape_empirical import (
    admissible_mask_from_edges,
    fixed_nu_marginal_with_coverage,
    paired_conditioned_marginal_tv,
    threshold_aware_state_edges,
)


def _gen(theta: float, omit_second_regime: bool = False) -> GeneratorEstimate:
    p = 0.60
    kernels = {}
    row_counts = {}
    regime_counts = {(0,): 100, (1,): 100}
    regimes = [(0,)] if omit_second_regime else [(0,), (1,)]
    for z in regimes:
        K = np.eye(3, dtype=float)
        pv = p + theta if z == (0,) else p - theta
        K[0] = [0.0, pv, 1.0 - pv]
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


def test_threshold_is_explicit_state_boundary():
    values = np.linspace(0.2, 9.8, 500)
    edges = threshold_aware_state_edges(values, n_bins=6, threshold=3.0)
    assert np.any(np.isclose(edges[np.isfinite(edges)], 3.0))
    adm = admissible_mask_from_edges(edges, 3.0)
    assert adm.any()
    assert not adm.all()
    threshold_idx = int(np.where(np.isclose(edges, 3.0))[0][0])
    assert not adm[threshold_idx - 1]
    assert adm[threshold_idx]


def test_paired_aliasing_gap_obeys_convexity():
    a = _gen(-0.30)
    b = _gen(+0.30)
    nu = {(0,): 0.5, (1,): 0.5}
    out = paired_conditioned_marginal_tv(
        a, b, nu=nu, state_weights=[1.0, 0.0, 0.0]
    )
    assert out.conditioned_tv > 0.5
    assert np.isclose(out.marginal_tv, 0.0)
    assert np.isclose(out.aliasing_gap, out.conditioned_tv)
    assert np.isclose(out.joint_support_coverage, 1.0)


def test_fixed_nu_support_gate_is_explicit():
    sparse = _gen(0.10, omit_second_regime=True)
    nu = {(0,): 0.5, (1,): 0.5}
    K_strict, cov = fixed_nu_marginal_with_coverage(
        sparse, nu, min_regime_coverage=0.90
    )
    assert np.allclose(cov, 0.5)
    assert np.isnan(K_strict).all()

    K_relaxed, cov2 = fixed_nu_marginal_with_coverage(
        sparse, nu, min_regime_coverage=0.50
    )
    assert np.allclose(cov2, 0.5)
    assert np.isfinite(K_relaxed).all()
