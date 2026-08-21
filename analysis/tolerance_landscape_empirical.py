from __future__ import annotations

"""Empirical safeguards for the Elkhorn Tolerance Landscape workhorse.

This module keeps the paper's static generator-space interpretation while
adding constraints needed for a defensible real-data comparison:

* the oxygen-admissibility threshold is an explicit state-bin boundary;
* conditioned and marginalized TV use the same fixed state and perturbation
  weights, so the aliasing gap has the Jensen inequality expected by theory;
* candidate marginal kernels are evaluated under the fixed reference
  perturbation law and carry an explicit support-coverage diagnostic;
* same-state/different-architecture pairs use a direct generator-to-generator
  distance rather than a difference of distances to a third reference.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analysis import tolerance_landscape as tl

EPS = 1e-12


@dataclass
class PairedDistance:
    conditioned_tv: float
    marginal_tv: float
    aliasing_gap: float
    joint_support_coverage: float


def threshold_aware_state_edges(values: Sequence[float], n_bins: int, threshold: float) -> np.ndarray:
    """Quantile bins with `threshold` forced to be an exact boundary."""
    edges = tl.quantile_edges(values, n_bins=n_bins, extend=True)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Viability threshold must be finite.")
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("No finite state observations.")
    if finite_values.min() < threshold < finite_values.max():
        edges = np.sort(np.unique(np.append(edges, threshold)))
    elif np.isclose(finite_values.min(), threshold) or np.isclose(finite_values.max(), threshold):
        edges = np.sort(np.unique(np.append(edges, threshold)))
    return edges


def admissible_mask_from_edges(state_edges: Sequence[float], threshold: float) -> np.ndarray:
    """Mark state bins lying wholly at or above the admissibility threshold."""
    edges = np.asarray(state_edges, dtype=float)
    threshold = float(threshold)
    if not np.any(np.isclose(edges[np.isfinite(edges)], threshold)):
        raise ValueError("Threshold must be an explicit state-bin edge.")
    return edges[:-1] >= threshold - 1e-12


def fixed_nu_marginal_with_coverage(
    gen: tl.GeneratorEstimate,
    nu: Mapping[tuple[int, ...], float],
    *,
    min_regime_coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate each state row under fixed `nu`, refusing weakly supported rows.

    A row is returned only if the available perturbation regimes account for at
    least `min_regime_coverage` of the declared reference law. Available mass is
    renormalized only after the coverage gate is passed. Per-row coverage is
    returned so the approximation is visible rather than silent.
    """
    n = gen.n_states
    K = np.full((n, n), np.nan, dtype=float)
    coverage = np.zeros(n, dtype=float)
    total_nu = float(sum(float(w) for w in nu.values() if np.isfinite(w) and w > 0))
    if total_nu <= 0:
        return K, coverage

    for i in range(n):
        acc = np.zeros(n, dtype=float)
        mass = 0.0
        for z, w in nu.items():
            w = float(w)
            if not np.isfinite(w) or w <= 0:
                continue
            kz = gen.kernels.get(z)
            if kz is None or i >= kz.shape[0] or not np.all(np.isfinite(kz[i])):
                continue
            acc += w * kz[i]
            mass += w
        coverage[i] = mass / total_nu
        if coverage[i] + 1e-12 < float(min_regime_coverage) or mass <= 0:
            continue
        row = acc / mass
        if row.sum() > 0:
            K[i] = row / row.sum()
    return K, coverage


def paired_conditioned_marginal_tv(
    a: tl.GeneratorEstimate,
    b: tl.GeneratorEstimate,
    *,
    nu: Mapping[tuple[int, ...], float],
    state_weights: Sequence[float],
) -> PairedDistance:
    """Compare two generators using one common evaluation measure.

    For each state, both the conditioned distance and the two marginalized rows
    use exactly the same common perturbation support and the same normalized
    restriction of `nu`. State aggregation then uses the same fixed state
    weights. Therefore marginal TV cannot exceed conditioned TV except for
    floating-point noise.
    """
    mu = np.asarray(state_weights, dtype=float)
    if len(mu) != a.n_states or a.n_states != b.n_states:
        raise ValueError("State-weight dimension mismatch.")
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, 0.0)
    total_mu = float(mu.sum())
    total_nu = float(sum(float(w) for w in nu.values() if np.isfinite(w) and w > 0))
    if total_mu <= 0 or total_nu <= 0:
        return PairedDistance(np.nan, np.nan, np.nan, 0.0)

    conditioned_sum = 0.0
    marginal_sum = 0.0
    state_mass = 0.0
    support_mass_weighted = 0.0

    for i, mu_i in enumerate(mu):
        if mu_i <= 0:
            continue
        common = []
        support_mass = 0.0
        for z, w in nu.items():
            w = float(w)
            if not np.isfinite(w) or w <= 0:
                continue
            ka = a.kernels.get(z)
            kb = b.kernels.get(z)
            if ka is None or kb is None:
                continue
            if i >= ka.shape[0] or i >= kb.shape[0]:
                continue
            if not (np.all(np.isfinite(ka[i])) and np.all(np.isfinite(kb[i]))):
                continue
            common.append((z, w, ka[i], kb[i]))
            support_mass += w
        if support_mass <= 0:
            continue

        pa = np.zeros(a.n_states, dtype=float)
        pb = np.zeros(a.n_states, dtype=float)
        cond_i = 0.0
        for _z, w, row_a, row_b in common:
            wn = w / support_mass
            cond_i += wn * tl.total_variation_rows(row_a, row_b)
            pa += wn * row_a
            pb += wn * row_b
        marg_i = tl.total_variation_rows(pa, pb)

        conditioned_sum += mu_i * cond_i
        marginal_sum += mu_i * marg_i
        state_mass += mu_i
        support_mass_weighted += mu_i * (support_mass / total_nu)

    if state_mass <= 0:
        return PairedDistance(np.nan, np.nan, np.nan, 0.0)

    conditioned = conditioned_sum / state_mass
    marginal = marginal_sum / state_mass
    gap = conditioned - marginal
    if gap < -1e-9:
        raise RuntimeError(
            f"Aliasing-gap invariant violated: conditioned={conditioned}, marginal={marginal}."
        )
    if gap < 0:
        gap = 0.0
    coverage = support_mass_weighted / total_mu
    return PairedDistance(conditioned, marginal, gap, coverage)


def run_empirical_windowed_analysis(
    df: pd.DataFrame,
    *,
    time_col: str,
    state_col: str,
    condition_cols: Sequence[str],
    reference_df: pd.DataFrame,
    state_edges: np.ndarray,
    condition_edges: Mapping[str, np.ndarray],
    lag_steps: int,
    min_row_count: int,
    window_days: float,
    step_days: float,
    admissible_threshold: float,
    horizon_steps: int,
    min_regime_coverage: float = 0.90,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, tl.GeneratorEstimate], dict]:
    """Run the primary static TL analysis and retain window generators."""
    work = df.copy()
    ref = reference_df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    ref[time_col] = pd.to_datetime(ref[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    ref = ref.dropna(subset=[time_col]).sort_values(time_col)

    ref_gen = tl.estimate_generator(
        ref,
        time_col=time_col,
        state_col=state_col,
        condition_cols=condition_cols,
        state_edges=state_edges,
        condition_edges=condition_edges,
        lag_steps=lag_steps,
        min_row_count=min_row_count,
    )
    nu_ref = tl.empirical_regime_weights(ref_gen)
    mu_ref = tl.initial_state_distribution(ref[state_col].to_numpy(), state_edges)
    reps = tl.state_midpoints(state_edges, pd.concat([work[state_col], ref[state_col]]).to_numpy())
    admissible = admissible_mask_from_edges(state_edges, admissible_threshold)

    K_ref, K_ref_cov = fixed_nu_marginal_with_coverage(
        ref_gen, nu_ref, min_regime_coverage=min_regime_coverage
    )

    idx = pd.DatetimeIndex(work[time_col])
    rows = []
    generators: dict[pd.Timestamp, tl.GeneratorEstimate] = {}
    for start, end in tl._window_ranges(idx, window_days, step_days):
        w = work[(work[time_col] >= start) & (work[time_col] < end)].copy()
        if len(w) < max(100, min_row_count * 4):
            continue
        gen = tl.estimate_generator(
            w,
            time_col=time_col,
            state_col=state_col,
            condition_cols=condition_cols,
            state_edges=state_edges,
            condition_edges=condition_edges,
            lag_steps=lag_steps,
            min_row_count=min_row_count,
        )
        center = start + (end - start) / 2
        generators[pd.Timestamp(center)] = gen

        paired = paired_conditioned_marginal_tv(
            ref_gen, gen, nu=nu_ref, state_weights=mu_ref
        )
        K_w, row_cov = fixed_nu_marginal_with_coverage(
            gen, nu_ref, min_regime_coverage=min_regime_coverage
        )
        vf = tl.viable_future_summary(K_w, mu_ref, admissible, horizon_steps)
        expected_do = tl.expected_payoff(K_w, mu_ref, reps)
        state_vals = pd.to_numeric(w[state_col], errors="coerce")

        finite_rows = np.all(np.isfinite(K_w), axis=1)
        weighted_kernel_coverage = float(np.sum(mu_ref * row_cov))
        fully_evaluable_state_weight = float(np.sum(mu_ref[finite_rows]))
        ref_fully_evaluable_state_weight = float(np.sum(mu_ref[np.all(np.isfinite(K_ref), axis=1)]))

        rows.append({
            "window_start": start,
            "window_end": end,
            "window_center": center,
            "n_rows": len(w),
            "n_transitions": gen.n_transitions,
            "state_mean": float(state_vals.mean()),
            "state_median": float(state_vals.median()),
            "state_min": float(state_vals.min()),
            "state_last": float(state_vals.dropna().iloc[-1]) if state_vals.notna().any() else np.nan,
            "conditioned_tv_to_reference": paired.conditioned_tv,
            "marginal_tv_to_reference": paired.marginal_tv,
            "aliasing_gap": paired.aliasing_gap,
            "joint_support_coverage": paired.joint_support_coverage,
            "fixed_nu_weighted_coverage": weighted_kernel_coverage,
            "fully_evaluable_state_weight": fully_evaluable_state_weight,
            "reference_fully_evaluable_state_weight": ref_fully_evaluable_state_weight,
            "expected_do_1step_mg_l": expected_do,
            **vf,
        })

    context = {
        "reference_generator": ref_gen,
        "nu_ref": nu_ref,
        "mu_ref": mu_ref,
        "state_representatives": reps,
        "admissible_mask": admissible,
        "reference_marginal_kernel": K_ref,
        "reference_row_coverage": K_ref_cov,
        "min_regime_coverage": float(min_regime_coverage),
    }
    return pd.DataFrame(rows), generators, context


def find_direct_state_similar_architecture_different(
    metrics: pd.DataFrame,
    generators: Mapping[pd.Timestamp, tl.GeneratorEstimate],
    *,
    nu: Mapping[tuple[int, ...], float],
    state_weights: Sequence[float],
    state_tolerance: float = 0.25,
    min_architecture_distance: float = 0.15,
    min_separation_days: float = 30.0,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """Find similar-state windows separated directly in generator space."""
    if metrics.empty:
        return pd.DataFrame()
    m = metrics.copy().reset_index(drop=True)
    m["window_center"] = pd.to_datetime(m["window_center"], errors="coerce", utc=True)
    rows = []
    for i in range(len(m)):
        ti = pd.Timestamp(m.loc[i, "window_center"])
        gi = generators.get(ti)
        if gi is None:
            continue
        for j in range(i + 1, len(m)):
            tj = pd.Timestamp(m.loc[j, "window_center"])
            if abs((tj - ti).total_seconds()) < float(min_separation_days) * 86400:
                continue
            ds = abs(float(m.loc[i, "state_mean"]) - float(m.loc[j, "state_mean"]))
            if not np.isfinite(ds) or ds > float(state_tolerance):
                continue
            gj = generators.get(tj)
            if gj is None:
                continue
            d = paired_conditioned_marginal_tv(gi, gj, nu=nu, state_weights=state_weights)
            if np.isfinite(d.conditioned_tv) and d.conditioned_tv >= float(min_architecture_distance):
                vm_i = m.loc[i, "viable_mass"] if "viable_mass" in m else np.nan
                vm_j = m.loc[j, "viable_mass"] if "viable_mass" in m else np.nan
                rows.append({
                    "i": i,
                    "j": j,
                    "time_i": ti,
                    "time_j": tj,
                    "state_difference_mg_l": ds,
                    "direct_conditioned_tv": d.conditioned_tv,
                    "direct_marginal_tv": d.marginal_tv,
                    "direct_aliasing_gap": d.aliasing_gap,
                    "joint_support_coverage": d.joint_support_coverage,
                    "viable_mass_difference": abs(float(vm_i) - float(vm_j)) if np.isfinite(vm_i) and np.isfinite(vm_j) else np.nan,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "i", "j", "time_i", "time_j", "state_difference_mg_l",
            "direct_conditioned_tv", "direct_marginal_tv", "direct_aliasing_gap",
            "joint_support_coverage", "viable_mass_difference",
        ])
    return (
        pd.DataFrame(rows)
        .sort_values(["direct_conditioned_tv", "viable_mass_difference"], ascending=False)
        .head(max_pairs)
        .reset_index(drop=True)
    )
