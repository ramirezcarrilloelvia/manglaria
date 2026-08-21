from __future__ import annotations

"""Empirical tolerance-landscape utilities for ecological time series.

The implementation is intentionally conservative:
- generators are estimated separately in observational windows;
- no dynamics on generator space are assumed;
- all comparisons use a common discretization and a declared evaluation law;
- observed-only data can be required so gap-filled values do not silently alter transitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class GeneratorEstimate:
    """Finite perturbation-conditioned Markov generator estimate."""

    kernels: dict[tuple[int, ...], np.ndarray]
    row_counts: dict[tuple[tuple[int, ...], int], int]
    regime_counts: dict[tuple[int, ...], int]
    state_edges: np.ndarray
    condition_edges: dict[str, np.ndarray]
    n_transitions: int

    @property
    def n_states(self) -> int:
        return max(len(self.state_edges) - 1, 0)


def _strict_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    edges = np.unique(edges[np.isfinite(edges)])
    if edges.size < 2:
        raise ValueError("At least two distinct finite bin edges are required.")
    return edges


def quantile_edges(values: Sequence[float], n_bins: int, *, extend: bool = True) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(4, n_bins):
        raise ValueError("Insufficient finite observations to define quantile bins.")
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = _strict_edges(np.quantile(x, q))
    if edges.size < 3:
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, n_bins + 1)
    if extend:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges


def digitize(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, -1, dtype=int)
    mask = np.isfinite(x)
    out[mask] = np.digitize(x[mask], edges[1:-1], right=False)
    return out


def build_common_edges(
    df: pd.DataFrame,
    state_col: str,
    condition_cols: Sequence[str],
    state_bins: int = 6,
    condition_bins: int | Mapping[str, int] = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state_edges = quantile_edges(df[state_col].to_numpy(), state_bins)
    cond_edges: dict[str, np.ndarray] = {}
    for col in condition_cols:
        n = condition_bins[col] if isinstance(condition_bins, Mapping) else condition_bins
        cond_edges[col] = quantile_edges(df[col].to_numpy(), int(n))
    return state_edges, cond_edges


def _prepare_transition_table(
    df: pd.DataFrame,
    *,
    time_col: str,
    state_col: str,
    condition_cols: Sequence[str],
    state_edges: np.ndarray,
    condition_edges: Mapping[str, np.ndarray],
    lag_steps: int,
    observed_flag_cols: Sequence[str] | None = None,
    accepted_flags: Iterable[str] = ("observed",),
) -> pd.DataFrame:
    cols = [time_col, state_col, *condition_cols]
    if observed_flag_cols:
        cols += [c for c in observed_flag_cols if c in df.columns]
    work = df[cols].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if observed_flag_cols:
        accepted = set(map(str, accepted_flags))
        keep = np.ones(len(work), dtype=bool)
        for flag_col in observed_flag_cols:
            if flag_col in work.columns:
                keep &= work[flag_col].astype(str).isin(accepted).to_numpy()
        work = work.loc[keep].reset_index(drop=True)

    needed = [state_col, *condition_cols]
    for col in needed:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=needed).reset_index(drop=True)
    if len(work) <= lag_steps:
        return pd.DataFrame()

    x_now = digitize(work[state_col].to_numpy(), state_edges)
    x_next = np.roll(x_now, -lag_steps)
    z_bins = [digitize(work[c].to_numpy(), condition_edges[c]) for c in condition_cols]

    n = len(work) - lag_steps
    out = pd.DataFrame({
        "timestamp": work[time_col].iloc[:n].to_numpy(),
        "x": x_now[:n],
        "y": x_next[:n],
    })
    for c, zb in zip(condition_cols, z_bins):
        out[f"z__{c}"] = zb[:n]
    out = out[(out["x"] >= 0) & (out["y"] >= 0)].reset_index(drop=True)
    return out


def estimate_generator(
    df: pd.DataFrame,
    *,
    time_col: str,
    state_col: str,
    condition_cols: Sequence[str],
    state_edges: np.ndarray,
    condition_edges: Mapping[str, np.ndarray],
    lag_steps: int = 4,
    min_row_count: int = 20,
    pseudocount: float = 0.0,
    observed_flag_cols: Sequence[str] | None = None,
    accepted_flags: Iterable[str] = ("observed",),
) -> GeneratorEstimate:
    """Estimate K_z(i,j) from observed transitions using fixed bins."""
    trans = _prepare_transition_table(
        df,
        time_col=time_col,
        state_col=state_col,
        condition_cols=condition_cols,
        state_edges=state_edges,
        condition_edges=condition_edges,
        lag_steps=lag_steps,
        observed_flag_cols=observed_flag_cols,
        accepted_flags=accepted_flags,
    )
    n_states = len(state_edges) - 1
    kernels: dict[tuple[int, ...], np.ndarray] = {}
    row_counts: dict[tuple[tuple[int, ...], int], int] = {}
    regime_counts: dict[tuple[int, ...], int] = {}
    z_cols = [f"z__{c}" for c in condition_cols]

    if trans.empty:
        return GeneratorEstimate(kernels, row_counts, regime_counts, state_edges, dict(condition_edges), 0)

    for z_vals, grp in trans.groupby(z_cols, dropna=False):
        if not isinstance(z_vals, tuple):
            z_vals = (int(z_vals),)
        regime = tuple(int(v) for v in z_vals)
        counts = np.full((n_states, n_states), float(pseudocount), dtype=float)
        regime_counts[regime] = int(len(grp))
        for (i, j), n in grp.groupby(["x", "y"]).size().items():
            counts[int(i), int(j)] += int(n)
        K = np.full_like(counts, np.nan, dtype=float)
        for i in range(n_states):
            n_i = int((grp["x"] == i).sum())
            row_counts[(regime, i)] = n_i
            if n_i >= min_row_count:
                row_sum = counts[i].sum()
                if row_sum > 0:
                    K[i] = counts[i] / row_sum
        kernels[regime] = K

    return GeneratorEstimate(
        kernels=kernels,
        row_counts=row_counts,
        regime_counts=regime_counts,
        state_edges=np.asarray(state_edges, dtype=float),
        condition_edges={k: np.asarray(v, dtype=float) for k, v in condition_edges.items()},
        n_transitions=int(len(trans)),
    )


def empirical_regime_weights(gen: GeneratorEstimate) -> dict[tuple[int, ...], float]:
    total = float(sum(gen.regime_counts.values()))
    if total <= 0:
        return {}
    return {z: n / total for z, n in gen.regime_counts.items()}


def empirical_row_weights(gen: GeneratorEstimate) -> dict[tuple[tuple[int, ...], int], float]:
    valid = {key: n for key, n in gen.row_counts.items() if n > 0}
    total = float(sum(valid.values()))
    if total <= 0:
        return {}
    return {key: n / total for key, n in valid.items()}


def total_variation_rows(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def conditioned_tv_distance(
    a: GeneratorEstimate,
    b: GeneratorEstimate,
    *,
    evaluation_weights: Mapping[tuple[tuple[int, ...], int], float] | None = None,
) -> tuple[float, float]:
    """Weighted TV distance across perturbation-conditioned kernel rows.

    Returns (distance, evaluated_weight_fraction). Rows unavailable in either
    estimate are excluded and the remaining evaluation weights are renormalized.
    """
    weights = dict(evaluation_weights or empirical_row_weights(a))
    numer = 0.0
    denom = 0.0
    total_weight = float(sum(weights.values()))
    for (z, i), w in weights.items():
        if z not in a.kernels or z not in b.kernels:
            continue
        pa, pb = a.kernels[z][i], b.kernels[z][i]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        numer += float(w) * total_variation_rows(pa, pb)
        denom += float(w)
    if denom <= 0:
        return np.nan, 0.0
    coverage = denom / total_weight if total_weight > 0 else 0.0
    return numer / denom, coverage


def marginal_kernel(
    gen: GeneratorEstimate,
    regime_weights: Mapping[tuple[int, ...], float],
) -> np.ndarray:
    """Integrate K_z over a fixed empirical perturbation law nu."""
    n = gen.n_states
    out = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        acc = np.zeros(n, dtype=float)
        mass = 0.0
        for z, w in regime_weights.items():
            K = gen.kernels.get(z)
            if K is None or not np.all(np.isfinite(K[i])):
                continue
            acc += float(w) * K[i]
            mass += float(w)
        if mass > 0:
            row = acc / mass
            s = row.sum()
            if s > 0:
                out[i] = row / s
    return out


def marginal_tv_distance(a: np.ndarray, b: np.ndarray, row_weights: Sequence[float] | None = None) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    weights = np.ones(n, dtype=float) / n if row_weights is None else np.asarray(row_weights, dtype=float)
    numer = 0.0
    denom = 0.0
    total = float(np.nansum(weights))
    for i in range(n):
        if np.all(np.isfinite(a[i])) and np.all(np.isfinite(b[i])) and np.isfinite(weights[i]):
            numer += weights[i] * total_variation_rows(a[i], b[i])
            denom += weights[i]
    if denom <= 0:
        return np.nan, 0.0
    return numer / denom, denom / total if total > 0 else 0.0


def state_midpoints(edges: Sequence[float], observed_values: Sequence[float]) -> np.ndarray:
    """Finite representative value per state bin, using observed bin medians."""
    edges = np.asarray(edges, dtype=float)
    x = np.asarray(observed_values, dtype=float)
    bins = digitize(x, edges)
    reps = np.full(len(edges) - 1, np.nan)
    for i in range(len(reps)):
        vals = x[(bins == i) & np.isfinite(x)]
        if len(vals):
            reps[i] = np.median(vals)
    finite = np.isfinite(reps)
    if not finite.all() and finite.any():
        reps[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), reps[finite])
    return reps


def initial_state_distribution(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    bins = digitize(values, edges)
    n = len(edges) - 1
    counts = np.bincount(bins[bins >= 0], minlength=n).astype(float)
    if counts.sum() <= 0:
        return np.ones(n) / n
    return counts / counts.sum()


def expected_payoff(K: np.ndarray, init: Sequence[float], payoff: Sequence[float]) -> float:
    K = np.asarray(K, dtype=float)
    init = np.asarray(init, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    if not np.all(np.isfinite(K)):
        return np.nan
    return float(init @ K @ payoff)


def viable_future_summary(
    K: np.ndarray,
    init: Sequence[float],
    admissible_mask: Sequence[bool],
    horizon_steps: int,
) -> dict[str, float]:
    """Finite-horizon viable mass and conditional terminal-state diversity."""
    K = np.asarray(K, dtype=float)
    init = np.asarray(init, dtype=float)
    adm = np.asarray(admissible_mask, dtype=bool)
    if K.shape[0] != K.shape[1] or len(init) != K.shape[0] or len(adm) != K.shape[0]:
        raise ValueError("Dimension mismatch in viable_future_summary.")
    if not np.all(np.isfinite(K)):
        return {"viable_mass": np.nan, "terminal_entropy": np.nan, "effective_terminal_states": np.nan}
    idx = np.flatnonzero(adm)
    if idx.size == 0:
        return {"viable_mass": 0.0, "terminal_entropy": np.nan, "effective_terminal_states": 0.0}
    q0 = init[idx].astype(float)
    Q = K[np.ix_(idx, idx)]
    q = q0.copy()
    for _ in range(int(horizon_steps)):
        q = q @ Q
    mass = float(q.sum())
    if mass <= EPS:
        return {"viable_mass": mass, "terminal_entropy": np.nan, "effective_terminal_states": 0.0}
    p = q / mass
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return {
        "viable_mass": mass,
        "terminal_entropy": entropy,
        "effective_terminal_states": float(np.exp(entropy)),
    }


def _window_ranges(index: pd.DatetimeIndex, window_days: float, step_days: float):
    if len(index) == 0:
        return
    start = index.min()
    last = index.max()
    window = pd.Timedelta(days=float(window_days))
    step = pd.Timedelta(days=float(step_days))
    while start + window <= last + pd.Timedelta(seconds=1):
        end = start + window
        yield start, end
        start = start + step


def run_windowed_tolerance_analysis(
    df: pd.DataFrame,
    *,
    time_col: str,
    state_col: str,
    condition_cols: Sequence[str],
    reference_df: pd.DataFrame,
    state_edges: np.ndarray,
    condition_edges: Mapping[str, np.ndarray],
    lag_steps: int = 4,
    min_row_count: int = 20,
    window_days: float = 60.0,
    step_days: float = 7.0,
    admissible_threshold: float = 3.0,
    horizon_steps: int = 24,
    observed_flag_cols: Sequence[str] | None = None,
    accepted_flags: Iterable[str] = ("observed",),
) -> pd.DataFrame:
    """Estimate successive observational-window generators and TL summaries."""
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    ref = reference_df.copy()
    ref[time_col] = pd.to_datetime(ref[time_col], errors="coerce", utc=True)
    ref = ref.dropna(subset=[time_col]).sort_values(time_col)

    ref_gen = estimate_generator(
        ref,
        time_col=time_col,
        state_col=state_col,
        condition_cols=condition_cols,
        state_edges=state_edges,
        condition_edges=condition_edges,
        lag_steps=lag_steps,
        min_row_count=min_row_count,
        observed_flag_cols=observed_flag_cols,
        accepted_flags=accepted_flags,
    )
    nu_ref = empirical_regime_weights(ref_gen)
    row_weights_ref = empirical_row_weights(ref_gen)
    K_ref = marginal_kernel(ref_gen, nu_ref)
    init_ref = initial_state_distribution(ref[state_col].to_numpy(), state_edges)
    reps = state_midpoints(state_edges, pd.concat([work[state_col], ref[state_col]]).to_numpy())
    finite_reps = reps[np.isfinite(reps)]
    if finite_reps.size:
        lo, hi = finite_reps.min(), finite_reps.max()
        payoff = (reps - lo) / (hi - lo) if hi > lo else np.ones_like(reps)
    else:
        payoff = np.arange(len(reps), dtype=float)
    admissible = reps >= float(admissible_threshold)

    idx = pd.DatetimeIndex(work[time_col])
    rows = []
    for start, end in _window_ranges(idx, window_days, step_days):
        w = work[(work[time_col] >= start) & (work[time_col] < end)].copy()
        if len(w) < max(100, min_row_count * 4):
            continue
        gen = estimate_generator(
            w,
            time_col=time_col,
            state_col=state_col,
            condition_cols=condition_cols,
            state_edges=state_edges,
            condition_edges=condition_edges,
            lag_steps=lag_steps,
            min_row_count=min_row_count,
            observed_flag_cols=observed_flag_cols,
            accepted_flags=accepted_flags,
        )
        cond_d, cond_cov = conditioned_tv_distance(ref_gen, gen, evaluation_weights=row_weights_ref)
        K_w_fixed_nu = marginal_kernel(gen, nu_ref)
        marg_d, marg_cov = marginal_tv_distance(K_ref, K_w_fixed_nu, row_weights=init_ref)
        vf = viable_future_summary(K_w_fixed_nu, init_ref, admissible, horizon_steps)
        payoff_1 = expected_payoff(K_w_fixed_nu, init_ref, payoff)
        state_vals = pd.to_numeric(w[state_col], errors="coerce")
        rows.append({
            "window_start": start,
            "window_end": end,
            "window_center": start + (end - start) / 2,
            "n_rows": len(w),
            "n_transitions": gen.n_transitions,
            "state_mean": float(state_vals.mean()),
            "state_median": float(state_vals.median()),
            "state_min": float(state_vals.min()),
            "state_last": float(state_vals.dropna().iloc[-1]) if state_vals.notna().any() else np.nan,
            "conditioned_tv_to_reference": cond_d,
            "conditioned_tv_coverage": cond_cov,
            "marginal_tv_to_reference": marg_d,
            "marginal_tv_coverage": marg_cov,
            "aliasing_excess": cond_d - marg_d if np.isfinite(cond_d) and np.isfinite(marg_d) else np.nan,
            "expected_payoff_1step": payoff_1,
            **vf,
        })
    return pd.DataFrame(rows)


def merge_tmsi_metrics(
    tl_metrics: pd.DataFrame,
    tmsi_csv: str | Path,
    *,
    tolerance: str = "3D",
) -> pd.DataFrame:
    """Nearest-time merge with A-Frank-Analysis TMSI early-warning output."""
    tmsi = pd.read_csv(tmsi_csv)
    if "window_center" not in tmsi.columns:
        raise ValueError("TMSI CSV must contain window_center.")
    left = tl_metrics.copy()
    left["window_center"] = pd.to_datetime(left["window_center"], errors="coerce", utc=True)
    tmsi["window_center"] = pd.to_datetime(tmsi["window_center"], errors="coerce", utc=True)
    left = left.dropna(subset=["window_center"]).sort_values("window_center")
    tmsi = tmsi.dropna(subset=["window_center"]).sort_values("window_center")
    return pd.merge_asof(
        left,
        tmsi,
        on="window_center",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
        suffixes=("_tl", "_tmsi"),
    )


def find_state_similar_architecture_different(
    metrics: pd.DataFrame,
    *,
    state_tolerance: float = 0.25,
    min_architecture_distance: float = 0.15,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """Find window pairs with similar state but separated TL displacement."""
    rows = []
    m = metrics.reset_index(drop=True)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            ds = abs(m.loc[i, "state_mean"] - m.loc[j, "state_mean"])
            da = abs(m.loc[i, "conditioned_tv_to_reference"] - m.loc[j, "conditioned_tv_to_reference"])
            if ds <= state_tolerance and da >= min_architecture_distance:
                rows.append({
                    "i": i,
                    "j": j,
                    "time_i": m.loc[i, "window_center"],
                    "time_j": m.loc[j, "window_center"],
                    "state_difference": ds,
                    "architecture_displacement_difference": da,
                    "viable_mass_difference": abs(m.loc[i, "viable_mass"] - m.loc[j, "viable_mass"]),
                })
    if not rows:
        return pd.DataFrame(columns=["i", "j", "time_i", "time_j", "state_difference", "architecture_displacement_difference", "viable_mass_difference"])
    out = pd.DataFrame(rows).sort_values(
        ["architecture_displacement_difference", "viable_mass_difference"], ascending=False
    )
    return out.head(max_pairs).reset_index(drop=True)
