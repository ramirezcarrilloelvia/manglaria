from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from analysis import tolerance_landscape as tl
from analysis import tolerance_landscape_empirical as emp


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _filter_primary_rows(df: pd.DataFrame, flag_cols, accepted_flags) -> pd.DataFrame:
    """Keep only declared primary-analysis rows before binning/evaluation."""
    if not flag_cols:
        return df.copy()
    accepted = set(map(str, accepted_flags))
    keep = np.ones(len(df), dtype=bool)
    for col in flag_cols:
        if col not in df.columns:
            raise ValueError(f"Observed-only flag column is missing: {col}")
        keep &= df[col].astype(str).isin(accepted).to_numpy()
    return df.loc[keep].copy()


def _strict_transition_preparer(expected_sample_minutes: float):
    """Return a transition-table builder that rejects transitions across gaps."""
    expected_sample_minutes = float(expected_sample_minutes)

    def strict_prepare(
        df: pd.DataFrame,
        *,
        time_col: str,
        state_col: str,
        condition_cols,
        state_edges,
        condition_edges,
        lag_steps: int,
        observed_flag_cols=None,
        accepted_flags=("observed",),
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

        x_now = tl.digitize(work[state_col].to_numpy(), state_edges)
        x_next = np.roll(x_now, -lag_steps)
        z_bins = [tl.digitize(work[c].to_numpy(), condition_edges[c]) for c in condition_cols]

        n = len(work) - lag_steps
        t0 = work[time_col].iloc[:n].reset_index(drop=True)
        t1 = work[time_col].iloc[lag_steps:].reset_index(drop=True)
        dt_minutes = (t1 - t0).dt.total_seconds().div(60.0)
        expected = expected_sample_minutes * int(lag_steps)
        valid_dt = np.isclose(dt_minutes.to_numpy(), expected, atol=1.0 / 60.0, rtol=0.0)

        out = pd.DataFrame({
            "timestamp": t0.to_numpy(),
            "x": x_now[:n],
            "y": x_next[:n],
            "transition_minutes": dt_minutes.to_numpy(),
        })
        for c, zb in zip(condition_cols, z_bins):
            out[f"z__{c}"] = zb[:n]
        out = out[valid_dt & (out["x"] >= 0) & (out["y"] >= 0)].reset_index(drop=True)
        return out

    return strict_prepare


def run(config_path: str | Path):
    cfg = load_config(config_path)
    input_csv = Path(cfg["input_csv"])
    reference_csv = Path(cfg["reference_csv"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    ref = pd.read_csv(reference_csv)

    time_col = cfg["time_col"]
    state_col = cfg["state_col"]
    condition_cols = cfg["condition_cols"]

    obs_cfg = cfg.get("observed_only", {})
    flag_cols = obs_cfg.get("flag_cols") or None
    accepted_flags = obs_cfg.get("accepted_flags", ["observed"])

    # Flagged rows cannot influence bin edges, reference state weights or
    # generator estimation in the primary run.
    df_primary = _filter_primary_rows(df, flag_cols, accepted_flags)
    ref_primary = _filter_primary_rows(ref, flag_cols, accepted_flags)

    pooled = pd.concat(
        [df_primary[[state_col, *condition_cols]], ref_primary[[state_col, *condition_cols]]],
        ignore_index=True,
    )
    threshold = float(cfg["viability"]["oxygen_threshold_mg_l"])
    state_edges = emp.threshold_aware_state_edges(
        pooled[state_col].to_numpy(),
        n_bins=int(cfg["binning"]["state_bins"]),
        threshold=threshold,
    )
    condition_edges = {}
    cond_cfg = cfg["binning"]["condition_bins"]
    for col in condition_cols:
        n_bins = cond_cfg[col] if isinstance(cond_cfg, dict) else cond_cfg
        condition_edges[col] = tl.quantile_edges(pooled[col].to_numpy(), int(n_bins))

    # Prevent QA exclusions or missing timestamps from being bridged into a
    # false one-hour Markov transition.
    sample_minutes = float(cfg["transition"].get("sample_minutes", 15.0))
    tl._prepare_transition_table = _strict_transition_preparer(sample_minutes)

    min_regime_coverage = float(cfg.get("evaluation", {}).get("min_regime_coverage", 0.90))
    metrics, generators, context = emp.run_empirical_windowed_analysis(
        df_primary,
        time_col=time_col,
        state_col=state_col,
        condition_cols=condition_cols,
        reference_df=ref_primary,
        state_edges=state_edges,
        condition_edges=condition_edges,
        lag_steps=int(cfg["transition"]["lag_steps"]),
        min_row_count=int(cfg["transition"]["min_row_count"]),
        window_days=float(cfg["windows"]["window_days"]),
        step_days=float(cfg["windows"]["step_days"]),
        admissible_threshold=threshold,
        horizon_steps=int(cfg["viability"]["horizon_steps"]),
        min_regime_coverage=min_regime_coverage,
    )
    metrics.to_csv(output_dir / "tolerance_landscape_windows.csv", index=False)

    pair_cfg = cfg.get("falsification", {})
    pairs = emp.find_direct_state_similar_architecture_different(
        metrics,
        generators,
        nu=context["nu_ref"],
        state_weights=context["mu_ref"],
        state_tolerance=float(pair_cfg.get("state_tolerance_mg_l", 0.25)),
        min_architecture_distance=float(pair_cfg.get("min_direct_conditioned_tv", 0.15)),
        min_separation_days=float(pair_cfg.get("min_pair_separation_days", 30.0)),
        max_pairs=int(pair_cfg.get("max_pairs", 50)),
    )
    pairs.to_csv(output_dir / "state_similar_architecture_different.csv", index=False)

    edges_payload = {
        "state_edges": [None if not pd.notna(v) or np.isinf(v) else float(v) for v in state_edges],
        "condition_edges": {
            k: [None if not pd.notna(v) or np.isinf(v) else float(v) for v in arr]
            for k, arr in condition_edges.items()
        },
        "oxygen_threshold_is_explicit_state_edge": bool(
            np.any(np.isclose(state_edges[np.isfinite(state_edges)], threshold))
        ),
        "oxygen_threshold_mg_l": threshold,
        "transition_sample_minutes": sample_minutes,
        "transition_lag_steps": int(cfg["transition"]["lag_steps"]),
        "transition_gap_bridging_allowed": False,
        "fixed_nu_min_regime_coverage": min_regime_coverage,
        "reference_nu": {str(k): float(v) for k, v in context["nu_ref"].items()},
        "reference_state_weights": [float(v) for v in context["mu_ref"]],
        "state_representatives_mg_l": [None if not np.isfinite(v) else float(v) for v in context["state_representatives"]],
        "admissible_state_mask": [bool(v) for v in context["admissible_mask"]],
    }
    with open(output_dir / "binning.json", "w", encoding="utf-8") as f:
        json.dump(edges_payload, f, indent=2)

    tmsi_csv = cfg.get("complexity_bridge", {}).get("tmsi_csv")
    if tmsi_csv:
        merged = tl.merge_tmsi_metrics(
            metrics,
            tmsi_csv,
            tolerance=cfg.get("complexity_bridge", {}).get("merge_tolerance", "3D"),
        )
        merged.to_csv(output_dir / "tolerance_landscape_plus_tmsi.csv", index=False)

    print(f"Tolerance-landscape analysis complete: {output_dir}")
    print(f"Primary rows: {len(df_primary):,}; reference rows: {len(ref_primary):,}")
    print(f"Valid observational windows: {len(metrics):,}")
    print(f"Direct same-state/different-architecture pairs: {len(pairs):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical Tolerance Landscape analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
