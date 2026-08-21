from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from analysis import tolerance_landscape as tl


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _filter_primary_rows(df: pd.DataFrame, flag_cols, accepted_flags) -> pd.DataFrame:
    """Keep only declared primary-analysis rows before binning/evaluation.

    Filtering before common-bin construction ensures flagged values do not
    influence the generator geometry, reference state distribution, or payoff
    representatives. Timestamp continuity is checked separately for every
    transition, so filtering cannot bridge across excluded observations.
    """
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
        # One-second tolerance is ample for nominal 15-min SWMP timestamps while
        # refusing to create transitions across QA/QC exclusions or data gaps.
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

    # Primary geometry and evaluation context are estimated only from rows that
    # satisfy the declared QA/QC rule.
    df_primary = _filter_primary_rows(df, flag_cols, accepted_flags)
    ref_primary = _filter_primary_rows(ref, flag_cols, accepted_flags)

    pooled = pd.concat(
        [df_primary[[state_col, *condition_cols]], ref_primary[[state_col, *condition_cols]]],
        ignore_index=True,
    )
    state_edges, condition_edges = tl.build_common_edges(
        pooled,
        state_col=state_col,
        condition_cols=condition_cols,
        state_bins=int(cfg["binning"]["state_bins"]),
        condition_bins=cfg["binning"]["condition_bins"],
    )

    # Patch the internal transition constructor for this empirical run so that
    # row filtering can never turn a temporal gap into a false Markov step.
    sample_minutes = float(cfg["transition"].get("sample_minutes", 15.0))
    tl._prepare_transition_table = _strict_transition_preparer(sample_minutes)

    metrics = tl.run_windowed_tolerance_analysis(
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
        admissible_threshold=float(cfg["viability"]["oxygen_threshold_mg_l"]),
        horizon_steps=int(cfg["viability"]["horizon_steps"]),
        observed_flag_cols=flag_cols,
        accepted_flags=accepted_flags,
    )
    metrics.to_csv(output_dir / "tolerance_landscape_windows.csv", index=False)

    pairs = tl.find_state_similar_architecture_different(metrics)
    pairs.to_csv(output_dir / "state_similar_architecture_different.csv", index=False)

    edges_payload = {
        "state_edges": [None if not pd.notna(v) or np.isinf(v) else float(v) for v in state_edges],
        "condition_edges": {
            k: [None if not pd.notna(v) or np.isinf(v) else float(v) for v in arr]
            for k, arr in condition_edges.items()
        },
        "transition_sample_minutes": sample_minutes,
        "transition_lag_steps": int(cfg["transition"]["lag_steps"]),
        "transition_gap_bridging_allowed": False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical Tolerance Landscape analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
