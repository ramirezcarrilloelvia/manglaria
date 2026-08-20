from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd
import yaml

from analysis.tolerance_landscape import (
    build_common_edges,
    find_state_similar_architecture_different,
    merge_tmsi_metrics,
    run_windowed_tolerance_analysis,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    pooled = pd.concat([df[[state_col, *condition_cols]], ref[[state_col, *condition_cols]]], ignore_index=True)
    state_edges, condition_edges = build_common_edges(
        pooled,
        state_col=state_col,
        condition_cols=condition_cols,
        state_bins=int(cfg["binning"]["state_bins"]),
        condition_bins=cfg["binning"]["condition_bins"],
    )

    obs_cfg = cfg.get("observed_only", {})
    flag_cols = obs_cfg.get("flag_cols") or None
    accepted_flags = obs_cfg.get("accepted_flags", ["observed"])

    metrics = run_windowed_tolerance_analysis(
        df,
        time_col=time_col,
        state_col=state_col,
        condition_cols=condition_cols,
        reference_df=ref,
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

    pairs = find_state_similar_architecture_different(metrics)
    pairs.to_csv(output_dir / "state_similar_architecture_different.csv", index=False)

    edges_payload = {
        "state_edges": [None if not pd.notna(v) or v in [float("inf"), float("-inf")] else float(v) for v in state_edges],
        "condition_edges": {
            k: [None if not pd.notna(v) or v in [float("inf"), float("-inf")] else float(v) for v in arr]
            for k, arr in condition_edges.items()
        },
    }
    with open(output_dir / "binning.json", "w", encoding="utf-8") as f:
        json.dump(edges_payload, f, indent=2)

    tmsi_csv = cfg.get("complexity_bridge", {}).get("tmsi_csv")
    if tmsi_csv:
        merged = merge_tmsi_metrics(
            metrics,
            tmsi_csv,
            tolerance=cfg.get("complexity_bridge", {}).get("merge_tolerance", "3D"),
        )
        merged.to_csv(output_dir / "tolerance_landscape_plus_tmsi.csv", index=False)

    print(f"Tolerance-landscape analysis complete: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical Tolerance Landscape analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
