from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_output_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_runs(mask: np.ndarray):
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out["hour"] = out[time_col].dt.hour
    out["minute"] = out[time_col].dt.minute
    out["dayofyear"] = out[time_col].dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60.0)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def regularize_timeseries(df: pd.DataFrame, time_col: str, freq_minutes: int) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.dropna(subset=[time_col]).sort_values(time_col)

    if out[time_col].duplicated().any():
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = [c for c in out.columns if c not in numeric_cols and c != time_col]
        g_num = out.groupby(time_col, as_index=False)[numeric_cols].mean()
        if other_cols:
            g_other = out.groupby(time_col, as_index=False)[other_cols].first()
            out = g_num.merge(g_other, on=time_col, how="left")
        else:
            out = g_num

    full_index = pd.date_range(
        start=out[time_col].min(),
        end=out[time_col].max(),
        freq=f"{freq_minutes}min",
        tz="UTC",
    )

    out = out.set_index(time_col).reindex(full_index)
    out.index.name = time_col
    out = out.reset_index()
    return out


def classify_structure_combined(
    df: pd.DataFrame,
    vars_for_structure: list[str],
    time_col: str,
    structural_gap_steps: int,
    min_segment_points: int,
):
    availability = df[vars_for_structure].notna().any(axis=1).to_numpy()
    gap_runs = find_runs(~availability)

    segment_id = np.full(len(df), 1.0)
    current = 1
    for start, end in gap_runs:
        gap_len = end - start + 1
        if gap_len > structural_gap_steps and end + 1 < len(df):
            current += 1
            segment_id[end + 1 :] = current

    seg_series = pd.Series(segment_id, index=df.index, name="segment_id")

    rows = []
    for seg in sorted(seg_series.dropna().unique()):
        seg = int(seg)
        idx = np.where(seg_series.values == seg)[0]
        n_points = len(idx)

        rows.append(
            {
                "segment_id": seg,
                "start_idx": int(idx.min()),
                "end_idx": int(idx.max()),
                "n_points": int(n_points),
                "start_time": df.loc[idx.min(), time_col],
                "end_time": df.loc[idx.max(), time_col],
                "duration_hours": float(
                    (df.loc[idx.max(), time_col] - df.loc[idx.min(), time_col]).total_seconds() / 3600.0
                ),
                "action_taken": "main_segment" if n_points >= min_segment_points else "short_segment",
                "is_valid_for_analysis": True,
            }
        )

    return seg_series, pd.DataFrame(rows)


def gap_fill_short_internal_gaps(series: pd.Series, max_gap_steps: int):
    s = series.copy()
    interp = s.interpolate(method="time", limit=max_gap_steps, limit_area="inside")

    flags = pd.Series("observed", index=s.index, dtype="object")
    mask = s.isna().to_numpy()

    for start, end in find_runs(mask):
        gap_len = end - start + 1
        left_bounded = start > 0 and pd.notna(s.iloc[start - 1])
        right_bounded = end < len(s) - 1 and pd.notna(s.iloc[end + 1])

        if left_bounded and right_bounded and gap_len <= max_gap_steps:
            flags.iloc[start : end + 1] = "imputed_time_limited"
        else:
            interp.iloc[start : end + 1] = np.nan
            if left_bounded and right_bounded:
                flags.iloc[start : end + 1] = "left_as_nan_large_gap"
            else:
                flags.iloc[start : end + 1] = "left_as_nan_edge_gap"

    return interp, flags


def _build_predictor_set(df: pd.DataFrame, target_col: str, predictor_cols: list[str], time_col: str):
    out = add_time_features(df, time_col=time_col)
    base = [c for c in predictor_cols if c in out.columns and c != target_col]
    time_feats = [
        "hour",
        "minute",
        "dayofyear",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "doy_sin",
        "doy_cos",
    ]

    cols = []
    seen = set()
    for c in base + time_feats:
        if c not in seen and c in out.columns:
            cols.append(c)
            seen.add(c)
    return out, cols


def rf_gapfill_variable(df: pd.DataFrame, target_col: str, predictor_cols: list[str], time_col: str, rf_params: dict):
    data, X_cols = _build_predictor_set(df, target_col, predictor_cols, time_col)

    observed_mask = data[target_col].notna()
    train_df = data.loc[observed_mask].dropna(subset=X_cols).copy()

    pred_mask = data[target_col].isna()
    pred_df = data.loc[pred_mask].dropna(subset=X_cols).copy()

    flags = pd.Series("observed", index=data.index, name=f"{target_col}_rf_flag", dtype="object")
    out = df.copy()

    if len(train_df) < 30:
        return out, flags

    model = RandomForestRegressor(**rf_params)
    model.fit(train_df[X_cols], train_df[target_col])

    if len(pred_df) > 0:
        y_pred = model.predict(pred_df[X_cols])
        out.loc[pred_df.index, target_col] = y_pred
        flags.loc[pred_df.index] = "imputed_rf"

    missing_predictors_idx = data.loc[pred_mask].index.difference(pred_df.index)
    if len(missing_predictors_idx) > 0:
        flags.loc[missing_predictors_idx] = "left_as_nan_missing_predictors"

    return out, flags


def rf_gapfill_time_only(df: pd.DataFrame, target_col: str, time_col: str, rf_params: dict):
    data = add_time_features(df, time_col=time_col)
    X_cols = [
        "hour",
        "minute",
        "dayofyear",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "doy_sin",
        "doy_cos",
    ]

    observed_mask = data[target_col].notna()
    train_df = data.loc[observed_mask].dropna(subset=X_cols).copy()

    pred_mask = data[target_col].isna()
    pred_df = data.loc[pred_mask].dropna(subset=X_cols).copy()

    flags = pd.Series("observed", index=data.index, name=f"{target_col}_rf_time_flag", dtype="object")
    out = df.copy()

    if len(train_df) < 30:
        return out, flags

    model = RandomForestRegressor(**rf_params)
    model.fit(train_df[X_cols], train_df[target_col])

    if len(pred_df) > 0:
        y_pred = model.predict(pred_df[X_cols])
        out.loc[pred_df.index, target_col] = y_pred
        flags.loc[pred_df.index] = "imputed_rf_time_only"

    return out, flags


def hybrid_gapfill_segment(
    df_segment: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    time_col: str,
    max_interp_gap_steps: int,
    max_rf_gap_steps: int,
    rf_params: dict,
):
    out = df_segment.copy()

    temp = out.set_index(time_col)[target_col].copy()
    interp, interp_flags = gap_fill_short_internal_gaps(temp, max_gap_steps=max_interp_gap_steps)
    out[target_col] = interp.values
    flags = pd.Series(interp_flags.values, index=out.index, name=f"{target_col}_flag", dtype="object")

    remaining_mask = out[target_col].isna().to_numpy()
    remaining_runs = find_runs(remaining_mask)

    rf_candidate_idx = []
    for start, end in remaining_runs:
        gap_len = end - start + 1
        if gap_len <= max_rf_gap_steps:
            rf_candidate_idx.extend(range(start, end + 1))

    if len(rf_candidate_idx) > 0:
        rf_out, rf_flags = rf_gapfill_variable(
            df=out,
            target_col=target_col,
            predictor_cols=predictor_cols,
            time_col=time_col,
            rf_params=rf_params,
        )

        for idx in rf_candidate_idx:
            if pd.isna(out.iloc[idx][target_col]) and pd.notna(rf_out.iloc[idx][target_col]):
                out.iloc[idx, out.columns.get_loc(target_col)] = rf_out.iloc[idx][target_col]
                flags.iloc[idx] = "imputed_rf"

    still_missing = out[target_col].isna()
    if still_missing.any():
        rf_time_out, rf_time_flags = rf_gapfill_time_only(
            df=out,
            target_col=target_col,
            time_col=time_col,
            rf_params=rf_params,
        )

        newly_filled = still_missing & rf_time_out[target_col].notna()
        out.loc[newly_filled, target_col] = rf_time_out.loc[newly_filled, target_col]
        flags.loc[newly_filled] = "imputed_rf_time_only"

    still_missing = out[target_col].isna()
    flags.loc[still_missing & flags.eq("observed")] = "left_as_nan_large_gap"

    return out, flags


def validate_by_masking_hybrid(
    df_segment: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    time_col: str,
    max_interp_gap_steps: int,
    max_rf_gap_steps: int,
    rf_params: dict,
    n_trials: int,
    seed: int,
):
    s = df_segment[target_col].copy()
    values = s.to_numpy()
    rng = np.random.default_rng(seed)

    eligible = []
    for gap_len in range(1, max_rf_gap_steps + 1):
        for start in range(1, len(s) - gap_len - 1):
            block = values[start : start + gap_len]
            if np.all(pd.notna(block)) and pd.notna(values[start - 1]) and pd.notna(values[start + gap_len]):
                eligible.append((start, gap_len))

    if len(eligible) == 0:
        return {
            "variable": target_col,
            "n_trials": 0,
            "n_points_scored": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
        }

    chosen = rng.choice(len(eligible), size=min(n_trials, len(eligible)), replace=False)

    truth_values, pred_values = [], []
    for idx in chosen:
        start, gap_len = eligible[idx]
        sub = df_segment.copy()
        truth = sub.iloc[start : start + gap_len][target_col].copy()
        sub.iloc[start : start + gap_len, sub.columns.get_loc(target_col)] = np.nan

        pred_df, pred_flags = hybrid_gapfill_segment(
            df_segment=sub,
            target_col=target_col,
            predictor_cols=predictor_cols,
            time_col=time_col,
            max_interp_gap_steps=max_interp_gap_steps,
            max_rf_gap_steps=max_rf_gap_steps,
            rf_params=rf_params,
        )

        pred = pred_df.iloc[start : start + gap_len][target_col]
        valid = truth.notna() & pred.notna()

        if valid.any():
            truth_values.extend(truth[valid].tolist())
            pred_values.extend(pred[valid].tolist())

    truth_values = np.asarray(truth_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    if len(truth_values) == 0:
        return {
            "variable": target_col,
            "n_trials": int(len(chosen)),
            "n_points_scored": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
        }

    mae = float(np.mean(np.abs(pred_values - truth_values)))
    rmse = float(np.sqrt(np.mean((pred_values - truth_values) ** 2)))
    bias = float(np.mean(pred_values - truth_values))
    denom = float(np.sum((truth_values - truth_values.mean()) ** 2))
    r2 = float(1 - np.sum((truth_values - pred_values) ** 2) / denom) if denom > 0 else np.nan

    return {
        "variable": target_col,
        "n_trials": int(len(chosen)),
        "n_points_scored": int(len(truth_values)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
    }


def plot_layers_3panel(
    df_regularized: pd.DataFrame,
    df_gapfilled_all: pd.DataFrame,
    df_gapfill_flags: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    time_col: str,
    fig_dpi: int,
):
    raw = df_regularized[[time_col, target_col]].copy().sort_values(time_col).reset_index(drop=True)
    filled = df_gapfilled_all[[time_col, target_col]].copy().sort_values(time_col).reset_index(drop=True)

    flag_col = f"{target_col}_flag"
    if flag_col in df_gapfill_flags.columns:
        imputed_mask = df_gapfill_flags[flag_col].isin(
            ["imputed_time_limited", "imputed_rf", "imputed_rf_time_only"]
        ).to_numpy()
    else:
        imputed_mask = np.zeros(len(filled), dtype=bool)

    imputed = filled.copy()
    imputed.loc[~imputed_mask, target_col] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    axes[0].plot(raw[time_col], raw[target_col], ".", color="black", markersize=1.2, alpha=0.65)
    axes[0].set_title(f"{target_col} · datos reales")
    axes[0].set_ylabel(target_col)

    axes[1].plot(imputed[time_col], imputed[target_col], ".", color="tab:orange", markersize=2.5, alpha=0.8)
    axes[1].set_title(f"{target_col} · datos imputados")
    axes[1].set_ylabel(target_col)

    axes[2].plot(filled[time_col], filled[target_col], "-", color="tab:blue", linewidth=1.0, alpha=0.9)
    axes[2].set_title(f"{target_col} · serie gapfilled")
    axes[2].set_ylabel(target_col)
    axes[2].set_xlabel("Tiempo")

    fig.suptitle(f"Gap filling por capas: {target_col}", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_col}_layers_3panel.png", dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(config_path: str | Path):
    cfg = load_config(config_path)

    input_csv = Path(cfg["input_csv"])
    output_dir = ensure_output_dir(cfg["output_dir"])

    time_col = cfg["time_col"]
    freq_minutes = int(cfg["freq_minutes"])
    target_cols = cfg["target_cols"]
    structure_vars = cfg["structure_vars"]
    predictor_map = cfg["predictor_map"]

    max_interp_gap_steps = int(cfg["max_interp_gap_steps"])
    max_rf_gap_steps = int(cfg["max_rf_gap_steps"])
    structural_gap_steps = int(cfg["structural_gap_steps"])
    min_segment_points = int(cfg["min_segment_points"])

    rf_params = cfg["rf_params"]
    n_validation_trials = int(cfg["n_validation_trials"])
    fig_dpi = int(cfg.get("fig_dpi", 220))

    analysis_ready_dir = output_dir / "analysis_ready_segments"
    analysis_ready_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    df_regularized = regularize_timeseries(df_raw, time_col=time_col, freq_minutes=freq_minutes)

    candidate_numeric = set(target_cols)
    for v in predictor_map.values():
        candidate_numeric.update(v)

    for col in candidate_numeric:
        if col in df_regularized.columns:
            df_regularized[col] = pd.to_numeric(df_regularized[col], errors="coerce")

    segment_id, segment_summary = classify_structure_combined(
        df=df_regularized,
        vars_for_structure=[c for c in structure_vars if c in df_regularized.columns],
        time_col=time_col,
        structural_gap_steps=structural_gap_steps,
        min_segment_points=min_segment_points,
    )
    df_regularized["segment_id"] = segment_id

    segment_outputs = []
    flag_tables = []
    validation_rows = []
    method_counts = []

    for seg in segment_summary["segment_id"].astype(int).tolist():
        sub_idx = df_regularized["segment_id"].eq(seg)
        sub = df_regularized.loc[sub_idx].copy().reset_index(drop=True)
        segment_flags = pd.DataFrame({time_col: sub[time_col]})

        for i, target_col in enumerate(target_cols):
            if target_col not in sub.columns:
                continue

            predictors = predictor_map.get(target_col, [])
            before_missing = int(sub[target_col].isna().sum())

            sub_filled, flags = hybrid_gapfill_segment(
                df_segment=sub,
                target_col=target_col,
                predictor_cols=predictors,
                time_col=time_col,
                max_interp_gap_steps=max_interp_gap_steps,
                max_rf_gap_steps=max_rf_gap_steps,
                rf_params=rf_params,
            )

            after_missing = int(sub_filled[target_col].isna().sum())
            counts = flags.value_counts()

            method_counts.append(
                {
                    "segment_id": seg,
                    "variable": target_col,
                    "n_rows": len(sub),
                    "missing_before": before_missing,
                    "missing_after": after_missing,
                    "filled_total": before_missing - after_missing,
                    "filled_time_limited": int(counts.get("imputed_time_limited", 0)),
                    "filled_rf": int(counts.get("imputed_rf", 0)),
                    "filled_rf_time_only": int(counts.get("imputed_rf_time_only", 0)),
                    "left_as_nan_large_gap": int(counts.get("left_as_nan_large_gap", 0)),
                    "left_as_nan_edge_gap": int(counts.get("left_as_nan_edge_gap", 0)),
                    "left_as_nan_missing_predictors": int(counts.get("left_as_nan_missing_predictors", 0)),
                }
            )

            metrics = validate_by_masking_hybrid(
                df_segment=sub,
                target_col=target_col,
                predictor_cols=predictors,
                time_col=time_col,
                max_interp_gap_steps=max_interp_gap_steps,
                max_rf_gap_steps=max_rf_gap_steps,
                rf_params=rf_params,
                n_trials=n_validation_trials,
                seed=42 + i + seg,
            )
            metrics["segment_id"] = seg
            validation_rows.append(metrics)

            sub[target_col] = sub_filled[target_col].values
            segment_flags[f"{target_col}_flag"] = flags.values

        sub["segment_id"] = seg
        segment_outputs.append(sub)

        segment_flags["segment_id"] = seg
        flag_tables.append(segment_flags)

        ready_subset = [c for c in target_cols if c in sub.columns]
        ready_sub = sub.dropna(subset=ready_subset).copy()
        ready_sub.to_csv(analysis_ready_dir / f"segment_{seg:02d}_analysis_ready.csv", index=False)

    df_gapfilled_all = pd.concat(segment_outputs, ignore_index=True) if segment_outputs else pd.DataFrame()
    df_gapfill_flags = pd.concat(flag_tables, ignore_index=True) if flag_tables else pd.DataFrame()
    df_validation = pd.DataFrame(validation_rows) if validation_rows else pd.DataFrame()
    df_method_counts = pd.DataFrame(method_counts) if method_counts else pd.DataFrame()

    df_regularized.to_csv(output_dir / "art_regularized.csv", index=False)
    segment_summary.to_csv(output_dir / "art_segment_summary.csv", index=False)
    df_gapfilled_all.to_csv(output_dir / "art_gapfilled_all_segments.csv", index=False)
    df_gapfill_flags.to_csv(output_dir / "art_gapfill_flags.csv", index=False)
    df_validation.to_csv(output_dir / "art_gapfill_validation_summary.csv", index=False)
    df_method_counts.to_csv(output_dir / "art_gapfill_method_counts.csv", index=False)

    for target_col in target_cols:
        if target_col in df_gapfilled_all.columns:
            plot_layers_3panel(
                df_regularized=df_regularized,
                df_gapfilled_all=df_gapfilled_all,
                df_gapfill_flags=df_gapfill_flags,
                target_col=target_col,
                output_dir=output_dir,
                time_col=time_col,
                fig_dpi=fig_dpi,
            )

    if not df_validation.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df_validation.groupby("variable")["rmse"].mean().plot(kind="bar", ax=axes[0], title="RMSE promedio")
        axes[0].set_ylabel("RMSE")
        df_validation.groupby("variable")["r2"].mean().plot(kind="bar", ax=axes[1], title="R² promedio")
        axes[1].set_ylabel("R²")
        plt.tight_layout()
        plt.savefig(output_dir / "gapfill_validation_metrics.png", dpi=fig_dpi)
        plt.close(fig)

    summary = df_method_counts.merge(df_validation, on=["segment_id", "variable"], how="left").sort_values(
        ["segment_id", "variable"]
    )
    summary.to_csv(output_dir / "art_gapfill_summary_combined.csv", index=False)

    print("Pipeline terminado.")
    print(f"Directorio de salida: {output_dir}")
    print(f"Segmentos detectados: {segment_summary['segment_id'].astype(int).tolist()}")
    return {
        "output_dir": str(output_dir),
        "segments": segment_summary.to_dict(orient="records"),
        "summary_rows": len(summary),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gap filling híbrido configurado por YAML.")
    parser.add_argument("--config", required=True, help="Ruta al archivo YAML de configuración.")
    args = parser.parse_args()
    run_pipeline(args.config)
