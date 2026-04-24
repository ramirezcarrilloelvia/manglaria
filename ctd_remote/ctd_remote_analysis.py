from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_vars(df: pd.DataFrame, cfg: dict):
    required_core = [v for v in cfg["required_core_vars"] if v in df.columns]
    optional_core = [v for v in cfg["optional_core_vars"] if v in df.columns]
    vars_all = required_core + optional_core
    missing_required = [v for v in cfg["required_core_vars"] if v not in df.columns]
    if missing_required:
        raise ValueError(f"Faltan variables requeridas: {missing_required}")
    return vars_all


def compute_anomalies(df_input: pd.DataFrame, vars_all: list[str], time_col: str) -> pd.DataFrame:
    df_anom = df_input.copy()
    df_anom[time_col] = pd.to_datetime(df_anom[time_col], errors="coerce", utc=True)
    df_anom = df_anom.sort_values(time_col).set_index(time_col)
    df_anom["hour"] = df_anom.index.hour
    df_anom["minute"] = df_anom.index.minute

    for var in vars_all:
        df_anom[f"{var}_anom"] = df_anom[var] - df_anom.groupby(["hour", "minute"])[var].transform("mean")

    return df_anom


def interpret_beta(beta):
    if np.isnan(beta):
        return "not available"
    elif beta < 0.5:
        return "white-noise-like, weak structure"
    elif beta < 1.0:
        return "weak-moderate temporal correlation"
    elif beta < 1.3:
        return "1/f-like, multiscale structure"
    elif beta < 1.8:
        return "strong temporal correlation"
    else:
        return "strong memory, smooth dynamics"


def run_psd_beta(df_anom, anom_cols, fs, max_nperseg, figures_dir, metrics_dir):
    fig, axes = plt.subplots(len(anom_cols), 1, figsize=(10, max(3 * len(anom_cols), 6)))
    axes = np.atleast_1d(axes)
    beta_rows = []

    for i, var in enumerate(anom_cols):
        x = df_anom[var].dropna().values
        if len(x) < 8:
            beta_rows.append({"variable": var, "beta": np.nan, "r2": np.nan, "n_points": len(x), "interpretation": "insufficient data"})
            axes[i].axis("off")
            continue

        nperseg = min(max_nperseg, len(x))
        f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
        f = f[1:]
        Pxx = Pxx[1:]
        valid = (f > 0) & (Pxx > 0)
        f = f[valid]
        Pxx = Pxx[valid]

        if len(f) < 3:
            beta_rows.append({"variable": var, "beta": np.nan, "r2": np.nan, "n_points": len(x), "interpretation": "insufficient spectral points"})
            axes[i].axis("off")
            continue

        log_f = np.log10(f)
        log_Pxx = np.log10(Pxx)
        slope, intercept, r_value, *_ = stats.linregress(log_f, log_Pxx)
        beta = -slope
        r2 = r_value ** 2

        beta_rows.append({"variable": var, "beta": beta, "r2": r2, "n_points": len(x), "interpretation": interpret_beta(beta)})

        axes[i].plot(log_f, log_Pxx, label="PSD")
        axes[i].plot(log_f, intercept + slope * log_f, "--", label=f"β={beta:.2f}")
        axes[i].set_title(var)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "psd_beta.png", dpi=300)
    plt.close(fig)

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(metrics_dir / "beta_results.csv", index=False)
    return beta_df


def interpret_alpha(alpha, r2):
    if np.isnan(alpha):
        return "not available"
    if alpha >= 2.5:
        text = "Variance strongly concentrated in dominant modes."
    elif alpha >= 1.5:
        text = "Variance moderately concentrated."
    else:
        text = "Variance spread across several modes."
    if r2 < 0.5:
        text += " Fit is weak."
    return text


def run_pca_alpha(df_anom, var_list, group_name, figures_dir):
    anom_group_cols = [f"{v}_anom" for v in var_list if f"{v}_anom" in df_anom.columns]
    if len(anom_group_cols) < 2:
        return None, pd.DataFrame([{"group": group_name, "alpha": np.nan, "r2": np.nan, "n_obs": 0, "n_vars": len(anom_group_cols), "interpretation": "insufficient variables"}])

    X = df_anom[anom_group_cols].dropna().copy()
    if X.shape[0] < 5:
        return None, pd.DataFrame([{"group": group_name, "alpha": np.nan, "r2": np.nan, "n_obs": X.shape[0], "n_vars": X.shape[1], "interpretation": "insufficient observations"}])

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    k = np.arange(1, len(eigenvalues) + 1)
    valid = eigenvalues > 0

    variance_table = pd.DataFrame({
        "group": group_name,
        "component": np.arange(1, len(explained_variance_ratio) + 1),
        "explained_variance_ratio": explained_variance_ratio,
        "explained_variance_pct": explained_variance_ratio * 100.0,
    })

    if valid.sum() < 2:
        return variance_table, pd.DataFrame([{"group": group_name, "alpha": np.nan, "r2": np.nan, "n_obs": X.shape[0], "n_vars": X.shape[1], "interpretation": "insufficient positive eigenvalues"}])

    log_k = np.log10(k[valid])
    log_lambda = np.log10(eigenvalues[valid])
    slope, intercept, r_value, *_ = stats.linregress(log_k, log_lambda)
    alpha = -slope
    r2 = r_value ** 2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(log_k, log_lambda, "o", label="eigenvalues")
    ax.plot(log_k, intercept + slope * log_k, "--", label=f"α={alpha:.2f}")
    ax.set_title(f"PCA alpha: {group_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = group_name.lower().replace(" ", "_")
    plt.savefig(figures_dir / f"pca_alpha_{safe_name}.png", dpi=300)
    plt.close(fig)

    summary = pd.DataFrame([{
        "group": group_name,
        "alpha": alpha,
        "r2": r2,
        "n_obs": X.shape[0],
        "n_vars": X.shape[1],
        "interpretation": interpret_alpha(alpha, r2),
    }])
    return variance_table, summary


def make_global_bins(x, min_bins=10, max_bins=40):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    if np.all(x == x[0]):
        return np.linspace(x[0] - 0.5, x[0] + 0.5, min_bins + 1)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        n_bins = min_bins
    else:
        bin_width = 2 * iqr * (len(x) ** (-1 / 3))
        if bin_width <= 0:
            n_bins = min_bins
        else:
            n_bins = int(np.ceil((x.max() - x.min()) / bin_width))
            n_bins = max(min_bins, min(n_bins, max_bins))
    return np.linspace(x.min(), x.max(), n_bins + 1) if x.max() != x.min() else np.linspace(x.min() - 0.5, x.max() + 0.5, n_bins + 1)


def fisher_information_1d(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.sum() <= 0:
        return np.nan
    probs = probs / probs.sum()
    return 4 * np.sum(np.diff(np.sqrt(probs)) ** 2)


def compute_univariate_fi(series, window_size, step_size, epsilon=1e-12):
    series = series.dropna().copy()
    if len(series) < window_size:
        return pd.DataFrame(columns=["FI", "FI2", "occupied_bins", "entropy"]), None
    bin_edges = make_global_bins(series.values)
    if bin_edges is None:
        return pd.DataFrame(columns=["FI", "FI2", "occupied_bins", "entropy"]), None

    results = []
    for start in range(0, len(series) - window_size + 1, step_size):
        end = start + window_size
        window = series.iloc[start:end]
        counts, _ = np.histogram(window.values, bins=bin_edges)
        probs = counts.astype(float) + epsilon
        probs = probs / probs.sum()
        fi = fisher_information_1d(probs)
        fi2 = fi ** 2 if np.isfinite(fi) else np.nan
        occupied_bins = np.sum(counts > 0)
        entropy = -np.sum(probs * np.log(probs))
        t_center = window.index[len(window) // 2]
        results.append({"TIMESTAMP": t_center, "FI": fi, "FI2": fi2, "occupied_bins": occupied_bins, "entropy": entropy})
    return (pd.DataFrame(results).set_index("TIMESTAMP"), bin_edges) if results else (pd.DataFrame(columns=["FI", "FI2", "occupied_bins", "entropy"]), bin_edges)


def detect_abrupt_fi_changes(df_fi_var, k=3):
    if df_fi_var is None or df_fi_var.empty or "FI" not in df_fi_var.columns:
        return pd.DataFrame(columns=["FI", "FI2", "occupied_bins", "entropy", "abs_dFI", "abrupt_change"]), np.nan, np.nan, np.nan
    df_tmp = df_fi_var.copy()
    df_tmp["abs_dFI"] = df_tmp["FI"].diff().abs()
    valid_dfi = df_tmp["abs_dFI"].dropna()
    if valid_dfi.empty:
        df_tmp["abrupt_change"] = False
        return df_tmp, np.nan, np.nan, np.nan
    median_val = valid_dfi.median()
    mad_val = np.median(np.abs(valid_dfi - median_val))
    threshold = median_val + k * mad_val if not np.isnan(mad_val) else np.nan
    df_tmp["abrupt_change"] = df_tmp["abs_dFI"] > threshold if np.isfinite(threshold) else False
    return df_tmp, threshold, median_val, mad_val


def fisher_information_from_probs(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.sum() <= 0:
        return np.nan
    probs = probs / probs.sum()
    return 4 * np.sum(np.diff(np.sqrt(probs)) ** 2)


def compute_multivariate_fi(df_input, columns, window_size, step_size, n_bins=5, epsilon=1e-12, min_coverage=0.8, global_bins=True):
    df_base = df_input[columns].copy()
    if len(df_base) < window_size:
        return pd.DataFrame(columns=["FI", "FI2", "n_states", "entropy", "n_samples_window", "coverage_pct"])

    bin_edges = {}
    if global_bins:
        for col in columns:
            x = df_base[col].dropna().values
            if len(x) == 0:
                return pd.DataFrame(columns=["FI", "FI2", "n_states", "entropy", "n_samples_window", "coverage_pct"])
            edges = np.histogram_bin_edges(x, bins=n_bins)
            if len(edges) < 2:
                edges = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, n_bins + 1)
            bin_edges[col] = edges

    results = []
    for start in range(0, len(df_base) - window_size + 1, step_size):
        end = start + window_size
        window = df_base.iloc[start:end].copy()
        window_valid = window.dropna()
        coverage_pct = len(window_valid) / len(window) if len(window) > 0 else 0
        if coverage_pct < min_coverage or len(window_valid) < 3:
            continue

        if not global_bins:
            bin_edges = {}
            for col in columns:
                x = window_valid[col].values
                edges = np.histogram_bin_edges(x, bins=n_bins)
                if len(edges) < 2:
                    edges = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, n_bins + 1)
                bin_edges[col] = edges

        state_df = pd.DataFrame(index=window_valid.index)
        for col in columns:
            state_df[col] = np.digitize(window_valid[col].values, bin_edges[col][1:-1], right=False)

        states = state_df.apply(lambda row: tuple(row.values), axis=1)
        counts = states.value_counts().sort_values(ascending=False).values.astype(float)
        probs = counts + epsilon
        probs = probs / probs.sum()

        fi = fisher_information_from_probs(probs)
        fi2 = fi ** 2 if np.isfinite(fi) else np.nan
        entropy = -np.sum(probs * np.log(probs))
        n_states = len(counts)
        t_center = window.index[len(window) // 2]

        results.append({
            "TIMESTAMP": t_center,
            "FI": fi,
            "FI2": fi2,
            "n_states": n_states,
            "entropy": entropy,
            "n_samples_window": len(window_valid),
            "coverage_pct": coverage_pct * 100.0,
        })

    return pd.DataFrame(results).set_index("TIMESTAMP") if results else pd.DataFrame(columns=["FI", "FI2", "n_states", "entropy", "n_samples_window", "coverage_pct"])


def dynamic_functional_graph(df_anom, anom_cols, window_size, step_size, corr_threshold=0.3):
    results = []
    df_base = df_anom[anom_cols].copy()
    for start in range(0, len(df_base) - window_size + 1, step_size):
        end = start + window_size
        window = df_base.iloc[start:end]
        window_valid = window.dropna()

        if len(window_valid) < window_size * 0.8:
            continue

        corr = window_valid.corr()
        corr_vals = corr.values.copy()
        np.fill_diagonal(corr_vals, 0)
        abs_corr = np.abs(corr_vals)

        mean_connectivity = abs_corr.mean()
        total_strength = abs_corr.sum()
        strong_links = np.sum(abs_corr >= corr_threshold)
        possible_links = abs_corr.size - len(corr)
        link_density = strong_links / possible_links if possible_links > 0 else np.nan

        t_center = window.index[len(window) // 2]
        results.append({
            "TIMESTAMP": t_center,
            "mean_connectivity": mean_connectivity,
            "total_strength": total_strength,
            "strong_links": strong_links,
            "link_density": link_density,
        })

    return pd.DataFrame(results).set_index("TIMESTAMP") if results else pd.DataFrame(columns=["mean_connectivity", "total_strength", "strong_links", "link_density"])


def analyze_segment(segment_file, cfg):
    output_root = ensure_dir(cfg["output_dir"])
    segment_name = segment_file.stem
    segment_dir = ensure_dir(output_root / segment_name)
    figures_dir = ensure_dir(segment_dir / "figures")
    metrics_dir = ensure_dir(segment_dir / "metrics")

    df = pd.read_csv(segment_file)
    time_col = cfg["time_col"]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.sort_values(time_col)

    vars_all = infer_vars(df, cfg)

    missing_summary = df[vars_all].isna().sum().reset_index()
    missing_summary.columns = ["variable", "missing_n"]
    missing_summary["missing_pct"] = 100 * missing_summary["missing_n"] / len(df)
    missing_summary.to_csv(metrics_dir / "missing_summary.csv", index=False)

    outlier_rows = []
    for var in vars_all:
        x = pd.to_numeric(df[var], errors="coerce")
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr):
            outlier_n = np.nan
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_n = int(((x < lower) | (x > upper)).sum())
        outlier_rows.append({"variable": var, "outlier_n": outlier_n})
    pd.DataFrame(outlier_rows).to_csv(metrics_dir / "outlier_summary.csv", index=False)

    df_anom = compute_anomalies(df, vars_all, time_col)
    anom_cols = [f"{v}_anom" for v in vars_all if f"{v}_anom" in df_anom.columns]
    df_anom.reset_index().to_csv(metrics_dir / "anomalies_long.csv", index=False)

    fig, axes = plt.subplots(len(vars_all), 1, figsize=(12, max(3 * len(vars_all), 6)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, var in enumerate(vars_all):
        axes[i].plot(df_anom.index, df_anom[var], alpha=0.5, linewidth=1, label="Original")
        axes[i].plot(df_anom.index, df_anom[f"{var}_anom"], alpha=0.9, linewidth=1.2, label="Anomaly")
        axes[i].axhline(0, linestyle="--")
        axes[i].set_title(var)
        if i == 0:
            axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "original_vs_anomaly.png", dpi=300)
    plt.close(fig)

    corr_matrix = df_anom[anom_cols].corr()
    adj_matrix = corr_matrix.where(np.abs(corr_matrix) >= cfg["graph"]["corr_threshold"], 0.0).copy()
    for col in adj_matrix.columns:
        adj_matrix.loc[col, col] = 0.0
    corr_matrix.to_csv(metrics_dir / "functional_graph_correlation_matrix.csv")
    adj_matrix.to_csv(metrics_dir / "functional_graph_adjacency_matrix.csv")

    beta_df = run_psd_beta(df_anom, anom_cols, cfg["psd"]["fs_hz"], cfg["psd"]["max_nperseg"], figures_dir, metrics_dir)

    alpha_tables, variance_tables = [], []
    for group_name, var_list in cfg["pca_groups"].items():
        use_vars = [v for v in var_list if v in vars_all]
        var_table, alpha_summary = run_pca_alpha(df_anom, use_vars, group_name, figures_dir)
        alpha_tables.append(alpha_summary)
        if var_table is not None:
            variance_tables.append(var_table)

    alpha_df = pd.concat(alpha_tables, ignore_index=True) if alpha_tables else pd.DataFrame()
    alpha_df.to_csv(metrics_dir / "alpha_results.csv", index=False)
    if variance_tables:
        pd.concat(variance_tables, ignore_index=True).to_csv(metrics_dir / "pca_variance_tables.csv", index=False)

    samples_per_hour = cfg["fi"]["samples_per_hour"]
    window_size = int(cfg["fi"]["window_hours"] * samples_per_hour)
    step_size = int(cfg["fi"]["step_hours"] * samples_per_hour)
    epsilon = float(cfg["fi"]["epsilon"])

    fi_results, summary_rows, fi_long_rows = {}, [], []
    for var in vars_all:
        col = f"{var}_anom"
        df_fi_var, _ = compute_univariate_fi(df_anom[col], window_size, step_size, epsilon=epsilon)
        fi_results[var] = df_fi_var
        if df_fi_var.empty:
            summary_rows.append({"variable": var, "mean_FI": np.nan, "std_FI": np.nan, "min_FI": np.nan, "max_FI": np.nan, "n_windows": 0})
            continue
        tmp = df_fi_var.copy()
        tmp["variable"] = var
        fi_long_rows.append(tmp.reset_index())
        summary_rows.append({
            "variable": var,
            "mean_FI": df_fi_var["FI"].mean(),
            "std_FI": df_fi_var["FI"].std(),
            "min_FI": df_fi_var["FI"].min(),
            "max_FI": df_fi_var["FI"].max(),
            "n_windows": len(df_fi_var),
        })

    pd.DataFrame(summary_rows).to_csv(metrics_dir / "fi_summary.csv", index=False)
    if fi_long_rows:
        pd.concat(fi_long_rows, ignore_index=True).to_csv(metrics_dir / "fi_univariate_long.csv", index=False)

    abrupt_rows, abrupt_long_rows = [], []
    for var, dfi in fi_results.items():
        df_diag, threshold, median_val, mad_val = detect_abrupt_fi_changes(dfi, k=cfg["abrupt"]["k_mad"])
        if df_diag.empty:
            abrupt_rows.append({"variable": var, "n_abrupt_events": 0, "event_rate_pct": np.nan, "threshold_abs_dFI": threshold, "median_abs_dFI": median_val, "MAD_abs_dFI": mad_val, "max_abs_dFI": np.nan})
            continue
        n_events = int(df_diag["abrupt_change"].sum())
        event_rate = 100 * n_events / len(df_diag) if len(df_diag) > 0 else np.nan
        abrupt_rows.append({
            "variable": var,
            "n_abrupt_events": n_events,
            "event_rate_pct": event_rate,
            "threshold_abs_dFI": threshold,
            "median_abs_dFI": median_val,
            "MAD_abs_dFI": mad_val,
            "max_abs_dFI": df_diag["abs_dFI"].max(),
        })
        tmp = df_diag.copy()
        tmp["variable"] = var
        abrupt_long_rows.append(tmp.reset_index())

    pd.DataFrame(abrupt_rows).to_csv(metrics_dir / "abrupt_summary.csv", index=False)
    if abrupt_long_rows:
        pd.concat(abrupt_long_rows, ignore_index=True).to_csv(metrics_dir / "abrupt_windows_long.csv", index=False)

    mv_cols = [c for c in anom_cols if c in df_anom.columns]
    df_fi_multivar = compute_multivariate_fi(
        df_input=df_anom,
        columns=mv_cols,
        window_size=window_size,
        step_size=step_size,
        n_bins=cfg["fi"]["multivariate_bins"],
        epsilon=epsilon,
        min_coverage=cfg["fi"]["min_coverage"],
        global_bins=True,
    )
    df_fi_multivar.to_csv(metrics_dir / "fi_multivariate.csv")

    df_graph_dyn = dynamic_functional_graph(df_anom, mv_cols, window_size, step_size, corr_threshold=cfg["graph"]["corr_threshold"])
    df_graph_dyn.to_csv(metrics_dir / "dynamic_graph_metrics.csv")

    df_combined = df_fi_multivar.join(df_graph_dyn, how="inner")
    df_combined.to_csv(metrics_dir / "fi_graph_combined.csv")

    if not df_combined.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(df_combined["mean_connectivity"], df_combined["FI"], alpha=0.7)
        ax.set_xlabel("Mean connectivity")
        ax.set_ylabel("FI")
        ax.set_title("FI vs Network Connectivity")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fi_vs_network_connectivity.png", dpi=300)
        plt.close(fig)

    use_features = [f for f in cfg["regimes"]["features"] if f in df_combined.columns]
    if len(use_features) >= 2 and not df_combined.empty:
        df_regime = df_combined[use_features].dropna().copy()
        if len(df_regime) >= cfg["regimes"]["k"]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_regime)
            kmeans = KMeans(n_clusters=cfg["regimes"]["k"], random_state=42, n_init=10)
            df_regime["regime"] = kmeans.fit_predict(X_scaled)
            df_regime.to_csv(metrics_dir / "system_regimes.csv")

            fig, ax = plt.subplots(figsize=(8, 6))
            for r in sorted(df_regime["regime"].unique()):
                subset = df_regime[df_regime["regime"] == r]
                ax.scatter(subset[use_features[1]], subset[use_features[0]], label=f"Regime {r}", alpha=0.7)
            ax.set_xlabel(use_features[1])
            ax.set_ylabel(use_features[0])
            ax.set_title("Detected regimes")
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "system_regimes.png", dpi=300)
            plt.close(fig)
        else:
            pd.DataFrame().to_csv(metrics_dir / "system_regimes.csv")
    else:
        pd.DataFrame().to_csv(metrics_dir / "system_regimes.csv")

    df_fi_multivar_diag, threshold_mv, median_mv, mad_mv = detect_abrupt_fi_changes(df_fi_multivar, k=cfg["abrupt"]["k_mad"])
    df_fi_multivar_diag.to_csv(metrics_dir / "fi_multivariate_abrupt.csv")

    if not df_fi_multivar_diag.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_fi_multivar_diag.index, df_fi_multivar_diag["FI"], linewidth=1.8)
        abrupt_mask = df_fi_multivar_diag["abrupt_change"].fillna(False)
        ax.scatter(df_fi_multivar_diag.index[abrupt_mask], df_fi_multivar_diag["FI"][abrupt_mask], s=35)
        ax.set_title("Multivariate FI with abrupt events")
        ax.set_xlabel("Time")
        ax.set_ylabel("FI")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fi_multivariate_abrupt_events.png", dpi=300)
        plt.close(fig)

    return {"segment_name": segment_file.stem, "n_rows": len(df), "output_dir": str(segment_dir)}


def run_pipeline(config_path):
    cfg = load_config(config_path)
    input_dir = Path(cfg["input_dir"])
    output_dir = ensure_dir(cfg["output_dir"])

    segment_files = sorted(input_dir.glob(cfg.get("segment_glob", "segment_*_analysis_ready.csv")))
    if not segment_files:
        raise FileNotFoundError(f"No se encontraron segmentos en {input_dir}")

    summaries = []
    for seg in segment_files:
        print(f"Procesando: {seg.name}")
        summaries.append(analyze_segment(seg, cfg))

    pd.DataFrame(summaries).to_csv(output_dir / "analysis_run_summary.csv", index=False)
    print("Análisis CTD remote terminado.")
    print(f"Directorio de salida: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis posterior para CTD remote.")
    parser.add_argument("--config", required=True, help="Ruta al archivo YAML de configuración.")
    args = parser.parse_args()
    run_pipeline(args.config)
