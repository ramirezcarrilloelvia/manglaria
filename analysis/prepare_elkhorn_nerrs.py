from __future__ import annotations

"""Download, harmonize and audit Elkhorn Slough NERR water-quality archives.

Primary-analysis values are never imputed here. The script preserves the raw
parameter QA/QC flags, builds a conservative observed-only mask, records gap
structure, and writes canonical tables for the Tolerance Landscapes workhorse.
"""

from pathlib import Path
import argparse
import re
import shutil
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://cdmo.baruch.sc.edu/waf/swmp_data_archives"
STATIONS = {
    "south_marsh": "elksmwq",
    "azevedo_pond": "elkapwq",
    "north_marsh": "elknmwq",
    "vierra_mouth": "elkvmwq",
}
ACCEPTED_PRIMARY_FLAGS = {0, 5}
ACCEPTED_CDEPTH_FLAGS = {0, 3, 5}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    return {_norm(c): c for c in df.columns}


def _pick(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cmap = _column_map(df)
    for alias in aliases:
        col = cmap.get(_norm(alias))
        if col is not None:
            return col
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")


def _download(url: str, path: Path, timeout: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tolerance-landscape-research"})
    with urllib.request.urlopen(req, timeout=timeout) as response, open(path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    if path.stat().st_size < 1000:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")


def _parse_flag_number(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*(-?\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def _flag_ok(series: pd.Series | None, *, allow_calculated: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    flags = series.map(_parse_flag_number)
    accepted = ACCEPTED_CDEPTH_FLAGS if allow_calculated else ACCEPTED_PRIMARY_FLAGS
    return flags.isin(accepted)


def _timestamp(df: pd.DataFrame) -> pd.Series:
    col = _pick(df, ["DateTimeStamp", "datetimestamp", "timestamp", "datetime", "date_time"])
    if col:
        return pd.to_datetime(df[col], errors="coerce")
    date_col = _pick(df, ["Date", "sampledate"])
    time_col = _pick(df, ["Time", "sampletime"])
    if date_col and time_col:
        return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    raise ValueError(f"Could not identify timestamp column. Columns={list(df.columns)}")


def _event_text(df: pd.DataFrame) -> pd.Series:
    """Collapse all flag/code fields to text without depending on pandas agg semantics."""
    flag_cols = [c for c in df.columns if _norm(c).startswith("f")]
    if not flag_cols:
        return pd.Series("", index=df.index, dtype="object")
    subset = df[flag_cols]
    return subset.apply(
        lambda row: "|".join(str(value) for value in row.tolist() if pd.notna(value)),
        axis=1,
    )


def _canonicalize(df: pd.DataFrame, station_name: str, station_code: str, year: int) -> tuple[pd.DataFrame, dict]:
    temp_col = _pick(df, ["Temp", "temperature", "water_temperature"])
    sal_col = _pick(df, ["Sal", "salinity"])
    do_col = _pick(df, ["DO_mgl", "do_mgl", "dissolved_oxygen_mgl"])
    cdepth_col = _pick(df, ["cDepth", "corrected_depth"])
    depth_col = cdepth_col or _pick(df, ["Depth", "depth"])

    if not all([temp_col, sal_col, do_col, depth_col]):
        raise ValueError(
            f"Missing required variables for {station_code} {year}: "
            f"temp={temp_col}, sal={sal_col}, do={do_col}, depth={depth_col}; columns={list(df.columns)}"
        )

    def flag_for(value_col: str) -> str | None:
        return _pick(df, [f"F_{value_col}", f"F{value_col}", f"flag_{value_col}"])

    f_temp = flag_for(temp_col)
    f_sal = flag_for(sal_col)
    f_do = flag_for(do_col)
    f_depth = flag_for(depth_col)
    f_record = _pick(df, ["F_Record", "FRecord", "record_flag"])

    out = pd.DataFrame({
        "timestamp": _timestamp(df),
        "station": station_name,
        "station_code": station_code,
        "year": year,
        "temp": pd.to_numeric(df[temp_col], errors="coerce"),
        "sal": pd.to_numeric(df[sal_col], errors="coerce"),
        "do_mgl": pd.to_numeric(df[do_col], errors="coerce"),
        "depth": pd.to_numeric(df[depth_col], errors="coerce"),
    })

    raw_flags = {
        "temp": df[f_temp] if f_temp else None,
        "sal": df[f_sal] if f_sal else None,
        "do_mgl": df[f_do] if f_do else None,
        "depth": df[f_depth] if f_depth else None,
    }
    for key, series in raw_flags.items():
        out[f"raw_flag_{key}"] = series.astype(str) if series is not None else ""
    out["raw_flag_record"] = df[f_record].astype(str) if f_record else ""

    ok_components = []
    missing_flag_cols = []
    for key, series in raw_flags.items():
        if series is None:
            missing_flag_cols.append(key)
            ok = pd.Series(False, index=out.index)
        else:
            ok = _flag_ok(series, allow_calculated=(key == "depth" and cdepth_col is not None))
            ok.index = out.index
        out[f"qa_ok_{key}"] = ok.to_numpy()
        ok_components.append(ok.to_numpy())

    numeric_ok = out[["temp", "sal", "do_mgl", "depth"]].notna().all(axis=1).to_numpy()
    qa_all = np.logical_and.reduce(ok_components) if ok_components else np.zeros(len(out), dtype=bool)
    out["qa_status"] = np.where(numeric_ok & qa_all, "observed", "excluded")

    flag_text = _event_text(df)
    for code in ["CDA", "CRE", "CWE", "CAB", "CFK", "CLT"]:
        out[f"event_{code.lower()}"] = flag_text.str.contains(code, regex=False, na=False).to_numpy()

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    expected = 35040 + (96 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 0)
    usable = out["qa_status"].eq("observed")
    usable_times = pd.DatetimeIndex(out.loc[usable, "timestamp"])
    if len(usable_times) > 1:
        gaps_h = pd.Series(usable_times).diff().dt.total_seconds().div(3600.0)
        max_gap_h = float(gaps_h.max())
        gap_gt_1h = int((gaps_h > 1.0).sum())
        gap_gt_6h = int((gaps_h > 6.0).sum())
        gap_gt_24h = int((gaps_h > 24.0).sum())
    else:
        max_gap_h = np.nan
        gap_gt_1h = gap_gt_6h = gap_gt_24h = 0

    summary = {
        "station": station_name,
        "station_code": station_code,
        "year": year,
        "n_rows": int(len(out)),
        "expected_15min_rows": int(expected),
        "raw_timestamp_coverage_pct": 100.0 * len(out) / expected,
        "usable_rows": int(usable.sum()),
        "usable_coverage_pct": 100.0 * usable.sum() / expected,
        "max_usable_gap_hours": max_gap_h,
        "n_gaps_gt_1h": gap_gt_1h,
        "n_gaps_gt_6h": gap_gt_6h,
        "n_gaps_gt_24h": gap_gt_24h,
        "hypoxia_lt3_n": int((out.loc[usable, "do_mgl"] < 3.0).sum()),
        "hypoxia_lt2_n": int((out.loc[usable, "do_mgl"] < 2.0).sum()),
        "cda_flag_n": int(out["event_cda"].sum()),
        "rain_flag_n": int(out["event_cre"].sum()),
        "weather_flag_n": int(out["event_cwe"].sum()),
        "bloom_flag_n": int(out["event_cab"].sum()),
        "missing_flag_columns": "|".join(missing_flag_cols),
        "depth_source": str(depth_col),
    }
    return out, summary


def run(start_year: int, end_year: int, out_dir: Path) -> None:
    raw_dir = out_dir / "raw"
    canonical_dir = out_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    station_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in STATIONS}

    for station_name, station_code in STATIONS.items():
        for year in range(start_year, end_year + 1):
            url = f"{BASE_URL}/{station_code}{year}.csv"
            path = raw_dir / f"{station_code}{year}.csv"
            print(f"Downloading/reading {station_code} {year}")
            try:
                _download(url, path)
                raw = _read_csv(path)
                canon, summary = _canonicalize(raw, station_name, station_code, year)
                station_frames[station_name].append(canon)
                summaries.append(summary)
            except Exception as exc:
                summaries.append({"station": station_name, "station_code": station_code, "year": year, "error": repr(exc)})
                print(f"WARNING {station_code} {year}: {exc}")

    for station_name, frames in station_frames.items():
        if not frames:
            continue
        full = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
        full.to_csv(canonical_dir / f"{station_name}.csv", index=False)
        full[full["qa_status"].eq("observed")].to_csv(
            canonical_dir / f"{station_name}_observed_only.csv", index=False
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "elkhorn_qaqc_by_year.csv", index=False)

    if not summary_df.empty and "usable_coverage_pct" in summary_df.columns:
        valid = summary_df.dropna(subset=["usable_coverage_pct"])
        if not valid.empty:
            agg = valid.groupby(["station", "station_code"], as_index=False).agg(
                years=("year", "count"),
                median_usable_coverage_pct=("usable_coverage_pct", "median"),
                min_usable_coverage_pct=("usable_coverage_pct", "min"),
                total_usable_rows=("usable_rows", "sum"),
                total_hypoxia_lt3=("hypoxia_lt3_n", "sum"),
                total_cda_flags=("cda_flag_n", "sum"),
                max_gap_hours=("max_usable_gap_hours", "max"),
                total_gaps_gt_24h=("n_gaps_gt_24h", "sum"),
            )
            agg.to_csv(out_dir / "elkhorn_qaqc_station_summary.csv", index=False)

    sm = canonical_dir / "south_marsh.csv"
    if sm.exists():
        target_dir = Path("data/elkhorn")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sm, target_dir / "south_marsh.csv")
        shutil.copy2(sm, target_dir / "south_marsh_reference.csv")

    print(f"Elkhorn archive preparation complete: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--out-dir", default="output/elkhorn_data_audit")
    args = parser.parse_args()
    run(args.start_year, args.end_year, Path(args.out_dir))
