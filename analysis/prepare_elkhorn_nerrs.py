from __future__ import annotations

"""Download and conservatively audit Elkhorn Slough NERR water-quality data."""

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
# SWMP flag 3 denotes calculated/derived accepted values. Standard variables
# such as salinity and DO_mgl can legitimately carry this flag. Flags 1
# (suspect) and negative/rejected flags remain excluded from the primary run.
ACCEPTED_PRIMARY_FLAGS = {0, 3, 5}


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _pick(df: pd.DataFrame, aliases) -> str | None:
    cmap = {_norm(c): c for c in df.columns}
    for alias in aliases:
        if _norm(alias) in cmap:
            return cmap[_norm(alias)]
    return None


def _download(url: str, path: Path, timeout: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tolerance-landscape-research"})
    with urllib.request.urlopen(req, timeout=timeout) as response, open(path, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")


def _flag_number(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*(-?\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def _timestamp(df: pd.DataFrame) -> pd.Series:
    col = _pick(df, ["DateTimeStamp", "timestamp", "datetime", "date_time"])
    if col:
        return pd.to_datetime(df[col], errors="coerce")
    date_col = _pick(df, ["Date", "sampledate"])
    time_col = _pick(df, ["Time", "sampletime"])
    if date_col and time_col:
        return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    raise ValueError("No timestamp field found")


def _event_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if _norm(c).startswith("f")]
    if not cols:
        return pd.Series("", index=df.index, dtype="object")
    return df[cols].apply(
        lambda row: "|".join(str(v) for v in row.tolist() if pd.notna(v)), axis=1
    )


def _canonicalize(df: pd.DataFrame, station: str, code: str, year: int):
    temp_col = _pick(df, ["Temp", "temperature", "water_temperature"])
    sal_col = _pick(df, ["Sal", "salinity"])
    do_col = _pick(df, ["DO_mgl", "do_mgl", "dissolved_oxygen_mgl"])
    depth_col = _pick(df, ["Depth", "depth"])
    if depth_col is None:
        depth_col = _pick(df, ["cDepth", "corrected_depth"])

    if not all([temp_col, sal_col, do_col, depth_col]):
        raise ValueError(
            f"Required columns missing: temp={temp_col}, sal={sal_col}, do={do_col}, depth={depth_col}"
        )

    def flag_for(value_col: str) -> str | None:
        return _pick(df, [f"F_{value_col}", f"F{value_col}", f"flag_{value_col}"])

    vars_and_cols = {"temp": temp_col, "sal": sal_col, "do_mgl": do_col, "depth": depth_col}
    out = pd.DataFrame({
        "timestamp": _timestamp(df),
        "station": station,
        "station_code": code,
        "year": year,
        **{name: pd.to_numeric(df[col], errors="coerce") for name, col in vars_and_cols.items()},
    })

    qa_arrays = []
    missing_flags = []
    for name, value_col in vars_and_cols.items():
        fcol = flag_for(value_col)
        if fcol is None:
            missing_flags.append(name)
            out[f"raw_flag_{name}"] = ""
            ok = np.ones(len(out), dtype=bool)
        else:
            raw = df[fcol]
            out[f"raw_flag_{name}"] = raw.astype(str)
            ok = raw.map(_flag_number).isin(ACCEPTED_PRIMARY_FLAGS).to_numpy()
        out[f"qa_ok_{name}"] = ok
        qa_arrays.append(ok)

    numeric_ok = out[["temp", "sal", "do_mgl", "depth"]].notna().all(axis=1).to_numpy()
    qa_ok = np.logical_and.reduce(qa_arrays) if qa_arrays else np.ones(len(out), dtype=bool)
    out["qa_status"] = np.where(numeric_ok & qa_ok, "observed", "excluded")

    text = _event_text(df)
    for event_code in ["CDA", "CRE", "CWE", "CAB", "CFK", "CLT"]:
        out[f"event_{event_code.lower()}"] = text.str.contains(event_code, regex=False, na=False).to_numpy()

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp").reset_index(drop=True)

    expected = 35136 if pd.Timestamp(f"{year}-01-01").is_leap_year else 35040
    usable = out["qa_status"].eq("observed")
    times = pd.DatetimeIndex(out.loc[usable, "timestamp"])
    if len(times) > 1:
        gaps_h = pd.Series(times).diff().dt.total_seconds().div(3600)
        max_gap = float(gaps_h.max())
        n1, n6, n24 = (int((gaps_h > h).sum()) for h in (1, 6, 24))
    else:
        max_gap, n1, n6, n24 = np.nan, 0, 0, 0

    summary = {
        "station": station,
        "station_code": code,
        "year": year,
        "n_rows": len(out),
        "expected_15min_rows": expected,
        "raw_timestamp_coverage_pct": 100 * len(out) / expected,
        "usable_rows": int(usable.sum()),
        "usable_coverage_pct": 100 * usable.sum() / expected,
        "max_usable_gap_hours": max_gap,
        "n_gaps_gt_1h": n1,
        "n_gaps_gt_6h": n6,
        "n_gaps_gt_24h": n24,
        "hypoxia_lt3_n": int((out.loc[usable, "do_mgl"] < 3).sum()),
        "hypoxia_lt2_n": int((out.loc[usable, "do_mgl"] < 2).sum()),
        "cda_flag_n": int(out["event_cda"].sum()),
        "rain_flag_n": int(out["event_cre"].sum()),
        "weather_flag_n": int(out["event_cwe"].sum()),
        "bloom_flag_n": int(out["event_cab"].sum()),
        "missing_flag_columns": "|".join(missing_flags),
        "depth_source": depth_col,
    }
    return out, summary


def run(start_year: int, end_year: int, out_dir: Path) -> None:
    raw_dir = out_dir / "raw"
    canonical_dir = out_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    station_frames = {name: [] for name in STATIONS}

    for station, code in STATIONS.items():
        for year in range(start_year, end_year + 1):
            print(f"Downloading/reading {code} {year}")
            url = f"{BASE_URL}/{code}{year}.csv"
            path = raw_dir / f"{code}{year}.csv"
            try:
                _download(url, path)
                canon, summary = _canonicalize(_read_csv(path), station, code, year)
                station_frames[station].append(canon)
                summaries.append(summary)
            except Exception as exc:
                summaries.append({"station": station, "station_code": code, "year": year, "error": repr(exc)})
                print(f"WARNING {code} {year}: {exc}")

    for station, frames in station_frames.items():
        if not frames:
            continue
        full = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
        full.to_csv(canonical_dir / f"{station}.csv", index=False)
        full[full["qa_status"].eq("observed")].to_csv(canonical_dir / f"{station}_observed_only.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "elkhorn_qaqc_by_year.csv", index=False)
    valid = summary_df.dropna(subset=["usable_coverage_pct"]) if "usable_coverage_pct" in summary_df else pd.DataFrame()
    if not valid.empty:
        station_summary = valid.groupby(["station", "station_code"], as_index=False).agg(
            years=("year", "count"),
            median_usable_coverage_pct=("usable_coverage_pct", "median"),
            min_usable_coverage_pct=("usable_coverage_pct", "min"),
            total_usable_rows=("usable_rows", "sum"),
            total_hypoxia_lt3=("hypoxia_lt3_n", "sum"),
            total_cda_flags=("cda_flag_n", "sum"),
            max_gap_hours=("max_usable_gap_hours", "max"),
            total_gaps_gt_24h=("n_gaps_gt_24h", "sum"),
        )
        station_summary.to_csv(out_dir / "elkhorn_qaqc_station_summary.csv", index=False)
        print("\nStation QA/QC summary:\n" + station_summary.to_string(index=False))

    sm = canonical_dir / "south_marsh.csv"
    if sm.exists():
        target = Path("data/elkhorn")
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sm, target / "south_marsh.csv")
        shutil.copy2(sm, target / "south_marsh_reference.csv")

    print(f"Elkhorn archive preparation complete: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--out-dir", default="output/elkhorn_data_audit")
    args = parser.parse_args()
    run(args.start_year, args.end_year, Path(args.out_dir))
