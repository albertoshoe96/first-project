from __future__ import annotations

import argparse
import calendar
import logging
import math
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# =========================
# Config di default
# =========================

SOURCE_TABLE = "canonical_pod_modelbase_15m_v2_nosplit"
STATIC_TABLE = "dq_static_exog_portal_447_model_final_v1"
WEATHER_TABLE = "dq_openmeteo_hourly_weather_best49_v1"

POD_COL = "IdDevice"
TIME_COL = "DateUTC"
TARGET_COL = "Load_15m"

MONTHLY_TARGET = "y_monthly_kwh"

BASE_MONTHLY_DATASET_TABLE = "monthly_dataset"
BASE_FORECAST_TABLE = "monthly_forecast"
BASE_HORIZON_METRICS_TABLE = "monthly_horizon_metrics"
BASE_POD_METRICS_TABLE = "monthly_pod_metrics"

# Schema-resolution candidates (lowercase match) for robustness.
DEVICE_ID_CANDIDATES = ["IdDevice", "id_device", "device_id", "pod", "pod_id"]
GEO_KEY_CANDIDATES = [
    "geo_key", "geokey", "weather_geo_key", "geo_id", "best49_key",
    "weather_key", "location_key", "id_geo",
]
HAS_COORDS_CANDIDATES = ["has_coordinates", "has_coords", "has_geo", "coordinates_ok"]
WEATHER_DT_CANDIDATES = ["DateUTC", "datetime", "date_time", "time", "timestamp", "date", "ts"]
TEMP_CANDIDATES = ["temperature_2m", "temperature", "temp", "t2m", "air_temperature"]
HUMIDITY_CANDIDATES = ["relative_humidity_2m", "humidity", "relative_humidity", "rh"]
SOLAR_CANDIDATES = [
    "shortwave_radiation", "solar_radiation", "global_radiation", "ghi",
    "radiation", "direct_radiation",
]
PRECIP_CANDIDATES = ["precipitation", "precip", "rain", "total_precipitation", "rainfall"]

# Static-feature canonical lists + alias resolution.
STATIC_NUMERIC = ["contractual_power_kw", "has_pv", "pv_power_kw", "n_peripherals"]
STATIC_CATEGORICAL = ["pod_type", "customer_class", "region", "province", "zone", "cap"]
STATIC_ALIASES: dict[str, list[str]] = {
    "contractual_power_kw": [
        "contractual_power_kw", "contractual_power", "power_kw",
        "potenza_contrattuale", "contractualpower",
    ],
    "pod_type": ["pod_type", "podtype", "tipo_pod", "type"],
    "customer_class": ["customer_class", "customerclass", "classe_cliente", "customer"],
    "region": ["region", "regione"],
    "province": ["province", "provincia", "prov"],
    "zone": ["zone", "zona", "market_zone", "zona_mercato"],
    "has_pv": ["has_pv", "haspv", "has_photovoltaic", "pv_flag", "HasPvProduction"],
    "pv_power_kw": ["pv_power_kw", "pvpower", "potenza_pv", "pv_kw"],
    "n_peripherals": ["n_peripherals", "num_peripherals", "n_periferiche", "peripherals"],
    "cap": ["cap", "postal_code", "zip", "zipcode"],
}

CALENDAR_FEATURES = [
    "month_int", "quarter", "year", "days_in_month", "n_weekend_days",
    "n_working_days", "n_holidays", "n_holiday_bridges", "is_august",
    "is_december", "is_january", "month_sin", "month_cos", "year_progress",
]
WEATHER_FEATURES = [
    "avg_temperature", "min_temperature", "max_temperature", "temperature_range",
    "sum_HDD", "sum_CDD", "n_days_above_25C", "n_days_below_5C",
    "avg_solar_radiation", "sum_precipitation", "n_rainy_days", "avg_humidity",
]
AUTOREGRESSIVE_FEATURES = [
    "y_monthly_lag_1", "y_monthly_lag_2", "y_monthly_lag_3", "y_monthly_lag_12",
    "y_monthly_rolling_3", "y_monthly_rolling_12", "y_monthly_max_last_12",
    "y_monthly_min_last_12", "y_monthly_std_last_12", "ratio_y_vs_lag_12",
    "trend_last_3",
]

# Sottoinsieme persistito nella tabella forecast per audit.
AUDIT_FEATURES = [
    "month_int", "days_in_month", "n_working_days", "n_holidays",
    "contractual_power_kw", "pod_type", "customer_class", "region",
    "avg_temperature", "sum_HDD", "sum_CDD",
    "y_monthly_lag_1", "y_monthly_lag_12", "y_monthly_rolling_3",
    "y_monthly_rolling_12", "ratio_y_vs_lag_12",
]

# Un mese pieno ha 96*30=2880 obs (o 96*31=2976). 2500 ≈ 85% di copertura.
DEFAULT_MIN_MONTH_COVERAGE = 2500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lgbm_monthly_forecast_v1")


# =========================
# Utils
# =========================

def q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def open_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-524288;")  # ~512MB
    conn.execute("PRAGMA mmap_size=536870912;")  # 512MB
    return conn


def write_df(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    if_exists: str = "replace",
    chunksize: int = 100_000,
    create_indexes_sql: Optional[list[str]] = None,
) -> None:
    t0 = time.time()
    df.to_sql(
        table,
        conn,
        index=False,
        if_exists=if_exists,
        chunksize=chunksize,
        method=None,
    )
    if create_indexes_sql:
        cur = conn.cursor()
        for stmt in create_indexes_sql:
            cur.execute(stmt)
        conn.commit()
    log.info("Scritto %s (%s righe) in %.1f sec", table, f"{len(df):,}", time.time() - t0)


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        info = pd.read_sql_query(f"PRAGMA table_info({q(table)})", conn)
    except Exception:
        return []
    return info["name"].tolist() if "name" in info.columns else []


def pick(lower_map: dict[str, str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name present (case-insensitive)."""
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan,
            "wape_pct": np.nan, "smape_pct": np.nan, "bias": np.nan,
        }
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom_wape = float(np.sum(np.abs(y_true)))
    wape = float(100.0 * np.sum(abs_err) / denom_wape) if denom_wape > 0 else np.nan

    denom_smape = np.abs(y_true) + np.abs(y_pred)
    smape_arr = np.divide(
        2.0 * abs_err, denom_smape,
        out=np.zeros_like(abs_err, dtype=float),
        where=denom_smape != 0,
    )
    smape = float(100.0 * np.mean(smape_arr))
    bias = float(np.mean(err))

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "wape_pct": wape,
        "smape_pct": smape,
        "bias": bias,
    }


# =========================
# Build monthly dataset
# =========================

def build_monthly_dataset(
    conn: sqlite3.Connection,
    source_table: str,
    monthly_table: str,
    min_month_coverage: int,
    max_pods: Optional[int],
    date_from: Optional[str],
) -> pd.DataFrame:
    """Aggregate the 15-minute table into one row per (IdDevice, year_month)."""
    where = [f"{q(TARGET_COL)} IS NOT NULL"]
    params: list = []
    if date_from:
        where.append(f"{q(TIME_COL)} >= ?")
        params.append(date_from)

    pod_filter = ""
    if max_pods is not None:
        pod_filter = f"""
        AND {q(POD_COL)} IN (
            SELECT DISTINCT {q(POD_COL)}
            FROM {q(source_table)}
            ORDER BY {q(POD_COL)}
            LIMIT {int(max_pods)}
        )
        """

    create_sql = f"""
    CREATE TABLE {q(monthly_table)} AS
    SELECT
        {q(POD_COL)} AS {q(POD_COL)},
        strftime('%Y-%m', {q(TIME_COL)}) AS year_month,
        CAST(strftime('%Y', {q(TIME_COL)}) AS INTEGER) AS year,
        CAST(strftime('%m', {q(TIME_COL)}) AS INTEGER) AS month,
        SUM({q(TARGET_COL)}) AS {q(MONTHLY_TARGET)},
        COUNT(*) AS n_observations_15m,
        SUM(CASE WHEN {q(TARGET_COL)} IS NULL THEN 1 ELSE 0 END) AS n_null_15m
    FROM {q(source_table)}
    WHERE {" AND ".join(where)}
    {pod_filter}
    GROUP BY {q(POD_COL)}, year_month
    HAVING COUNT(*) >= {int(min_month_coverage)}
    ORDER BY {q(POD_COL)}, year_month
    """

    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {q(monthly_table)}")
    cur.execute(create_sql, params)
    conn.commit()

    df = pd.read_sql_query(f"SELECT * FROM {q(monthly_table)}", conn)
    df[POD_COL] = df[POD_COL].astype("int32")
    df["year_month"] = df["year_month"].astype(str)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df[MONTHLY_TARGET] = df[MONTHLY_TARGET].astype("float32")

    log.info(
        "Monthly dataset: %s righe, %s POD, mesi %s..%s (tabella %s)",
        f"{len(df):,}",
        df[POD_COL].nunique(),
        df["year_month"].min(),
        df["year_month"].max(),
        monthly_table,
    )
    return df


# =========================
# Feature engineering
# =========================

def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _italian_holidays_fallback(year: int) -> set[date]:
    es = _easter(year)
    em = es + timedelta(days=1)
    return {
        date(year, 1, 1), date(year, 1, 6), es, em,
        date(year, 4, 25), date(year, 5, 1), date(year, 6, 2),
        date(year, 8, 15), date(year, 11, 1), date(year, 12, 8),
        date(year, 12, 25), date(year, 12, 26),
    }


def build_holiday_set(years) -> set[date]:
    years_sorted = sorted(set(int(y) for y in years))
    try:
        import holidays as holidays_lib
        it = holidays_lib.Italy(years=years_sorted)
        return set(it.keys())
    except Exception:
        out: set[date] = set()
        for y in years_sorted:
            out |= _italian_holidays_fallback(y)
        return out


def add_calendar_features(df: pd.DataFrame, holiday_set: set[date]) -> pd.DataFrame:
    out = df.copy()
    unique_ym = sorted(out["year_month"].unique())
    rows = []
    for ym in unique_ym:
        year, month = int(ym[:4]), int(ym[5:7])
        n_days = calendar.monthrange(year, month)[1]
        days = [date(year, month, d) for d in range(1, n_days + 1)]

        def is_off(d: date) -> bool:
            return d.weekday() >= 5 or d in holiday_set

        n_weekend = sum(1 for d in days if d.weekday() >= 5)
        n_holidays = sum(1 for d in days if d in holiday_set)
        n_working = sum(
            1 for d in days
            if d.weekday() < 5 and d not in holiday_set
        )
        # A "bridge" working day is sandwiched between two days off, so taking
        # it off connects a holiday/weekend to another holiday/weekend.
        n_bridges = sum(
            1 for d in days
            if d.weekday() < 5 and d not in holiday_set
            and is_off(d - timedelta(days=1)) and is_off(d + timedelta(days=1))
        )
        rows.append({
            "year_month": ym,
            "month_int": month,
            "quarter": (month - 1) // 3 + 1,
            "year": year,
            "days_in_month": n_days,
            "n_weekend_days": n_weekend,
            "n_working_days": n_working,
            "n_holidays": n_holidays,
            "n_holiday_bridges": n_bridges,
            "is_august": int(month == 8),
            "is_december": int(month == 12),
            "is_january": int(month == 1),
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "year_progress": (month - 0.5) / 12,
        })
    cal = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in ("year", "month") if c in out.columns])
    return out.merge(cal, on="year_month", how="left")


def resolve_static_columns(conn: sqlite3.Connection):
    cols = table_columns(conn, STATIC_TABLE)
    if not cols:
        log.warning("Tabella %s non trovata: static features = NaN", STATIC_TABLE)
        return None, {}, None, None
    lower = {c.lower(): c for c in cols}
    device_col = pick(lower, DEVICE_ID_CANDIDATES)
    geo_col = pick(lower, GEO_KEY_CANDIDATES)
    has_coords_col = pick(lower, HAS_COORDS_CANDIDATES)
    resolved: dict[str, str] = {}
    for canon in STATIC_NUMERIC + STATIC_CATEGORICAL:
        src = pick(lower, STATIC_ALIASES.get(canon, [canon]))
        if src is not None:
            resolved[canon] = src
    return device_col, resolved, geo_col, has_coords_col


def add_static_features(df: pd.DataFrame, conn: sqlite3.Connection):
    """Join POD anagrafica. Returns (df, pod_geo_map)."""
    out = df.copy()
    device_col, resolved, geo_col, has_coords_col = resolve_static_columns(conn)

    # Pre-create canonical columns so the feature schema is stable.
    for canon in STATIC_NUMERIC + STATIC_CATEGORICAL:
        out[canon] = np.nan

    pod_geo = None
    if device_col is None:
        log.warning("Nessuna colonna device id risolta in %s: static = NaN", STATIC_TABLE)
        return out, pod_geo

    select_cols = [device_col] + list(resolved.values())
    if geo_col:
        select_cols.append(geo_col)
    if has_coords_col:
        select_cols.append(has_coords_col)
    select_cols = list(dict.fromkeys(select_cols))

    static = pd.read_sql_query(
        f"SELECT {', '.join(q(c) for c in select_cols)} FROM {q(STATIC_TABLE)}",
        conn,
    )
    static = static.rename(columns={device_col: POD_COL})
    static[POD_COL] = pd.to_numeric(static[POD_COL], errors="coerce").astype("Int32")
    static = static.dropna(subset=[POD_COL]).copy()
    static[POD_COL] = static[POD_COL].astype("int32")
    static = static.drop_duplicates(subset=POD_COL, keep="first")

    rename_map = {src: canon for canon, src in resolved.items()}
    static = static.rename(columns=rename_map)

    merge_cols = [POD_COL] + list(resolved.keys())
    out = out.drop(columns=[c for c in resolved.keys()])  # drop the NaN placeholders
    out = out.merge(static[merge_cols], on=POD_COL, how="left")

    for canon in STATIC_NUMERIC + STATIC_CATEGORICAL:
        if canon not in out.columns:
            out[canon] = np.nan

    if geo_col:
        gmap = static[[POD_COL, geo_col]].rename(columns={geo_col: "geo_key"})
        if has_coords_col and has_coords_col in static.columns:
            bad = static[has_coords_col].fillna(0).astype(float) == 0
            gmap.loc[bad.values, "geo_key"] = np.nan
        pod_geo = gmap

    log.info(
        "Static: risolte %s/%s feature; geo_key=%s; PODs con statiche=%s",
        len(resolved), len(STATIC_NUMERIC + STATIC_CATEGORICAL),
        geo_col, static[POD_COL].nunique(),
    )
    return out, pod_geo


def build_weather_monthly(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    cols = table_columns(conn, WEATHER_TABLE)
    if not cols:
        log.warning("Tabella %s non trovata: weather features = NaN", WEATHER_TABLE)
        return None
    lower = {c.lower(): c for c in cols}
    key_col = pick(lower, GEO_KEY_CANDIDATES)
    dt_col = pick(lower, WEATHER_DT_CANDIDATES)
    temp_col = pick(lower, TEMP_CANDIDATES)
    hum_col = pick(lower, HUMIDITY_CANDIDATES)
    solar_col = pick(lower, SOLAR_CANDIDATES)
    precip_col = pick(lower, PRECIP_CANDIDATES)

    if key_col is None or dt_col is None or temp_col is None:
        log.warning(
            "Weather: colonne geo_key/datetime/temperature non risolte; weather = NaN"
        )
        return None

    select = [key_col, dt_col, temp_col]
    for c in (hum_col, solar_col, precip_col):
        if c:
            select.append(c)
    select = list(dict.fromkeys(select))

    w = pd.read_sql_query(
        f"SELECT {', '.join(q(c) for c in select)} FROM {q(WEATHER_TABLE)}",
        conn,
    )
    w = w.rename(columns={key_col: "geo_key", dt_col: "dt", temp_col: "temp"})
    if hum_col:
        w = w.rename(columns={hum_col: "humidity"})
    if solar_col:
        w = w.rename(columns={solar_col: "solar"})
    if precip_col:
        w = w.rename(columns={precip_col: "precip"})

    w["dt"] = pd.to_datetime(w["dt"], errors="coerce")
    w = w.dropna(subset=["dt", "temp"])
    if w.empty:
        log.warning("Weather: nessuna riga utile dopo il parsing")
        return None
    w["year_month"] = w["dt"].dt.strftime("%Y-%m")
    w["day"] = w["dt"].dt.floor("D")
    w["hdd"] = (18.0 - w["temp"]).clip(lower=0)
    w["cdd"] = (w["temp"] - 22.0).clip(lower=0)

    g = w.groupby(["geo_key", "year_month"])
    monthly = pd.DataFrame({
        "avg_temperature": g["temp"].mean(),
        "min_temperature": g["temp"].min(),
        "max_temperature": g["temp"].max(),
        "sum_HDD": g["hdd"].sum(),
        "sum_CDD": g["cdd"].sum(),
    })
    monthly["temperature_range"] = monthly["max_temperature"] - monthly["min_temperature"]
    if "humidity" in w.columns:
        monthly["avg_humidity"] = g["humidity"].mean()
    if "solar" in w.columns:
        monthly["avg_solar_radiation"] = g["solar"].mean()
    if "precip" in w.columns:
        monthly["sum_precipitation"] = g["precip"].sum()

    daily_parts = {
        "day_tmax": w.groupby(["geo_key", "day"])["temp"].max(),
        "day_tmin": w.groupby(["geo_key", "day"])["temp"].min(),
    }
    if "precip" in w.columns:
        daily_parts["day_precip"] = w.groupby(["geo_key", "day"])["precip"].sum()
    daily = pd.DataFrame(daily_parts).reset_index()
    daily["year_month"] = daily["day"].dt.strftime("%Y-%m")
    daily["above25"] = (daily["day_tmax"] > 25).astype(int)
    daily["below5"] = (daily["day_tmin"] < 5).astype(int)
    dg = daily.groupby(["geo_key", "year_month"])
    day_monthly = pd.DataFrame({
        "n_days_above_25C": dg["above25"].sum(),
        "n_days_below_5C": dg["below5"].sum(),
    })
    if "day_precip" in daily.columns:
        daily["rainy"] = (daily["day_precip"] > 1.0).astype(int)
        day_monthly["n_rainy_days"] = (
            daily.groupby(["geo_key", "year_month"])["rainy"].sum()
        )

    monthly = monthly.join(day_monthly).reset_index()
    monthly["geo_key"] = monthly["geo_key"].astype(str)
    log.info(
        "Weather monthly: %s righe (geo_key, mese), %s geo_key distinti",
        f"{len(monthly):,}", monthly["geo_key"].nunique(),
    )
    return monthly


def add_weather_features(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    pod_geo: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = df.copy()
    weather_monthly = build_weather_monthly(conn)

    if weather_monthly is None or pod_geo is None:
        for c in WEATHER_FEATURES:
            out[c] = np.nan
        return out

    pod_geo = pod_geo.copy()
    pod_geo["geo_key"] = pod_geo["geo_key"].astype("string")
    out = out.merge(pod_geo, on=POD_COL, how="left")
    out["geo_key"] = out["geo_key"].astype("string")
    weather_monthly["geo_key"] = weather_monthly["geo_key"].astype("string")
    out = out.merge(weather_monthly, on=["geo_key", "year_month"], how="left")
    out = out.drop(columns=["geo_key"])
    for c in WEATHER_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out


def add_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly lags / rollings per POD su calendario continuo.

    Reindexing on a continuous monthly index guarantees that lag_12 always
    refers to the same calendar month one year earlier, even when intermediate
    months were dropped by the coverage filter (those become NaN, as expected).
    """
    out = df.copy()
    out["period"] = pd.PeriodIndex(out["year_month"], freq="M")
    parts = []
    for pod, g in out.groupby(POD_COL, sort=False):
        g = g.sort_values("period").set_index("period")
        full_idx = pd.period_range(g.index.min(), g.index.max(), freq="M")
        full_idx.name = "period"
        g = g.reindex(full_idx)
        y = g[MONTHLY_TARGET]
        ys = y.shift(1)
        g["y_monthly_lag_1"] = y.shift(1)
        g["y_monthly_lag_2"] = y.shift(2)
        g["y_monthly_lag_3"] = y.shift(3)
        g["y_monthly_lag_12"] = y.shift(12)
        g["y_monthly_rolling_3"] = ys.rolling(3, min_periods=1).mean()
        g["y_monthly_rolling_12"] = ys.rolling(12, min_periods=1).mean()
        g["y_monthly_max_last_12"] = ys.rolling(12, min_periods=1).max()
        g["y_monthly_min_last_12"] = ys.rolling(12, min_periods=1).min()
        g["y_monthly_std_last_12"] = ys.rolling(12, min_periods=2).std()
        g["ratio_y_vs_lag_12"] = g["y_monthly_lag_12"] / g["y_monthly_rolling_12"]
        g["trend_last_3"] = (
            (g["y_monthly_rolling_3"] - g["y_monthly_rolling_12"])
            / g["y_monthly_rolling_12"]
        )
        g[POD_COL] = pod
        g = g[g[MONTHLY_TARGET].notna()]  # drop synthetic calendar rows
        parts.append(g.reset_index())
    res = pd.concat(parts, ignore_index=True)
    res = res.replace([np.inf, -np.inf], np.nan)
    res = res.drop(columns=["period"])
    return res


def snapshot_raw_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Save string snapshots of categorical columns for audit before fitting
    the categorical schema (which will null out unseen-in-train values)."""
    out = df.copy()
    for c in STATIC_CATEGORICAL:
        if c in out.columns:
            raw = out[c].astype("string")
            out[f"_raw_{c}"] = raw
    return out


# =========================
# Split
# =========================

def split_pods(df: pd.DataFrame, cold_frac: float, seed: int) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    pods = np.array(sorted(df[POD_COL].unique()))
    rng.shuffle(pods)
    n_cold = max(1, int(round(len(pods) * cold_frac)))
    cold = set(int(x) for x in pods[:n_cold])
    train = set(int(x) for x in pods[n_cold:])
    return train, cold


def make_splits(
    df: pd.DataFrame,
    train_pods: set[int],
    cold_pods: set[int],
    temporal_test_months: int,
) -> dict[str, pd.DataFrame]:
    """Per-POD temporal split among known PODs; cold PODs held out entirely."""
    train_pool = df[df[POD_COL].isin(train_pods)].copy()
    cold = df[df[POD_COL].isin(cold_pods)].copy()

    train_parts, val_parts, test_parts = [], [], []
    for _, g in train_pool.groupby(POD_COL, sort=False):
        months = sorted(g["year_month"].unique())
        cut = max(0, len(months) - temporal_test_months) if temporal_test_months > 0 else len(months)
        test_m = months[cut:]
        rest = months[:cut]
        val_m = rest[-1:]
        train_m = rest[:-1]
        ym = g["year_month"]
        train_parts.append(g[ym.isin(train_m)])
        val_parts.append(g[ym.isin(val_m)])
        test_parts.append(g[ym.isin(test_m)])

    empty = train_pool.iloc[0:0]
    train = pd.concat(train_parts, ignore_index=True) if train_parts else empty
    val = pd.concat(val_parts, ignore_index=True) if val_parts else empty
    temporal = pd.concat(test_parts, ignore_index=True) if test_parts else empty

    return {
        "train": train,
        "val": val,
        "temporal": temporal,
        "cold_no_history": cold,
    }


# =========================
# Feature columns + categoricals
# =========================

def get_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (feature_cols, categorical_cols, autoregressive_cols).

    Drops feature columns that are entirely NaN in the dataset to avoid
    LightGBM issues. The model never receives IdDevice as a feature: cold
    PODs would otherwise be mapped to NaN by the train-only categorical
    schema, defeating the purpose of generalisation.
    """
    candidates = (
        CALENDAR_FEATURES
        + STATIC_NUMERIC + STATIC_CATEGORICAL
        + WEATHER_FEATURES
        + AUTOREGRESSIVE_FEATURES
    )
    candidates = [c for c in candidates if c in df.columns]
    nonnull = df[candidates].notna().sum()
    feature_cols = [c for c in candidates if nonnull[c] > 0]
    dropped = [c for c in candidates if nonnull[c] == 0]
    if dropped:
        log.info("Drop colonne tutte-NaN: %s", dropped)
    cat_cols = [c for c in STATIC_CATEGORICAL if c in feature_cols]
    ar_cols = [c for c in AUTOREGRESSIVE_FEATURES if c in feature_cols]
    return feature_cols, cat_cols, ar_cols


def fit_categorical_schema(train_df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list]:
    """Categorie definite SOLO sul training: i valori non visti diventano NaN."""
    schema: dict[str, list] = {}
    for c in categorical_cols:
        schema[c] = sorted(train_df[c].dropna().astype(str).unique().tolist())
    return schema


def apply_categorical_schema(dfs: list[pd.DataFrame], schema: dict[str, list]) -> None:
    for d in dfs:
        for c, cats in schema.items():
            if c in d.columns:
                col = d[c].astype("string")
                mask_known = col.isin(cats)
                col = col.where(mask_known, other=pd.NA)
                d[c] = pd.Categorical(col, categories=cats)


# =========================
# Training
# =========================

def train_regressor(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    objective: str = "regression",
) -> lgb.Booster:
    if train.empty:
        raise ValueError("Training split vuoto: impossibile addestrare.")

    dtrain = lgb.Dataset(
        train[feature_cols],
        label=train[MONTHLY_TARGET].astype("float32"),
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=100)]

    if not val.empty:
        dval = lgb.Dataset(
            val[feature_cols],
            label=val[MONTHLY_TARGET].astype("float32"),
            categorical_feature=categorical_cols,
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    else:
        log.warning("Validation vuota: training senza early stopping.")

    params = {
        "objective": objective,
        "metric": ["l1", "rmse"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


# =========================
# Predict + frames
# =========================

def predict_one(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    mask_history_cols: Optional[list[str]] = None,
) -> np.ndarray:
    """Predict and clip at zero; optionally mask AR features to NaN (cold-start)."""
    x = df[feature_cols].copy()
    if mask_history_cols:
        for c in mask_history_cols:
            if c in x.columns:
                x[c] = np.nan
    raw = model.predict(x, num_iteration=model.best_iteration)
    return np.maximum(np.asarray(raw, dtype="float32"), 0.0)


def build_forecast_frame(
    eval_df: pd.DataFrame,
    y_pred: np.ndarray,
    eval_name: str,
) -> pd.DataFrame:
    audit_present = [c for c in AUDIT_FEATURES if c in eval_df.columns]
    base = pd.DataFrame({
        POD_COL: eval_df[POD_COL].astype("int32").to_numpy(),
        "year_month": eval_df["year_month"].astype(str).to_numpy(),
        "y_monthly_true": eval_df[MONTHLY_TARGET].astype("float32").to_numpy(),
        "y_monthly_pred": np.asarray(y_pred, dtype="float32"),
        "eval_set": eval_name,
    })
    for c in audit_present:
        src = f"_raw_{c}" if f"_raw_{c}" in eval_df.columns else c
        val = eval_df[src]
        if isinstance(val.dtype, pd.CategoricalDtype):
            val = val.astype("string")
        base[c] = val.to_numpy()

    err = base["y_monthly_pred"] - base["y_monthly_true"]
    base["error_kwh"] = err.astype("float32")
    base["abs_error_kwh"] = err.abs().astype("float32")
    base["error_pct"] = np.where(
        base["y_monthly_true"] != 0,
        err / base["y_monthly_true"] * 100.0,
        np.nan,
    ).astype("float32")
    return base


def build_horizon_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for es, g in forecast.groupby("eval_set", sort=False):
        m = compute_metrics(g["y_monthly_true"].values, g["y_monthly_pred"].values)
        rows.append({"eval_set": es, **m})
    return pd.DataFrame(rows)


def build_pod_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pod, es), g in forecast.groupby([POD_COL, "eval_set"], sort=False):
        m = compute_metrics(g["y_monthly_true"].values, g["y_monthly_pred"].values)
        rows.append({POD_COL: int(pod), "eval_set": es, **m})
    return pd.DataFrame(rows)


# =========================
# Validation reporting
# =========================

def report_target_sanity(df: pd.DataFrame) -> None:
    y = df[MONTHLY_TARGET].astype(float)
    n_zero = int((y == 0).sum())
    n_neg = int((y < 0).sum())
    log.info(
        "Target: n=%s mean=%.1f median=%.1f min=%.1f max=%.1f zeros=%s neg=%s",
        f"{len(y):,}", y.mean(), y.median(), y.min(), y.max(), n_zero, n_neg,
    )
    pos = y[y > 0]
    if len(pos) > 5:
        log.info(
            "log10(target>0): mean=%.2f std=%.2f (atteso roughly log-normale)",
            np.log10(pos).mean(), np.log10(pos).std(),
        )


def report_static_coverage(df: pd.DataFrame) -> None:
    log.info("Coverage feature statiche:")
    for c in STATIC_NUMERIC + STATIC_CATEGORICAL:
        if c in df.columns:
            pct = df[c].notna().mean() * 100
            log.info("  %-24s %6.1f%% popolato", c, pct)


def report_feature_importance(model: lgb.Booster, top_n: int = 15) -> None:
    names = model.feature_name()
    gains = model.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1][:top_n]
    log.info("Top-%s feature per gain:", top_n)
    for i in order:
        log.info("  %-24s %.0f", names[i], gains[i])


# =========================
# Pipeline
# =========================

def run_pipeline(args: argparse.Namespace) -> int:
    t0 = time.time()
    conn = open_conn(args.db)

    suffix = args.output_suffix
    monthly_table = (
        args.monthly_dataset_table or f"{BASE_MONTHLY_DATASET_TABLE}{suffix}"
    )
    forecast_table = args.output_forecast_table or f"{BASE_FORECAST_TABLE}{suffix}"
    horizon_table = (
        args.output_horizon_metrics_table or f"{BASE_HORIZON_METRICS_TABLE}{suffix}"
    )
    pod_metrics_table = (
        args.output_pod_metrics_table or f"{BASE_POD_METRICS_TABLE}{suffix}"
    )

    # Build the monthly dataset.
    df = build_monthly_dataset(
        conn,
        source_table=args.source_table,
        monthly_table=monthly_table,
        min_month_coverage=args.min_month_coverage,
        max_pods=args.max_pods,
        date_from=args.date_from,
    )

    # Filter PODs with too little history (data-quality gate).
    months_per_pod = df.groupby(POD_COL)["year_month"].nunique()
    keep = months_per_pod[months_per_pod >= args.min_history_months].index
    n_before = df[POD_COL].nunique()
    df = df[df[POD_COL].isin(keep)].copy()
    log.info(
        "Mantenuti %s/%s POD con >= %s mesi di storia",
        df[POD_COL].nunique(), n_before, args.min_history_months,
    )
    if df.empty:
        raise RuntimeError("Dataset vuoto dopo i filtri. Verifica input e parametri.")

    report_target_sanity(df)

    # Feature engineering.
    df = add_autoregressive_features(df)  # before split, per spec
    holiday_set = build_holiday_set(df["year"].unique())
    df = add_calendar_features(df, holiday_set)
    df, pod_geo = add_static_features(df, conn)
    df = add_weather_features(df, conn, pod_geo)
    report_static_coverage(df)

    # Snapshot raw categorical strings so the audit values survive the
    # train-only categorical schema applied below.
    df = snapshot_raw_categoricals(df)

    feature_cols, categorical_cols, ar_cols = get_feature_cols(df)
    log.info(
        "Features=%s (categorical=%s, AR=%s)",
        len(feature_cols), len(categorical_cols), len(ar_cols),
    )

    # POD-level + temporal split.
    train_pods, cold_pods = split_pods(df, args.cold_test_frac, args.seed)
    log.info("POD train=%s | POD cold=%s", len(train_pods), len(cold_pods))
    splits = make_splits(df, train_pods, cold_pods, args.temporal_test_months)
    train, val = splits["train"], splits["val"]
    temporal, cold = splits["temporal"], splits["cold_no_history"]
    log.info(
        "Righe: train=%s | val=%s | temporal=%s | cold_no_history=%s",
        f"{len(train):,}", f"{len(val):,}", f"{len(temporal):,}", f"{len(cold):,}",
    )

    if train.empty:
        raise RuntimeError(
            "Training split vuoto. Aumenta --min-history-months o riduci --temporal-test-months."
        )

    # Categorie fit SOLO sul training (cold-start realistico).
    schema = fit_categorical_schema(train, categorical_cols)
    apply_categorical_schema([train, val, temporal, cold], schema)

    # Train.
    model = train_regressor(
        train, val, feature_cols, categorical_cols,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        objective=args.reg_objective,
    )
    log.info("best_iteration=%s | trees=%s", model.best_iteration, model.num_trees())
    report_feature_importance(model)

    # Predict on each eval set (cold uses NaN-masked AR features).
    eval_sets = [
        ("temporal", temporal, None),
        ("cold_no_history", cold, ar_cols),
    ]
    all_forecasts: list[pd.DataFrame] = []
    for eval_name, eval_df, mask_cols in eval_sets:
        if eval_df.empty:
            log.warning("Eval set %s vuoto, salto", eval_name)
            continue
        y_pred = predict_one(model, eval_df, feature_cols, mask_history_cols=mask_cols)
        fc = build_forecast_frame(eval_df, y_pred, eval_name)
        all_forecasts.append(fc)

    if not all_forecasts:
        raise RuntimeError("Nessun forecast prodotto. Verifica split e parametri.")

    forecast = pd.concat(all_forecasts, ignore_index=True)
    horizon = build_horizon_metrics(forecast)
    pod_metrics = build_pod_metrics(forecast)

    forecast_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{forecast_table}_eval "
        f"ON {q(forecast_table)} (eval_set)",
        f"CREATE INDEX IF NOT EXISTS idx_{forecast_table}_pod_ym "
        f"ON {q(forecast_table)} ({q(POD_COL)}, year_month)",
    ]
    horizon_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{horizon_table}_eval "
        f"ON {q(horizon_table)} (eval_set)",
    ]
    pod_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{pod_metrics_table}_pod_eval "
        f"ON {q(pod_metrics_table)} ({q(POD_COL)}, eval_set)",
    ]

    write_df(
        conn, forecast, forecast_table,
        chunksize=args.write_chunksize,
        create_indexes_sql=forecast_indexes,
    )
    write_df(
        conn, horizon, horizon_table,
        chunksize=args.write_chunksize,
        create_indexes_sql=horizon_indexes,
    )
    write_df(
        conn, pod_metrics, pod_metrics_table,
        chunksize=args.write_chunksize,
        create_indexes_sql=pod_indexes,
    )

    log.info("Output scritto su SQLite:")
    log.info("- %s", monthly_table)
    log.info("- %s", forecast_table)
    log.info("- %s", horizon_table)
    log.info("- %s", pod_metrics_table)
    log.info("Runtime %.1f sec", time.time() - t0)

    print("\n" + "=" * 100)
    print("RIEPILOGO LIGHTGBM MONTHLY FORECAST V1 — regressione mensile aggregata per POD")
    print("=" * 100)
    print(horizon.to_string(index=False))
    print("=" * 100)

    conn.close()
    return 0


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LightGBM monthly POD forecast V1: regressione del consumo "
            "totale mensile (kWh) per POD noti e POD nuovi."
        )
    )

    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--source-table", type=str, default=SOURCE_TABLE)
    p.add_argument("--max-pods", type=int, default=None,
                   help="Cap opzionale al numero di POD (per test rapidi).")
    p.add_argument("--date-from", type=str, default=None,
                   help="Filtro inferiore sulla data (formato YYYY-MM-DD).")

    p.add_argument("--temporal-test-months", type=int, default=2,
                   help="Mesi finali per POD usati come test temporale.")
    p.add_argument("--cold-test-frac", type=float, default=0.15,
                   help="Frazione di POD tenuti fuori come cold_no_history.")
    p.add_argument("--min-month-coverage", type=int, default=DEFAULT_MIN_MONTH_COVERAGE,
                   help="Osservazioni 15min minime per accettare un mese.")
    p.add_argument("--min-history-months", type=int, default=6,
                   help="POD con meno di N mesi di storia vengono esclusi.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-boost-round", type=int, default=2000)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument(
        "--reg-objective", type=str, default="regression",
        choices=["regression", "regression_l1", "huber", "fair"],
    )

    p.add_argument("--output-suffix", type=str, default="_v1")
    p.add_argument("--monthly-dataset-table", type=str, default=None)
    p.add_argument("--output-forecast-table", type=str, default=None)
    p.add_argument("--output-horizon-metrics-table", type=str, default=None)
    p.add_argument("--output-pod-metrics-table", type=str, default=None)
    p.add_argument("--write-chunksize", type=int, default=100_000)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
