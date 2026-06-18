from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

# =========================
# Config di default
# =========================

SOURCE_TABLE = "canonical_pod_modelbase_15m_v2_nosplit"
POD_COL = "IdDevice"
SEGMENT_COL = "PodSegmentId"
TIME_COL = "DateUTC"
TARGET_COL = "Load_15m"

EXOG_COLS = [
    "contractual_power_kw",
    "HasProduction",
    "BuildingLatitudine",
    "BuildingLongitudine",
    "HasPvProduction",
    "HasStorage",
    "HasEvCharger",
    "HasHeatPump",
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "precipitation_hourly",
    "precipitation_15m_rate",
    "rain_hourly",
    "rain_15m_rate",

    "local_hour",
    "local_dow",
    "is_italian_holiday",
    "is_saturday_local",
    "is_sunday_local",
    "tariff_band_code",
    "tariff_f1",
    "tariff_f2",
    "tariff_f3",
    "tariff_f23",
]

BASE_OUTPUT_FORECAST_TABLE = "lgbm_pod_two_stage_forecast_15m"
BASE_OUTPUT_METRICS_TABLE = "lgbm_pod_two_stage_metrics_15m"
BASE_OUTPUT_THRESHOLD_TABLE = "lgbm_pod_two_stage_thresholds_15m"

# === Quantile forecasting V1 ===
QUANTILES_DEFAULT = [0.10, 0.50, 0.90]
COVERAGE_TARGET_DEFAULT = 80.0
BASE_QUANTILE_FORECAST_TABLE = "twostage_normscale_quantile_forecast"
BASE_QUANTILE_METRICS_TABLE = "twostage_normscale_quantile_metrics"
CROSSING_WARN_PCT = 5.0

SLOTS_PER_DAY = 96
DEFAULT_LAGS = [1, 2, 4, 8, 24, 48, 96, 192, 672]
DEFAULT_ROLLINGS = [24, 96, 672]
DEFAULT_HORIZONS = [4, 24, 96]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lgbm_two_stage_normscale_quantile_v1")


# =========================
# Utils
# =========================

def q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def horizon_label(h: int) -> str:
    if h == 4:
        return "1h"
    if h == 24:
        return "6h"
    if h == 96:
        return "1d"
    return f"{h * 15}m"


def quantile_slot(alpha: float) -> str:
    return f"q{int(round(alpha * 100)):02d}"


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
    chunksize: int = 1_000,
    create_indexes_sql: Optional[Sequence[str]] = None,
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "wape_pct": np.nan, "smape_pct": np.nan, "bias": np.nan}

    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom_wape = np.sum(np.abs(y_true))
    wape = float(100.0 * np.sum(abs_err) / denom_wape) if denom_wape > 0 else np.nan

    denom_smape = np.abs(y_true) + np.abs(y_pred)
    smape_arr = np.divide(
        2.0 * abs_err,
        denom_smape,
        out=np.zeros_like(abs_err, dtype=float),
        where=denom_smape != 0,
    )
    smape = float(100.0 * np.mean(smape_arr))
    bias = float(np.mean(err))

    return {
        "mae": mae,
        "rmse": rmse,
        "wape_pct": wape,
        "smape_pct": smape,
        "bias": bias,
    }


# =========================
# Load + feature engineering
# =========================

def load_data(
    conn: sqlite3.Connection,
    source_table: str,
    max_pods: Optional[int],
    date_from: Optional[str],
) -> pd.DataFrame:
    where = ["1=1"]
    params: list = []

    if date_from:
        where.append(f"{q(TIME_COL)} >= ?")
        params.append(date_from)

    pod_filter = ""
    if max_pods is not None:
        pod_filter = f"""
        AND {q(POD_COL)} IN (
            SELECT {q(POD_COL)}
            FROM (
                SELECT DISTINCT {q(POD_COL)}
                FROM {q(source_table)}
                ORDER BY {q(POD_COL)}
                LIMIT {int(max_pods)}
            )
        )
        """

    sql = f"""
    SELECT *
    FROM {q(source_table)}
    WHERE {" AND ".join(where)}
    {pod_filter}
    """

    log.info("Carico dati da %s", source_table)
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=[TIME_COL])

    df[POD_COL] = df[POD_COL].astype("int32")
    df[SEGMENT_COL] = df[SEGMENT_COL].astype("int32")
    df[TARGET_COL] = df[TARGET_COL].astype("float32")

    df = df.sort_values([POD_COL, SEGMENT_COL, TIME_COL]).reset_index(drop=True)

    log.info(
        "Caricate %s righe, %s POD, %s segmenti",
        f"{len(df):,}",
        df[POD_COL].nunique(),
        df[[POD_COL, SEGMENT_COL]].drop_duplicates().shape[0],
    )

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out[TIME_COL]

    hour = t.dt.hour.astype("int16")
    dow = t.dt.dayofweek.astype("int16")
    month = t.dt.month.astype("int16")
    slot = (t.dt.hour * 4 + t.dt.minute // 15).astype("int16")

    out["hour_utc"] = hour
    out["day_of_week_utc"] = dow
    out["month_utc"] = month
    out["is_weekend_utc"] = dow.isin([5, 6]).astype("int8")
    out["slot_15m_in_day"] = slot

    out["slot_sin"] = np.sin(2 * np.pi * slot / SLOTS_PER_DAY).astype("float32")
    out["slot_cos"] = np.cos(2 * np.pi * slot / SLOTS_PER_DAY).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")
    out["month_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
    out["month_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")

    return out


def add_history_features(df: pd.DataFrame, lags: list[int], rollings: list[int]) -> pd.DataFrame:
    out = df.sort_values([POD_COL, SEGMENT_COL, TIME_COL]).reset_index(drop=True).copy()
    g = out.groupby([POD_COL, SEGMENT_COL], sort=False)[TARGET_COL]

    for lag in lags:
        lag_col = f"lag_{lag}_pod"
        out[lag_col] = g.shift(lag).astype("float32")
        out[f"lag_{lag}_available"] = out[lag_col].notna().astype("int8")
        out[f"lag_{lag}_is_positive"] = (out[lag_col] > 0).astype("int8")
        out.loc[out[lag_col].isna(), f"lag_{lag}_is_positive"] = 0

    shifted = g.shift(1)
    shifted_g = shifted.groupby([out[POD_COL], out[SEGMENT_COL]], sort=False)

    shifted_pos = (shifted > 0).astype("float32")
    shifted_pos_g = shifted_pos.groupby([out[POD_COL], out[SEGMENT_COL]], sort=False)

    for w in rollings:
        minp = max(2, min(w, 8))

        out[f"rolling_mean_{w}_pod"] = (
            shifted_g.rolling(w, min_periods=minp)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )

        out[f"rolling_std_{w}_pod"] = (
            shifted_g.rolling(w, min_periods=minp)
            .std()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )

        out[f"rolling_positive_rate_{w}_pod"] = (
            shifted_pos_g.rolling(w, min_periods=minp)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )

        out[f"rolling_zero_rate_{w}_pod"] = (
            1.0 - out[f"rolling_positive_rate_{w}_pod"]
        ).astype("float32")

        out[f"rolling_{w}_available"] = out[f"rolling_mean_{w}_pod"].notna().astype("int8")

    return out


def add_direct_target(df: pd.DataFrame, h: int) -> pd.DataFrame:
    out = df.sort_values([POD_COL, SEGMENT_COL, TIME_COL]).reset_index(drop=True).copy()
    g = out.groupby([POD_COL, SEGMENT_COL], sort=False)

    out["t_target"] = g[TIME_COL].shift(-h)
    out["y_target"] = g[TARGET_COL].shift(-h).astype("float32")

    expected_delta = pd.Timedelta(minutes=15 * h)

    out = out[out["t_target"].notna()].copy()
    out = out[(out["t_target"] - out[TIME_COL]) == expected_delta].copy()
    out = out.dropna(subset=["y_target"]).copy()
    out["target_positive"] = (out["y_target"] > 0).astype("int8")

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
    val_days: int,
    temporal_test_days: int,
) -> dict[str, pd.DataFrame]:
    train_pool = df[df[POD_COL].isin(train_pods)].copy()
    cold = df[df[POD_COL].isin(cold_pods)].copy()

    max_t = train_pool["t_target"].max()
    temporal_start = max_t - pd.Timedelta(days=temporal_test_days)
    val_start = temporal_start - pd.Timedelta(days=val_days)

    train = train_pool[train_pool["t_target"] < val_start].copy()
    val = train_pool[
        (train_pool["t_target"] >= val_start)
        & (train_pool["t_target"] < temporal_start)
    ].copy()
    temporal = train_pool[train_pool["t_target"] >= temporal_start].copy()

    return {
        "train": train,
        "val": val,
        "temporal": temporal,
        "cold_with_history": cold.copy(),
        "cold_no_history": cold.copy(),
    }


# =========================
# Feature columns + categoricals
# =========================

def get_feature_cols(
    lags: list[int],
    rollings: list[int],
    use_pod_id: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    calendar_cols = [
        "hour_utc",
        "day_of_week_utc",
        "month_utc",
        "is_weekend_utc",
        "slot_15m_in_day",
        "slot_sin",
        "slot_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    history_cols = (
        [f"lag_{lag}_pod" for lag in lags]
        + [f"lag_{lag}_available" for lag in lags]
        + [f"lag_{lag}_is_positive" for lag in lags]
        + [f"rolling_mean_{w}_pod" for w in rollings]
        + [f"rolling_std_{w}_pod" for w in rollings]
        + [f"rolling_positive_rate_{w}_pod" for w in rollings]
        + [f"rolling_zero_rate_{w}_pod" for w in rollings]
        + [f"rolling_{w}_available" for w in rollings]
    )

    categorical_cols = [SEGMENT_COL]
    if use_pod_id:
        categorical_cols = [POD_COL] + categorical_cols

    feature_cols = categorical_cols + calendar_cols + history_cols + EXOG_COLS

    return feature_cols, categorical_cols, history_cols


def fit_categorical_schema(train_df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list]:
    """
    Categorie definite SOLO sul training.
    Questo rende il cold test più realistico: i POD non visti diventano NaN.
    """
    schema: dict[str, list] = {}

    for c in categorical_cols:
        schema[c] = sorted(train_df[c].dropna().unique().tolist())

    return schema


def apply_categorical_schema(dfs: list[pd.DataFrame], schema: dict[str, list]) -> None:
    """
    Applica lo schema categorico fit SOLO sul training.

    I valori non presenti nelle categorie di training vengono esplicitamente
    convertiti a NaN prima della conversione a Categorical.
    """
    for d in dfs:
        for c, cats in schema.items():
            if c in d.columns:
                mask_known = d[c].isin(cats)
                d.loc[~mask_known, c] = np.nan
                d[c] = pd.Categorical(d[c], categories=cats)


# =========================
# Anchor (normscale)
# =========================

def compute_positive_scale_anchor(df: pd.DataFrame) -> np.ndarray:
    """Scala storica riga-per-riga per il regressore positivo (normscale)."""
    candidates = [
        "rolling_mean_96_pod",
        "lag_96_pod",
        "rolling_mean_24_pod",
        "lag_24_pod",
        "lag_192_pod",
        "lag_672_pod",
        "rolling_mean_672_pod",
    ]
    anchor = np.full(len(df), np.nan, dtype="float32")
    for c in candidates:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32")
        good = np.isfinite(v) & (v > 0)
        fill = np.isnan(anchor) & good
        anchor[fill] = v[fill]
    anchor = np.where(np.isfinite(anchor) & (anchor > 0), anchor, 1.0).astype("float32")
    anchor = np.clip(anchor, 1.0, None).astype("float32")
    return anchor


# =========================
# Training
# =========================

def train_classifier(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    pos = float(train["target_positive"].sum())
    neg = float(len(train) - pos)
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    log.info(
        "Class imbalance: pos=%s neg=%s scale_pos_weight=%.3f",
        f"{int(pos):,}",
        f"{int(neg):,}",
        scale_pos_weight,
    )

    dtrain = lgb.Dataset(
        train[feature_cols],
        label=train["target_positive"],
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    dval = lgb.Dataset(
        val[feature_cols],
        label=val["target_positive"],
        categorical_feature=categorical_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train_cls", "val_cls"],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )


def _prepare_positive_train_val(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sottoinsieme positivo + anchor + target log1p normalizzato + sample weights."""
    train_pos = train[train["y_target"] > 0].copy()
    val_pos = val[val["y_target"] > 0].copy()
    if train_pos.empty or val_pos.empty:
        raise ValueError(
            "Train o validation positiva vuota: impossibile addestrare il regressore positivo."
        )

    train_anchor = compute_positive_scale_anchor(train_pos)
    val_anchor = compute_positive_scale_anchor(val_pos)

    y_train_raw = train_pos["y_target"].values.astype("float32")
    y_val_raw = val_pos["y_target"].values.astype("float32")
    y_train = np.log1p(y_train_raw / train_anchor)
    y_val = np.log1p(y_val_raw / val_anchor)

    train_ratio = y_train_raw / train_anchor
    good_ratio = train_ratio[np.isfinite(train_ratio) & (train_ratio > 0)]
    median_ratio = float(np.nanmedian(good_ratio)) if len(good_ratio) > 0 else 1.0
    median_ratio = max(median_ratio, 1e-6)

    reg_weight = np.sqrt(np.maximum(train_ratio / median_ratio, 0.0))
    reg_weight = np.clip(reg_weight, 1.0, 5.0).astype("float32")

    log.info(
        "Positive normscale: train_anchor median=%.4f p90=%.4f | val_anchor median=%.4f p90=%.4f | "
        "weight mean=%.4f p90=%.4f max=%.4f",
        float(np.nanmedian(train_anchor)),
        float(np.nanpercentile(train_anchor, 90)),
        float(np.nanmedian(val_anchor)),
        float(np.nanpercentile(val_anchor, 90)),
        float(np.nanmean(reg_weight)),
        float(np.nanpercentile(reg_weight, 90)),
        float(np.nanmax(reg_weight)),
    )

    return train_pos, val_pos, y_train, y_val, reg_weight, val_anchor


def train_regressor_positive(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    objective: str = "regression_l1",
) -> lgb.Booster:
    """Regressore puntuale positivo (esistente, invariato dal modello normscale)."""
    train_pos, val_pos, y_train, y_val, reg_weight, _ = _prepare_positive_train_val(train, val)

    dtrain = lgb.Dataset(
        train_pos[feature_cols],
        label=y_train,
        weight=reg_weight,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_pos[feature_cols],
        label=y_val,
        categorical_feature=categorical_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": objective,
        "metric": ["l1", "rmse"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train_reg_pos", "val_reg_pos"],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )


def train_regressor_quantile(
    train_pos: pd.DataFrame,
    val_pos: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    reg_weight: np.ndarray,
    feature_cols: list[str],
    categorical_cols: list[str],
    alpha: float,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    """Regressore quantile su log1p(y/anchor): stessa pipeline normscale, alpha variabile.

    Parametri presi dalla spec (V1 quantile forecasting). I tre regressori sono
    identici tra loro a meno di `alpha` e `seed` (per diversificare il bootstrap).
    """
    dtrain = lgb.Dataset(
        train_pos[feature_cols],
        label=y_train,
        weight=reg_weight,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_pos[feature_cols],
        label=y_val,
        categorical_feature=categorical_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": "quantile",
        "alpha": float(alpha),
        "metric": ["l1"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": int(seed) + int(round(alpha * 100)),
        "num_threads": -1,
    }

    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=[f"train_q{int(round(alpha * 100)):02d}",
                     f"val_q{int(round(alpha * 100)):02d}"],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )


def train_quantile_regressors(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    quantiles: list[float],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> dict[float, lgb.Booster]:
    """Allena un regressore per ciascun quantile, condividendo dataset positivo e weights."""
    train_pos, val_pos, y_train, y_val, reg_weight, _ = _prepare_positive_train_val(train, val)

    models: dict[float, lgb.Booster] = {}
    for alpha in quantiles:
        log.info("Training quantile regressor alpha=%.2f", alpha)
        models[float(alpha)] = train_regressor_quantile(
            train_pos=train_pos,
            val_pos=val_pos,
            y_train=y_train,
            y_val=y_val,
            reg_weight=reg_weight,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            alpha=alpha,
            seed=seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
    return models


# =========================
# Predict + threshold
# =========================

def predict_two_stage(
    cls_model: lgb.Booster,
    reg_model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    mask_history_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Predizione puntuale two-stage (modello esistente, invariato)."""
    x = df[feature_cols].copy()
    if mask_history_cols:
        for c in mask_history_cols:
            if c in x.columns:
                x[c] = np.nan

    p_positive = cls_model.predict(x, num_iteration=cls_model.best_iteration)
    pred_log = reg_model.predict(x, num_iteration=reg_model.best_iteration)
    positive_scale_anchor = compute_positive_scale_anchor(x)
    pred_positive_norm = np.maximum(np.expm1(pred_log), 0.0)
    pred_positive = pred_positive_norm * positive_scale_anchor

    y_pred = np.where(p_positive >= threshold, pred_positive, 0.0)

    out = df[[POD_COL, SEGMENT_COL, TIME_COL, "t_target", "y_target"]].copy()
    out["p_positive"] = p_positive.astype("float32")
    out["positive_scale_anchor"] = positive_scale_anchor.astype("float32")
    out["y_pred_positive_norm"] = pred_positive_norm.astype("float32")
    out["y_pred_positive"] = pred_positive.astype("float32")
    out["threshold_positive"] = float(threshold)
    out["y_pred"] = y_pred.astype("float32")
    out["error"] = out["y_pred"] - out["y_target"]
    out["abs_error"] = np.abs(out["error"])
    return out


def predict_two_stage_quantile(
    cls_model: lgb.Booster,
    quantile_models: dict[float, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    quantiles_sorted: list[float],
    mask_history_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Predizione two-stage quantile.

    Per ogni quantile: log1p^-1 * anchor, clip a 0. Su righe classificate
    zero, tutti i quantili sono 0. Su righe positive applichiamo monotonia
    (sort lungo gli alpha) per eliminare il crossing.
    """
    x = df[feature_cols].copy()
    if mask_history_cols:
        for c in mask_history_cols:
            if c in x.columns:
                x[c] = np.nan

    p_positive = cls_model.predict(x, num_iteration=cls_model.best_iteration)
    anchor = compute_positive_scale_anchor(x)

    raw_preds = np.zeros((len(df), len(quantiles_sorted)), dtype="float32")
    for i, alpha in enumerate(quantiles_sorted):
        m = quantile_models[alpha]
        pred_norm = m.predict(x, num_iteration=m.best_iteration)
        pred_kwh = np.maximum(np.expm1(pred_norm), 0.0) * anchor
        raw_preds[:, i] = np.maximum(pred_kwh, 0.0).astype("float32")

    # Quantile crossing audit: una riga è "crossing" se i quantili predetti
    # non sono in ordine crescente lungo gli alpha (prima di sort).
    crossing_mask = (np.diff(raw_preds, axis=1) < 0).any(axis=1)

    # Forza la monotonia ordinando i quantili per ogni riga.
    sorted_preds = np.sort(raw_preds, axis=1)

    # Threshold: righe predette zero → tutti i quantili a 0.
    is_positive = p_positive >= threshold
    sorted_preds[~is_positive, :] = 0.0

    n_pos = int(is_positive.sum())
    n_cross_pos = int(crossing_mask[is_positive].sum()) if n_pos > 0 else 0
    crossing_info = {
        "n_rows": int(len(df)),
        "n_positive_rows": n_pos,
        "n_crossings_in_positive": n_cross_pos,
        "crossing_rate_in_positive_pct": (100.0 * n_cross_pos / n_pos) if n_pos > 0 else 0.0,
    }

    out = df[[POD_COL, SEGMENT_COL, TIME_COL, "t_target", "y_target"]].copy()
    out["p_positive"] = p_positive.astype("float32")
    out["positive_scale_anchor"] = anchor.astype("float32")
    out["threshold_positive"] = float(threshold)
    for i, alpha in enumerate(quantiles_sorted):
        out[f"y_pred_{quantile_slot(alpha)}"] = sorted_preds[:, i]

    if len(quantiles_sorted) >= 2:
        q_low = sorted_preds[:, 0]
        q_high = sorted_preds[:, -1]
        width = q_high - q_low
        in_band = (q_low <= out["y_target"].values) & (out["y_target"].values <= q_high)
        out["interval_width"] = width.astype("float32")
        out["in_interval_80"] = in_band.astype("int8")

    return out, crossing_info


def choose_threshold(
    cls_model: lgb.Booster,
    reg_model: lgb.Booster,
    val: pd.DataFrame,
    feature_cols: list[str],
    thresholds: list[float],
    max_positive_killed_pct: float,
) -> tuple[float, pd.DataFrame]:
    x = val[feature_cols]

    p_positive = cls_model.predict(x, num_iteration=cls_model.best_iteration)
    pred_log = reg_model.predict(x, num_iteration=reg_model.best_iteration)
    positive_scale_anchor = compute_positive_scale_anchor(x)
    pred_positive_norm = np.maximum(np.expm1(pred_log), 0.0)
    pred_positive = pred_positive_norm * positive_scale_anchor

    y = val["y_target"].values.astype(float)

    rows = []
    zero_mask = y == 0
    pos_mask = y > 0

    for th in thresholds:
        pred = np.where(p_positive >= th, pred_positive, 0.0)
        m = compute_metrics(y, pred)

        zero_correct = (
            float(np.mean(pred[zero_mask] == 0) * 100.0) if zero_mask.any() else np.nan
        )
        positive_killed = (
            float(np.mean(pred[pos_mask] == 0) * 100.0) if pos_mask.any() else np.nan
        )

        rows.append(
            {
                "threshold_positive": float(th),
                **m,
                "zero_correct_pct": zero_correct,
                "positive_killed_pct": positive_killed,
                "constraint_satisfied": bool(
                    np.isnan(positive_killed)
                    or positive_killed <= max_positive_killed_pct
                ),
            }
        )

    df_th = pd.DataFrame(rows)
    feasible = df_th[df_th["constraint_satisfied"]].copy()
    if feasible.empty:
        log.warning(
            "Nessuna soglia rispetta max_positive_killed_pct=%.2f. Fallback su WAPE/MAE minimi.",
            max_positive_killed_pct,
        )
        best_row = df_th.sort_values(["wape_pct", "mae"], na_position="last").iloc[0]
    else:
        best_row = feasible.sort_values(["wape_pct", "mae"], na_position="last").iloc[0]
    return float(best_row["threshold_positive"]), df_th


def build_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eval_set, hl), g in forecast.groupby(["eval_set", "horizon_label"], sort=False):
        for label, cond in [
            ("all", pd.Series(True, index=g.index)),
            ("zero", g["y_target"] == 0),
            ("positive", g["y_target"] > 0),
        ]:
            gg = g[cond]
            if gg.empty:
                continue
            m = compute_metrics(gg["y_target"].values, gg["y_pred"].values)
            rows.append(
                {
                    "eval_set": eval_set,
                    "horizon_label": hl,
                    "target_group": label,
                    "n": len(gg),
                    **m,
                    "avg_y_true": float(gg["y_target"].mean()),
                    "avg_y_pred": float(gg["y_pred"].mean()),
                    "pct_zero_true": float((gg["y_target"] == 0).mean() * 100.0),
                    "pct_pred_zero": float((gg["y_pred"] == 0).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


# =========================
# Quantile metrics
# =========================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    diff = y_true - y_pred
    loss = np.where(diff >= 0, alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def compute_quantile_group_metrics(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
    y_pred_point: np.ndarray,
    coverage_target: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    n = int(len(y_true))
    if n == 0:
        empty = {
            "n_observations": 0,
            "coverage_80_pct": np.nan,
            "coverage_gap_from_target_pp": np.nan,
            "avg_interval_width": np.nan,
            "median_interval_width": np.nan,
            "avg_relative_interval_width": np.nan,
            "median_relative_interval_width": np.nan,
            "avg_pinball_loss": np.nan,
            "mae_q50": np.nan,
            "rmse_q50": np.nan,
            "wape_q50_pct": np.nan,
            "wape_point_pct": np.nan,
        }
        for alpha in quantile_preds:
            empty[f"pinball_loss_{quantile_slot(alpha)}"] = np.nan
        return empty

    alphas_sorted = sorted(quantile_preds.keys())
    q_low = np.asarray(quantile_preds[alphas_sorted[0]], dtype=float)
    q_high = np.asarray(quantile_preds[alphas_sorted[-1]], dtype=float)

    in_band = (q_low <= y_true) & (y_true <= q_high)
    coverage_pct = float(np.mean(in_band) * 100.0)

    width = q_high - q_low
    avg_width = float(np.mean(width))
    median_width = float(np.median(width))

    nonzero = np.abs(y_true) > 0
    if nonzero.any():
        rel = width[nonzero] / np.abs(y_true[nonzero])
        avg_rel_width = float(np.mean(rel))
        median_rel_width = float(np.median(rel))
    else:
        avg_rel_width = np.nan
        median_rel_width = np.nan

    pinball_per_alpha = {}
    for alpha in alphas_sorted:
        pinball_per_alpha[alpha] = pinball_loss(y_true, quantile_preds[alpha], alpha)
    avg_pinball = float(np.mean(list(pinball_per_alpha.values())))

    # q50 come stima puntuale
    median_alpha = min(alphas_sorted, key=lambda a: abs(a - 0.5))
    q50 = np.asarray(quantile_preds[median_alpha], dtype=float)
    err_q50 = q50 - y_true
    mae_q50 = float(np.mean(np.abs(err_q50)))
    rmse_q50 = float(np.sqrt(np.mean(err_q50 ** 2)))
    denom = float(np.sum(np.abs(y_true)))
    wape_q50 = float(100.0 * np.sum(np.abs(err_q50)) / denom) if denom > 0 else np.nan

    # WAPE del punto preso dal regressore puntuale, per confronto diretto
    y_pred_point = np.asarray(y_pred_point, dtype=float)
    err_point = y_pred_point - y_true
    wape_point = float(100.0 * np.sum(np.abs(err_point)) / denom) if denom > 0 else np.nan

    out = {
        "n_observations": n,
        "coverage_80_pct": coverage_pct,
        "coverage_gap_from_target_pp": coverage_pct - coverage_target,
        "avg_interval_width": avg_width,
        "median_interval_width": median_width,
        "avg_relative_interval_width": avg_rel_width,
        "median_relative_interval_width": median_rel_width,
        "avg_pinball_loss": avg_pinball,
        "mae_q50": mae_q50,
        "rmse_q50": rmse_q50,
        "wape_q50_pct": wape_q50,
        "wape_point_pct": wape_point,
    }
    for alpha in alphas_sorted:
        out[f"pinball_loss_{quantile_slot(alpha)}"] = pinball_per_alpha[alpha]
    return out


def build_quantile_metrics(
    forecast: pd.DataFrame,
    quantiles_sorted: list[float],
    coverage_target: float,
) -> pd.DataFrame:
    slot_cols = [f"y_pred_{quantile_slot(a)}" for a in quantiles_sorted]
    rows = []
    for (eval_set, hl), g in forecast.groupby(["eval_set", "horizon_label"], sort=False):
        if g.empty:
            continue
        quantile_preds = {a: g[f"y_pred_{quantile_slot(a)}"].values for a in quantiles_sorted}
        m = compute_quantile_group_metrics(
            y_true=g["y_target"].values,
            quantile_preds=quantile_preds,
            y_pred_point=g["y_pred_point"].values,
            coverage_target=coverage_target,
        )
        rows.append(
            {
                "eval_set": eval_set,
                "horizon_label": hl,
                **m,
            }
        )
    return pd.DataFrame(rows)


# =========================
# Pipeline
# =========================

def run_pipeline(args: argparse.Namespace) -> int:
    t0 = time.time()
    conn = open_conn(args.db)

    lags = parse_int_list(args.lags)
    rollings = parse_int_list(args.rollings)
    horizons = parse_int_list(args.horizons)
    thresholds = parse_float_list(args.thresholds)
    quantiles = sorted(parse_float_list(args.quantiles))
    for a in quantiles:
        if not 0.0 < a < 1.0:
            raise ValueError(f"Quantile alpha non valido: {a} (deve essere in (0,1))")

    output_forecast_table = (
        args.output_forecast_table
        or f"{BASE_OUTPUT_FORECAST_TABLE}_{args.output_suffix}"
    )
    output_metrics_table = (
        args.output_metrics_table
        or f"{BASE_OUTPUT_METRICS_TABLE}_{args.output_suffix}"
    )
    output_threshold_table = (
        args.output_threshold_table
        or f"{BASE_OUTPUT_THRESHOLD_TABLE}_{args.output_suffix}"
    )
    quantile_forecast_table = (
        args.quantile_forecast_table
        or f"{BASE_QUANTILE_FORECAST_TABLE}_{args.quantile_output_suffix}"
    )
    quantile_metrics_table = (
        args.quantile_metrics_table
        or f"{BASE_QUANTILE_METRICS_TABLE}_{args.quantile_output_suffix}"
    )

    df = load_data(conn, args.source_table, args.max_pods, args.date_from)

    df["_PodId_raw"] = df[POD_COL].astype(str)
    df["_segment_raw"] = df[SEGMENT_COL].astype(str)

    df = add_calendar_features(df)
    df = add_history_features(df, lags, rollings)

    feature_cols, categorical_cols, history_cols = get_feature_cols(
        lags,
        rollings,
        use_pod_id=not args.no_pod_id_feature,
    )

    train_pods, cold_pods = split_pods(df, args.cold_test_frac, args.seed)
    log.info("POD train=%s | POD cold=%s", len(train_pods), len(cold_pods))

    all_forecasts: list[pd.DataFrame] = []
    all_quantile_forecasts: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    all_crossings: list[dict] = []

    for h in horizons:
        hl = horizon_label(h)
        log.info("=" * 80)
        log.info("Training two-stage + quantile orizzonte h=%s (%s)", h, hl)

        dh = add_direct_target(df, h)
        splits = make_splits(
            dh,
            train_pods=train_pods,
            cold_pods=cold_pods,
            val_days=args.val_days,
            temporal_test_days=args.temporal_test_days,
        )
        train = splits["train"]
        val = splits["val"]
        temporal = splits["temporal"]
        cold_with_history = splits["cold_with_history"]
        cold_no_history = splits["cold_no_history"]

        if train.empty or val.empty:
            log.warning("h=%s saltato: train o validation vuoti", h)
            continue

        schema = fit_categorical_schema(train, categorical_cols)
        apply_categorical_schema(
            [train, val, temporal, cold_with_history, cold_no_history],
            schema,
        )

        log.info(
            "h=%s | train=%s | val=%s | temporal=%s | cold=%s | features=%s | quantiles=%s",
            h,
            f"{len(train):,}",
            f"{len(val):,}",
            f"{len(temporal):,}",
            f"{len(cold_with_history):,}",
            len(feature_cols),
            quantiles,
        )

        cls_model = train_classifier(
            train, val, feature_cols, categorical_cols,
            seed=args.seed,
            num_boost_round=args.num_boost_round_cls,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        reg_model = train_regressor_positive(
            train, val, feature_cols, categorical_cols,
            seed=args.seed,
            num_boost_round=args.num_boost_round_reg,
            early_stopping_rounds=args.early_stopping_rounds,
            objective=args.reg_objective,
        )

        quantile_models = train_quantile_regressors(
            train, val, feature_cols, categorical_cols,
            quantiles=quantiles,
            seed=args.seed,
            num_boost_round=args.num_boost_round_quant,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        best_th, th_df = choose_threshold(
            cls_model, reg_model, val, feature_cols, thresholds,
            max_positive_killed_pct=args.max_positive_killed_pct,
        )
        th_df["horizon_steps"] = h
        th_df["horizon_label"] = hl
        th_df["selected_threshold"] = best_th
        th_df["max_positive_killed_pct_allowed"] = args.max_positive_killed_pct
        all_thresholds.append(th_df)
        log.info("h=%s | threshold selezionata su validation = %.3f", h, best_th)

        eval_sets = [
            ("temporal", temporal, None),
            ("cold_with_history", cold_with_history, None),
            ("cold_no_history", cold_no_history, history_cols),
        ]

        for eval_name, eval_df, mask_cols in eval_sets:
            if eval_df.empty:
                continue

            # --- Point forecast (esistente, invariato) ---
            fc = predict_two_stage(
                cls_model, reg_model, eval_df, feature_cols,
                threshold=best_th, mask_history_cols=mask_cols,
            )
            fc[POD_COL] = eval_df["_PodId_raw"].astype(str).to_numpy()
            if "_segment_raw" in eval_df.columns:
                fc[SEGMENT_COL] = eval_df["_segment_raw"].astype(str).to_numpy()
            else:
                fc[SEGMENT_COL] = eval_df[SEGMENT_COL].astype(str).to_numpy()
            fc["horizon_steps"] = h
            fc["horizon_label"] = hl
            fc["eval_set"] = eval_name
            fc["has_history"] = 0 if mask_cols else 1
            fc["model_type"] = "two_stage_zero_positive_normscale_v1"
            all_forecasts.append(fc)

            # --- Quantile forecast (nuovo) ---
            fq, cross_info = predict_two_stage_quantile(
                cls_model, quantile_models, eval_df, feature_cols,
                threshold=best_th,
                quantiles_sorted=quantiles,
                mask_history_cols=mask_cols,
            )
            fq[POD_COL] = eval_df["_PodId_raw"].astype(str).to_numpy()
            if "_segment_raw" in eval_df.columns:
                fq[SEGMENT_COL] = eval_df["_segment_raw"].astype(str).to_numpy()
            else:
                fq[SEGMENT_COL] = eval_df[SEGMENT_COL].astype(str).to_numpy()
            fq["horizon_steps"] = h
            fq["horizon_label"] = hl
            fq["eval_set"] = eval_name
            fq["has_history"] = 0 if mask_cols else 1
            # y_pred_point dal regressore puntuale (per confronto).
            fq["y_pred_point"] = fc["y_pred"].values.astype("float32")
            fq["model_type"] = "two_stage_zero_positive_normscale_quantile_v1"

            cross_info.update({
                "horizon_steps": h,
                "horizon_label": hl,
                "eval_set": eval_name,
            })
            all_crossings.append(cross_info)
            if cross_info["crossing_rate_in_positive_pct"] > CROSSING_WARN_PCT:
                log.warning(
                    "h=%s eval=%s: crossing rate %.2f%% (> %.1f%%): regularization da rivedere",
                    h, eval_name, cross_info["crossing_rate_in_positive_pct"], CROSSING_WARN_PCT,
                )
            else:
                log.info(
                    "h=%s eval=%s: crossing rate %.2f%% su righe positive (n_pos=%s)",
                    h, eval_name, cross_info["crossing_rate_in_positive_pct"], cross_info["n_positive_rows"],
                )

            all_quantile_forecasts.append(fq)

    if not all_forecasts:
        raise RuntimeError("Nessun forecast prodotto. Controlla split, date e parametri.")

    forecast = pd.concat(all_forecasts, ignore_index=True)
    threshold_df = pd.concat(all_thresholds, ignore_index=True)
    metrics = build_metrics(forecast)

    quantile_forecast = pd.concat(all_quantile_forecasts, ignore_index=True)
    quantile_metrics = build_quantile_metrics(
        quantile_forecast,
        quantiles_sorted=quantiles,
        coverage_target=args.coverage_target,
    )
    crossings_df = pd.DataFrame(all_crossings)

    forecast_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{output_forecast_table}_eval_h "
        f"ON {q(output_forecast_table)} (eval_set, horizon_label)",
        f"CREATE INDEX IF NOT EXISTS idx_{output_forecast_table}_pod_time "
        f"ON {q(output_forecast_table)} ({q(POD_COL)}, {q(TIME_COL)})",
    ]
    metrics_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{output_metrics_table}_eval_h_g "
        f"ON {q(output_metrics_table)} (eval_set, horizon_label, target_group)",
    ]
    quantile_forecast_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{quantile_forecast_table}_eval_h "
        f"ON {q(quantile_forecast_table)} (eval_set, horizon_label)",
        f"CREATE INDEX IF NOT EXISTS idx_{quantile_forecast_table}_pod_time "
        f"ON {q(quantile_forecast_table)} ({q(POD_COL)}, {q(TIME_COL)})",
    ]
    quantile_metrics_indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{quantile_metrics_table}_eval_h "
        f"ON {q(quantile_metrics_table)} (eval_set, horizon_label)",
    ]

    write_df(conn, forecast, output_forecast_table,
             chunksize=args.write_chunksize, create_indexes_sql=forecast_indexes)
    write_df(conn, metrics, output_metrics_table,
             chunksize=args.write_chunksize, create_indexes_sql=metrics_indexes)
    write_df(conn, threshold_df, output_threshold_table,
             chunksize=args.write_chunksize)
    write_df(conn, quantile_forecast, quantile_forecast_table,
             chunksize=args.write_chunksize,
             create_indexes_sql=quantile_forecast_indexes)
    write_df(conn, quantile_metrics, quantile_metrics_table,
             chunksize=args.write_chunksize,
             create_indexes_sql=quantile_metrics_indexes)

    log.info("Output scritto su SQLite:")
    log.info("- %s (point forecast)", output_forecast_table)
    log.info("- %s (point metrics)", output_metrics_table)
    log.info("- %s (thresholds)", output_threshold_table)
    log.info("- %s (quantile forecast)", quantile_forecast_table)
    log.info("- %s (quantile metrics)", quantile_metrics_table)
    log.info("Runtime %.1f sec", time.time() - t0)

    print("\n" + "=" * 100)
    print("RIEPILOGO TWO-STAGE NORMSCALE QUANTILE V1 — point + intervals (q10/q50/q90)")
    print("=" * 100)
    print("Point metrics (target_group=all):")
    point_all = metrics[metrics["target_group"] == "all"][[
        "eval_set", "horizon_label", "n", "mae", "rmse", "wape_pct", "smape_pct", "bias",
    ]]
    print(point_all.to_string(index=False))
    print("\nQuantile metrics:")
    qcols = [
        "eval_set", "horizon_label", "n_observations",
        "coverage_80_pct", "coverage_gap_from_target_pp",
        "avg_interval_width", "median_relative_interval_width",
        "avg_pinball_loss", "wape_q50_pct", "wape_point_pct",
    ]
    qcols = [c for c in qcols if c in quantile_metrics.columns]
    print(quantile_metrics[qcols].to_string(index=False))
    print("\nQuantile crossing audit (post-sort è sempre monotono):")
    print(crossings_df[[
        "eval_set", "horizon_label", "n_rows", "n_positive_rows",
        "n_crossings_in_positive", "crossing_rate_in_positive_pct",
    ]].to_string(index=False))
    print("=" * 100)

    conn.close()
    return 0


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LightGBM POD two-stage normscale + quantile forecasting V1: classificatore "
            "zero/positivo + regressore puntuale + tre regressori quantile (q10/q50/q90) "
            "per intervalli di confidenza all'80%."
        )
    )

    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--source-table", type=str, default=SOURCE_TABLE)
    p.add_argument("--max-pods", type=int, default=None)
    p.add_argument("--date-from", type=str, default=None)

    p.add_argument("--lags", type=str, default=",".join(map(str, DEFAULT_LAGS)))
    p.add_argument("--rollings", type=str, default=",".join(map(str, DEFAULT_ROLLINGS)))
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))

    p.add_argument("--cold-test-frac", type=float, default=0.15)
    p.add_argument("--val-days", type=int, default=14)
    p.add_argument("--temporal-test-days", type=int, default=30)

    p.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    p.add_argument("--max-positive-killed-pct", type=float, default=10.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-boost-round-cls", type=int, default=800)
    p.add_argument("--num-boost-round-reg", type=int, default=800)
    p.add_argument("--num-boost-round-quant", type=int, default=800,
                   help="Numero massimo di iterazioni per ciascun regressore quantile.")
    p.add_argument("--early-stopping-rounds", type=int, default=50)

    p.add_argument(
        "--reg-objective",
        type=str,
        default="regression_l1",
        choices=["regression_l1", "regression", "huber", "fair"],
    )

    p.add_argument(
        "--no-pod-id-feature",
        action="store_true",
        help="Esclude PodId dalle feature categoriche.",
    )

    p.add_argument(
        "--quantiles",
        type=str,
        default=",".join(f"{q_:.2f}" for q_ in QUANTILES_DEFAULT),
        help="Lista quantili da addestrare (alpha in (0,1)).",
    )
    p.add_argument(
        "--coverage-target",
        type=float,
        default=COVERAGE_TARGET_DEFAULT,
        help="Coverage nominale dell'intervallo (per coverage_gap_from_target_pp).",
    )

    p.add_argument("--output-suffix", type=str, default="v2")
    p.add_argument("--output-forecast-table", type=str, default=None)
    p.add_argument("--output-metrics-table", type=str, default=None)
    p.add_argument("--output-threshold-table", type=str, default=None)
    p.add_argument("--quantile-output-suffix", type=str, default="v1")
    p.add_argument("--quantile-forecast-table", type=str, default=None)
    p.add_argument("--quantile-metrics-table", type=str, default=None)
    p.add_argument("--write-chunksize", type=int, default=100_000)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
