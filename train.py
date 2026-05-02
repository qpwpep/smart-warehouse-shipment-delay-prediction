from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor
import hydra
from hydra.core.hydra_config import HydraConfig
import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from hydra.utils import to_absolute_path
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm


warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
MODEL_SEED = 42
DEFAULT_MODEL_SEEDS = [42, 2026, 777]
MODEL_SEEDS = DEFAULT_MODEL_SEEDS.copy()
N_SPLITS = 5
CV_SEED = 42

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
LAYOUT_PATH = DATA_DIR / "layout_info.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

RUN_DATE = datetime.now().astimezone().date().isoformat()
RUN_TIMESTAMP = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")
RUN_ID = RUN_TIMESTAMP
OUTPUT_DIR = Path("outputs") / RUN_ID
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
OOF_PATH = OUTPUT_DIR / "oof_train.csv"
REPORT_PATH = OUTPUT_DIR / "cv_report_train.json"
LOG_PATH = OUTPUT_DIR / "train.log"
HYDRA_RUN_DIR: Path | None = None
HYDRA_LOG_PATH: Path | None = None

LAG_ROLLING_COLS = [
    "order_inflow_15m",
    "congestion_score",
    "battery_mean",
    "low_battery_ratio",
    "robot_idle",
    "robot_charging",
    "charge_queue_length",
    "max_zone_density",
]

MODEL_FAMILIES = ["lightgbm", "xgboost", "catboost"]
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {},
    "xgboost": {},
    "catboost": {},
}
MODEL_FAMILY_SEEDS: dict[str, list[int]] = {}

FEATURE_DROP_COLS: list[str] = []
FEATURE_DROP_GROUPS: list[str] = []
TEMPORAL_SOURCE_COLS = ["day_of_week", "shift_hour"]
TEMPORAL_CATEGORICAL_FEATURES = ["day_of_week_cat", "shift_hour_cat"]
TEMPORAL_PERIODS = {
    "day_of_week": 7,
    "shift_hour": 24,
}

ANALYSIS_ENABLED = False
ANALYSIS_ABLATION_ENABLED = True
ANALYSIS_PERMUTATION_ENABLED = True
ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD: int | None = 5000
ANALYSIS_PERMUTATION_N_REPEATS = 2
ANALYSIS_RECOMMENDED_DROP_DELTA_THRESHOLD = 0.0
ANALYSIS_RECOMMENDED_DROP_POSITIVE_FOLD_RATE = 0.6

FEATURE_GROUP_NAMES = [
    "layout",
    "robot_battery",
    "congestion",
    "worker",
    "environment",
    "network",
    "order_flow",
    "shipping_kpi",
    "temporal",
    "missingness",
    "misc",
]

FEATURE_GROUP_COLUMNS = {
    "layout": {
        "layout_type",
        "aisle_width_avg",
        "intersection_count",
        "one_way_ratio",
        "pack_station_count",
        "charger_count",
        "layout_compactness",
        "zone_dispersion",
        "robot_total",
        "building_age_years",
        "floor_area_sqm",
        "ceiling_height_m",
        "fire_sprinkler_count",
        "emergency_exit_count",
        "storage_density_pct",
        "vertical_utilization",
        "racking_height_avg_m",
        "charger_per_robot",
        "pack_station_per_order",
        "area_per_robot",
        "intersection_per_area",
        "storage_pressure",
    },
    "robot_battery": {
        "robot_active",
        "robot_idle",
        "robot_charging",
        "robot_utilization",
        "avg_trip_distance",
        "task_reassign_15m",
        "battery_mean",
        "battery_std",
        "low_battery_ratio",
        "charge_queue_length",
        "avg_charge_wait",
        "fleet_age_months_avg",
        "maintenance_schedule_score",
        "robot_firmware_update_days",
        "avg_idle_duration_min",
        "charge_efficiency_pct",
        "battery_cycle_count_avg",
        "agv_task_success_rate",
        "robot_calibration_score",
        "charge_pressure",
        "robot_total_gap",
        "active_robot_share",
        "available_robot_share",
        "charging_robot_share",
        "idle_robot_share",
        "low_battery_robot_count",
        "battery_depletion_pressure",
        "charge_capacity_gap",
    },
    "congestion": {
        "congestion_score",
        "max_zone_density",
        "blocked_path_15m",
        "near_collision_15m",
        "fault_count_15m",
        "avg_recovery_time",
        "replenishment_overlap",
        "aisle_traffic_score",
        "path_optimization_score",
        "intersection_wait_time_avg",
        "congestion_pressure",
        "congestion_density_interaction",
        "incident_count_15m",
        "incident_per_active_robot",
    },
    "worker": {
        "worker_avg_tenure_months",
        "safety_score_monthly",
        "manual_override_ratio",
        "staff_on_floor",
        "forklift_active_count",
        "orders_per_staff",
        "sku_per_staff",
        "task_pressure",
    },
    "environment": {
        "warehouse_temp_avg",
        "humidity_pct",
        "external_temp_c",
        "wind_speed_kmh",
        "precipitation_mm",
        "lighting_level_lux",
        "ambient_noise_db",
        "floor_vibration_idx",
        "air_quality_idx",
        "co2_level_ppm",
        "hvac_power_kw",
        "ups_battery_pct",
        "lighting_zone_variance",
        "cold_storage_temp_c",
        "zone_temp_variance",
    },
    "network": {
        "wms_response_time_ms",
        "scanner_error_rate",
        "wifi_signal_db",
        "network_latency_ms",
        "label_print_queue",
        "barcode_read_success_rate",
        "network_error_pressure",
    },
    "order_flow": {
        "order_inflow_15m",
        "unique_sku_15m",
        "avg_items_per_order",
        "urgent_order_ratio",
        "heavy_item_ratio",
        "cold_chain_ratio",
        "sku_concentration",
        "return_order_ratio",
        "conveyor_speed_mps",
        "prev_shift_volume",
        "avg_package_weight_kg",
        "inventory_turnover_rate",
        "daily_forecast_accuracy",
        "order_wave_count",
        "pick_list_length_avg",
        "pack_utilization",
        "bulk_order_ratio",
        "pallet_wrap_time_min",
        "inflow_per_active_robot",
        "sku_per_order",
        "inflow_per_robot_total",
        "inflow_per_available_robot",
        "estimated_item_inflow",
        "items_per_active_robot",
        "urgent_order_count_est",
        "heavy_order_count_est",
        "cold_chain_order_count_est",
        "packing_load_per_station",
    },
    "shipping_kpi": {
        "kpi_otd_pct",
        "backorder_ratio",
        "shift_handover_delay_min",
        "sort_accuracy_pct",
        "outbound_truck_wait_min",
        "dock_to_stock_hours",
        "quality_check_rate",
        "loading_dock_util",
        "express_lane_util",
        "staging_area_util",
        "cross_dock_ratio",
        "packaging_material_cost",
        "dock_pressure",
        "truck_wait_load",
        "otd_gap",
        "sort_error_pct",
        "backorder_pressure",
    },
    "temporal": {
        "scenario_step",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_week_cat",
        "shift_hour_sin",
        "shift_hour_cos",
        "shift_hour_cat",
        "scenario_step_ratio",
        "scenario_step_is_first",
        "scenario_step_is_last",
    },
    "missingness": {
        "row_missing_count",
        "row_missing_ratio",
        "layout_missing_count",
        "robot_battery_missing_count",
        "congestion_missing_count",
        "worker_missing_count",
        "environment_missing_count",
        "network_missing_count",
        "order_flow_missing_count",
        "shipping_kpi_missing_count",
    },
}

LAG_DERIVED_SUFFIXES = (
    "_lag1",
    "_lag2",
    "_delta1",
    "_delta2",
    "_rolling3_mean",
    "_rolling5_mean",
    "_rolling10_mean",
    "_rolling5_std",
)

LOGGER = logging.getLogger("train")

OBJECTIVES = [
    {
        "name": "mae",
        "params": {
            "lightgbm": {"objective": "mae", "metric": "mae"},
            "xgboost": {"objective": "reg:absoluteerror"},
            "catboost": {"loss_function": "MAE"},
        },
    },
]


def apply_hydra_config(cfg: DictConfig) -> dict[str, Any]:
    global TARGET
    global ID_COLS
    global MODEL_SEED
    global MODEL_SEEDS
    global N_SPLITS
    global CV_SEED
    global DATA_DIR
    global TRAIN_PATH
    global TEST_PATH
    global LAYOUT_PATH
    global SAMPLE_SUBMISSION_PATH
    global RUN_DATE
    global RUN_TIMESTAMP
    global RUN_ID
    global OUTPUT_DIR
    global SUBMISSION_PATH
    global OOF_PATH
    global REPORT_PATH
    global LOG_PATH
    global LAG_ROLLING_COLS
    global MODEL_FAMILIES
    global MODEL_PARAMS
    global MODEL_FAMILY_SEEDS
    global OBJECTIVES
    global FEATURE_DROP_COLS
    global FEATURE_DROP_GROUPS
    global ANALYSIS_ENABLED
    global ANALYSIS_ABLATION_ENABLED
    global ANALYSIS_PERMUTATION_ENABLED
    global ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD
    global ANALYSIS_PERMUTATION_N_REPEATS
    global ANALYSIS_RECOMMENDED_DROP_DELTA_THRESHOLD
    global ANALYSIS_RECOMMENDED_DROP_POSITIVE_FOLD_RATE

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    TARGET = str(cfg.target)
    ID_COLS = [str(col) for col in cfg.id_cols]
    model_seeds_cfg = OmegaConf.select(cfg, "model_seeds")
    configured_model_seed = int(cfg.model_seed)
    if model_seeds_cfg is None:
        MODEL_SEEDS = [configured_model_seed]
    else:
        MODEL_SEEDS = [int(seed) for seed in model_seeds_cfg]
        if configured_model_seed != 42 and MODEL_SEEDS == DEFAULT_MODEL_SEEDS:
            MODEL_SEEDS = [configured_model_seed]
    if not MODEL_SEEDS:
        raise ValueError("model_seeds must contain at least one seed.")
    MODEL_SEED = int(MODEL_SEEDS[0])
    N_SPLITS = int(cfg.cv.n_splits)
    CV_SEED = int(cfg.cv.seed)
    RUN_DATE = str(OmegaConf.select(cfg, "run.date"))
    RUN_TIMESTAMP = str(OmegaConf.select(cfg, "run.timestamp"))
    RUN_ID = str(OmegaConf.select(cfg, "run.id"))

    DATA_DIR = Path(to_absolute_path(str(cfg.data.dir)))
    TRAIN_PATH = DATA_DIR / str(cfg.data.train)
    TEST_PATH = DATA_DIR / str(cfg.data.test)
    LAYOUT_PATH = DATA_DIR / str(cfg.data.layout)
    SAMPLE_SUBMISSION_PATH = DATA_DIR / str(cfg.data.sample_submission)

    OUTPUT_DIR = Path(to_absolute_path(str(cfg.output.dir)))
    SUBMISSION_PATH = OUTPUT_DIR / str(cfg.output.submission)
    OOF_PATH = OUTPUT_DIR / str(cfg.output.oof)
    REPORT_PATH = OUTPUT_DIR / str(cfg.output.report)
    LOG_PATH = OUTPUT_DIR / str(OmegaConf.select(cfg, "output.log", default="train.log"))

    FEATURE_DROP_COLS = [str(col) for col in OmegaConf.select(cfg, "features.drop_cols", default=[])]
    FEATURE_DROP_GROUPS = [
        str(group) for group in OmegaConf.select(cfg, "features.drop_groups", default=[])
    ]
    LAG_ROLLING_COLS = [str(col) for col in cfg.features.lag_rolling_cols]
    MODEL_FAMILIES = [str(model_family) for model_family in cfg.models.families]
    MODEL_PARAMS = OmegaConf.to_container(
        OmegaConf.select(cfg, "models.params", default={}),
        resolve=True,
    )
    raw_family_seeds = OmegaConf.to_container(
        OmegaConf.select(cfg, "models.family_seeds", default={}),
        resolve=True,
    )
    raw_family_seeds = {} if raw_family_seeds is None else dict(raw_family_seeds)
    raw_family_seeds = {
        model_family: seeds
        for model_family, seeds in raw_family_seeds.items()
        if model_family in MODEL_FAMILIES
    }
    MODEL_FAMILY_SEEDS = {}
    for model_family in MODEL_FAMILIES:
        seeds = raw_family_seeds.get(model_family)
        if seeds is not None:
            family_seeds = [int(seed) for seed in seeds]
            if not family_seeds:
                raise ValueError(f"models.family_seeds.{model_family} is empty.")
            MODEL_FAMILY_SEEDS[model_family] = family_seeds
    if MODEL_FAMILY_SEEDS:
        for model_family in MODEL_FAMILIES:
            MODEL_FAMILY_SEEDS.setdefault(model_family, MODEL_SEEDS.copy())
        unique_model_seeds: list[int] = []
        for model_family in MODEL_FAMILIES:
            for seed in MODEL_FAMILY_SEEDS[model_family]:
                if seed not in unique_model_seeds:
                    unique_model_seeds.append(seed)
        MODEL_SEEDS = unique_model_seeds
        MODEL_SEED = int(MODEL_SEEDS[0])
    OBJECTIVES = list(OmegaConf.to_container(cfg.objectives, resolve=True))
    ANALYSIS_ENABLED = bool(OmegaConf.select(cfg, "analysis.enabled", default=False))
    ANALYSIS_ABLATION_ENABLED = bool(
        OmegaConf.select(cfg, "analysis.ablation.enabled", default=True)
    )
    ANALYSIS_PERMUTATION_ENABLED = bool(
        OmegaConf.select(cfg, "analysis.permutation_importance.enabled", default=True)
    )
    max_rows = OmegaConf.select(
        cfg,
        "analysis.permutation_importance.max_rows_per_fold",
        default=5000,
    )
    ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD = None if max_rows is None else int(max_rows)
    ANALYSIS_PERMUTATION_N_REPEATS = int(
        OmegaConf.select(cfg, "analysis.permutation_importance.n_repeats", default=2)
    )
    ANALYSIS_RECOMMENDED_DROP_DELTA_THRESHOLD = float(
        OmegaConf.select(
            cfg,
            "analysis.permutation_importance.recommended_drop_delta_threshold",
            default=0.0,
        )
    )
    ANALYSIS_RECOMMENDED_DROP_POSITIVE_FOLD_RATE = float(
        OmegaConf.select(
            cfg,
            "analysis.permutation_importance.recommended_drop_positive_fold_rate",
            default=0.6,
        )
    )

    return resolved_cfg


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def get_hydra_run_dir() -> Path | None:
    try:
        return Path(str(HydraConfig.get().runtime.output_dir)).resolve()
    except Exception:
        return None


def setup_logging() -> None:
    global HYDRA_RUN_DIR
    global HYDRA_LOG_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HYDRA_RUN_DIR = get_hydra_run_dir()
    HYDRA_LOG_PATH = HYDRA_RUN_DIR / "train.log" if HYDRA_RUN_DIR is not None else None

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = TqdmLoggingHandler()
    stdout_handler.setFormatter(formatter)
    LOGGER.addHandler(stdout_handler)

    file_paths = [LOG_PATH]
    if HYDRA_LOG_PATH is not None and HYDRA_LOG_PATH.resolve() != LOG_PATH.resolve():
        file_paths.append(HYDRA_LOG_PATH)

    for path in file_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)

    log(f"[logging] output_log={LOG_PATH.resolve()}")
    if HYDRA_LOG_PATH is not None:
        log(f"[logging] hydra_log={HYDRA_LOG_PATH.resolve()}")


def log(message: str) -> None:
    LOGGER.info(message)


def path_str(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def to_float(value: Any) -> float:
    return float(np.asarray(value).item())


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Target column is missing from train: {TARGET}")
    if TARGET in test.columns:
        raise ValueError(f"Target column unexpectedly exists in test: {TARGET}")
    if not sample_submission["ID"].equals(test["ID"]):
        raise ValueError("sample_submission ID order does not match test ID order.")

    return train, test, layout, sample_submission


def add_layout_features(
    df: pd.DataFrame,
    layout: pd.DataFrame,
    layout_type_dtype: CategoricalDtype,
) -> pd.DataFrame:
    merged = df.merge(layout, on="layout_id", how="left", validate="many_to_one")
    layout_cols = [c for c in layout.columns if c != "layout_id"]
    missing_layout_rows = int(merged[layout_cols].isna().all(axis=1).sum())
    if missing_layout_rows:
        raise ValueError(f"layout_info merge failed for {missing_layout_rows} rows.")

    merged["layout_type"] = merged["layout_type"].astype(layout_type_dtype)
    if merged["layout_type"].isna().any():
        raise ValueError("layout_type contains missing values after category alignment.")
    return merged


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    source_groups = [
        group
        for group in FEATURE_GROUP_NAMES
        if group not in {"temporal", "missingness", "misc"}
    ]
    row_cols: list[str] = []

    for group in source_groups:
        cols = sorted(col for col in FEATURE_GROUP_COLUMNS[group] if col in df.columns)
        if not cols:
            df[f"{group}_missing_count"] = np.int16(0)
            continue
        df[f"{group}_missing_count"] = df[cols].isna().sum(axis=1).astype(np.int16)
        row_cols.extend(cols)

    row_cols = list(dict.fromkeys(row_cols))
    if row_cols:
        row_missing_count = df[row_cols].isna().sum(axis=1)
        df["row_missing_count"] = row_missing_count.astype(np.int16)
        df["row_missing_ratio"] = row_missing_count / len(row_cols)
    else:
        df["row_missing_count"] = np.int16(0)
        df["row_missing_ratio"] = 0.0

    return df


def add_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("scenario_id", sort=False)
    derived: dict[str, Any] = {}
    scenario_step = grouped.cumcount().astype(np.int16)
    derived["scenario_step"] = scenario_step
    scenario_size = grouped["ID"].transform("size")
    derived["scenario_step_ratio"] = safe_divide(
        scenario_step.astype(np.float64),
        (scenario_size - 1).astype(np.float64),
    ).fillna(0.0)
    derived["scenario_step_is_first"] = (scenario_step == 0).astype(np.int8)
    derived["scenario_step_is_last"] = (scenario_step == scenario_size - 1).astype(
        np.int8
    )

    for col in LAG_ROLLING_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing lag/rolling source column: {col}")
        lag_col = f"{col}_lag1"
        lag1 = grouped[col].shift(1)
        lag2 = grouped[col].shift(2)
        derived[lag_col] = lag1
        derived[f"{col}_lag2"] = lag2
        derived[f"{col}_delta1"] = df[col] - lag1
        derived[f"{col}_delta2"] = df[col] - lag2

        lagged_grouped = lag1.groupby(df["scenario_id"], sort=False)
        for window in (3, 5, 10):
            derived[f"{col}_rolling{window}_mean"] = lagged_grouped.transform(
                lambda s, window=window: s.rolling(window=window, min_periods=1).mean()
            )
        derived[f"{col}_rolling5_std"] = lagged_grouped.transform(
            lambda s: s.rolling(window=5, min_periods=2).std()
        )

    return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1).copy()


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    robot_total = df["robot_total"]
    order_inflow = df["order_inflow_15m"]
    available_robots = df["robot_active"] + df["robot_idle"]
    staff_capacity = df["staff_on_floor"] + df["forklift_active_count"]
    estimated_item_inflow = order_inflow * df["avg_items_per_order"]
    low_battery_robot_count = df["low_battery_ratio"] * robot_total
    incident_count = (
        df["blocked_path_15m"] + df["near_collision_15m"] + df["fault_count_15m"]
    )
    derived: dict[str, Any] = {}

    derived["inflow_per_active_robot"] = safe_divide(
        order_inflow,
        df["robot_active"] + 1,
    )
    derived["inflow_per_robot_total"] = safe_divide(order_inflow, robot_total)
    derived["inflow_per_available_robot"] = safe_divide(
        order_inflow,
        available_robots + 1,
    )
    derived["sku_per_order"] = safe_divide(df["unique_sku_15m"], order_inflow + 1)
    derived["orders_per_staff"] = safe_divide(order_inflow, staff_capacity + 1)
    derived["sku_per_staff"] = safe_divide(df["unique_sku_15m"], staff_capacity + 1)
    derived["estimated_item_inflow"] = estimated_item_inflow
    derived["items_per_active_robot"] = safe_divide(
        estimated_item_inflow,
        df["robot_active"] + 1,
    )
    derived["urgent_order_count_est"] = order_inflow * df["urgent_order_ratio"]
    derived["heavy_order_count_est"] = order_inflow * df["heavy_item_ratio"]
    derived["cold_chain_order_count_est"] = order_inflow * df["cold_chain_ratio"]
    derived["charge_pressure"] = (
        df["robot_charging"]
        + df["charge_queue_length"]
        + low_battery_robot_count
    )
    derived["low_battery_robot_count"] = low_battery_robot_count
    derived["battery_depletion_pressure"] = (
        low_battery_robot_count + df["charge_queue_length"] + df["avg_charge_wait"]
    )
    derived["charge_capacity_gap"] = (
        df["charger_count"] - df["robot_charging"] - df["charge_queue_length"]
    )
    derived["congestion_pressure"] = (
        df["congestion_score"]
        + 20 * df["max_zone_density"]
        + 2 * df["blocked_path_15m"]
    )
    derived["congestion_density_interaction"] = (
        df["congestion_score"] * df["max_zone_density"]
    )
    derived["incident_count_15m"] = incident_count
    derived["incident_per_active_robot"] = safe_divide(
        incident_count,
        df["robot_active"] + 1,
    )
    derived["charger_per_robot"] = safe_divide(df["charger_count"], robot_total)
    derived["pack_station_per_order"] = safe_divide(
        df["pack_station_count"],
        order_inflow + 1,
    )
    derived["area_per_robot"] = safe_divide(df["floor_area_sqm"], robot_total)
    derived["intersection_per_area"] = safe_divide(
        df["intersection_count"],
        df["floor_area_sqm"],
    )
    derived["robot_total_gap"] = robot_total - (
        df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    )
    derived["active_robot_share"] = safe_divide(df["robot_active"], robot_total)
    derived["available_robot_share"] = safe_divide(available_robots, robot_total)
    derived["charging_robot_share"] = safe_divide(df["robot_charging"], robot_total)
    derived["idle_robot_share"] = safe_divide(df["robot_idle"], robot_total)
    derived["packing_load_per_station"] = safe_divide(
        estimated_item_inflow * (1 + df["pack_utilization"]),
        df["pack_station_count"] + 1,
    )
    derived["task_pressure"] = (
        df["task_reassign_15m"] + df["manual_override_ratio"] * df["robot_active"]
    )
    derived["dock_pressure"] = (
        df["loading_dock_util"]
        + df["express_lane_util"]
        + df["staging_area_util"]
        + safe_divide(df["outbound_truck_wait_min"], pd.Series(60.0, index=df.index))
    )
    derived["truck_wait_load"] = df["outbound_truck_wait_min"] * df["loading_dock_util"]
    derived["otd_gap"] = 100.0 - df["kpi_otd_pct"]
    derived["sort_error_pct"] = 100.0 - df["sort_accuracy_pct"]
    derived["backorder_pressure"] = df["backorder_ratio"] * order_inflow
    derived["network_error_pressure"] = (
        df["wms_response_time_ms"] * df["scanner_error_rate"]
        + df["network_latency_ms"] * (1 - df["barcode_read_success_rate"])
    )
    derived["storage_pressure"] = df["storage_density_pct"] * df["vertical_utilization"]
    return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1).copy()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, period in TEMPORAL_PERIODS.items():
        if col not in df.columns:
            raise ValueError(f"Missing temporal source column: {col}")

        numeric = pd.to_numeric(df[col], errors="coerce")
        rounded = numeric.round()
        invalid = numeric.notna() & (
            (rounded < 0) | (rounded >= period) | ~np.isclose(numeric, rounded)
        )
        if invalid.any():
            bad_values = sorted(numeric[invalid].dropna().unique().tolist())[:10]
            raise ValueError(f"{col} contains invalid cyclic values: {bad_values}")

        radians = 2 * np.pi * numeric / period
        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)

        categories = [str(i) for i in range(period)] + ["missing"]
        labels = rounded.astype("Int64").astype("string").fillna("missing")
        df[f"{col}_cat"] = labels.astype(
            CategoricalDtype(categories=categories, ordered=False)
        )

    return df


def strip_lag_suffix(col: str) -> str:
    for suffix in LAG_DERIVED_SUFFIXES:
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def infer_feature_group(col: str) -> str:
    base_col = strip_lag_suffix(col)
    for group, columns in FEATURE_GROUP_COLUMNS.items():
        if base_col in columns:
            return group
    return "misc"


def build_feature_group_map(feature_cols: list[str]) -> dict[str, str]:
    return {col: infer_feature_group(col) for col in feature_cols}


def count_feature_groups(feature_group_map: dict[str, str]) -> dict[str, int]:
    return {
        group: int(sum(1 for value in feature_group_map.values() if value == group))
        for group in FEATURE_GROUP_NAMES
    }


def apply_feature_pruning(
    feature_cols: list[str],
    feature_group_map: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    unknown_groups = sorted(set(FEATURE_DROP_GROUPS) - set(FEATURE_GROUP_NAMES))
    if unknown_groups:
        raise ValueError(f"Unknown feature drop groups: {unknown_groups}")

    unknown_cols = sorted(set(FEATURE_DROP_COLS) - set(feature_cols))
    if unknown_cols:
        raise ValueError(f"Unknown feature drop columns: {unknown_cols}")

    dropped_by_group = {
        group: sorted([col for col, value in feature_group_map.items() if value == group])
        for group in FEATURE_DROP_GROUPS
    }
    dropped_cols = sorted(set(FEATURE_DROP_COLS).union(*dropped_by_group.values()))
    kept_cols = [col for col in feature_cols if col not in dropped_cols]
    kept_group_map = {col: feature_group_map[col] for col in kept_cols}

    report = {
        "drop_cols": FEATURE_DROP_COLS,
        "drop_groups": FEATURE_DROP_GROUPS,
        "dropped_cols": dropped_cols,
        "dropped_by_group": dropped_by_group,
        "dropped_count": int(len(dropped_cols)),
        "group_counts": count_feature_groups(kept_group_map),
    }
    return kept_cols, report


def build_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    layout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, str], dict[str, Any]]:
    layout_type_dtype = CategoricalDtype(
        categories=sorted(layout["layout_type"].dropna().unique()),
        ordered=False,
    )

    train = add_layout_features(train_raw, layout, layout_type_dtype)
    test = add_layout_features(test_raw, layout, layout_type_dtype)

    train = add_missingness_features(train)
    test = add_missingness_features(test)

    train = add_scenario_features(train)
    test = add_scenario_features(test)

    train = add_pressure_features(train)
    test = add_pressure_features(test)

    train = add_temporal_features(train)
    test = add_temporal_features(test)

    excluded_train_cols = set(ID_COLS + [TARGET] + TEMPORAL_SOURCE_COLS)
    excluded_test_cols = set(ID_COLS + TEMPORAL_SOURCE_COLS)
    feature_cols = [c for c in train.columns if c not in excluded_train_cols]
    test_feature_cols = [c for c in test.columns if c not in excluded_test_cols]
    if feature_cols != test_feature_cols:
        missing_in_test = sorted(set(feature_cols) - set(test_feature_cols))
        extra_in_test = sorted(set(test_feature_cols) - set(feature_cols))
        raise ValueError(
            "Train/test feature mismatch: "
            f"missing_in_test={missing_in_test}, extra_in_test={extra_in_test}"
        )

    base_feature_group_map = build_feature_group_map(feature_cols)
    feature_cols, pruning_report = apply_feature_pruning(feature_cols, base_feature_group_map)
    feature_group_map = {col: base_feature_group_map[col] for col in feature_cols}

    object_features = train[feature_cols].select_dtypes(include="object").columns.tolist()
    if object_features:
        raise ValueError(f"Unexpected object feature columns: {object_features}")

    categorical_features = [
        col for col in ["layout_type"] + TEMPORAL_CATEGORICAL_FEATURES if col in feature_cols
    ]
    return train, test, feature_cols, categorical_features, feature_group_map, pruning_report


def make_stratified_group_splits(
    train: pd.DataFrame,
    seed: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    scenario_target_mean = train.groupby("scenario_id")[TARGET].mean()
    scenario_bins = pd.qcut(scenario_target_mean, q=10, labels=False, duplicates="drop")
    row_bins = train["scenario_id"].map(scenario_bins).astype(int).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SEED if seed is None else seed,
    )
    splits = list(
        splitter.split(
            train,
            y=row_bins,
            groups=train["scenario_id"].to_numpy(),
        )
    )

    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for fold_idx, (_, val_idx) in enumerate(splits, start=1):
        fold_ids[val_idx] = fold_idx
    if (fold_ids < 0).any():
        raise ValueError("Some rows were not assigned to a scenario CV fold.")

    return splits, fold_ids


def make_model_key(model_family: str, objective: dict[str, Any], seed: int | None = None) -> str:
    base_key = f"{model_family}_{objective['name']}"
    return base_key if seed is None else f"{base_key}_seed{seed}"


def get_model_key_family(model_key: str) -> str:
    for objective in [objective["name"] for objective in OBJECTIVES]:
        seeded_marker = f"_{objective}_seed"
        if seeded_marker in model_key:
            return model_key.split(seeded_marker, maxsplit=1)[0]
        objective_suffix = f"_{objective}"
        if model_key.endswith(objective_suffix):
            return model_key[: -len(objective_suffix)]
    return model_key


def get_model_family_seeds(model_family: str) -> list[int]:
    return MODEL_FAMILY_SEEDS.get(model_family, MODEL_SEEDS)


def make_lightgbm_params(objective: dict[str, Any], seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 1800,
        "learning_rate": 0.035,
        "max_depth": -1,
        "num_leaves": 96,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }
    params.update(MODEL_PARAMS.get("lightgbm", {}))
    params.update(objective["params"]["lightgbm"])
    return params


def make_xgboost_params(
    model_family: str,
    objective: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 1800,
        "learning_rate": 0.035,
        "max_depth": 8,
        "min_child_weight": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "mae",
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }
    params.update(MODEL_PARAMS.get("xgboost", {}))
    if model_family != "xgboost":
        params.update(MODEL_PARAMS.get(model_family, {}))
    params.update(objective["params"]["xgboost"])
    return params


def make_catboost_params(objective: dict[str, Any], seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "iterations": 1800,
        "learning_rate": 0.035,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "eval_metric": "MAE",
        "od_type": "Iter",
        "od_wait": 100,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(MODEL_PARAMS.get("catboost", {}))
    params.update(objective["params"]["catboost"])
    return params


def fit_predict_fold(
    model_family: str,
    objective: dict[str, Any],
    seed: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    if model_family == "lightgbm":
        model = LGBMRegressor(**make_lightgbm_params(objective, seed))
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(model.best_iteration_ or model.n_estimators_)
        valid_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        return valid_pred, test_pred, best_iteration

    if model_family.startswith("xgboost"):
        model = XGBRegressor(**make_xgboost_params(model_family, objective, seed))
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_index = getattr(model, "best_iteration", None)
        best_iteration = (
            int(best_index + 1) if best_index is not None else int(model.n_estimators)
        )
        predict_kwargs = {"iteration_range": (0, best_iteration)}
        valid_pred = model.predict(X_val, **predict_kwargs)
        test_pred = model.predict(X_test, **predict_kwargs)
        return valid_pred, test_pred, best_iteration

    if model_family == "catboost":
        model = CatBoostRegressor(**make_catboost_params(objective, seed))
        model.fit(
            X_tr,
            y_tr,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        best_index = model.get_best_iteration()
        best_iteration = (
            int(best_index + 1) if best_index is not None else int(model.tree_count_)
        )
        valid_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        return valid_pred, test_pred, best_iteration

    raise ValueError(f"Unsupported model family: {model_family}")


def train_one_cv(
    model_family: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    objective: dict[str, Any],
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model_key = make_model_key(model_family, objective, model_seed)
    log(f"[train] start {model_key}")

    oof = np.full(len(train), np.nan, dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    model_started_at = time.time()

    fold_iter = tqdm(
        enumerate(splits, start=1),
        total=len(splits),
        desc=f"{model_key} folds",
        unit="fold",
        leave=False,
        dynamic_ncols=True,
    )
    for fold, (tr_idx, val_idx) in fold_iter:
        started_at = time.time()
        valid_pred, fold_test_pred, best_iteration = fit_predict_fold(
            model_family=model_family,
            objective=objective,
            seed=model_seed,
            X_tr=train.iloc[tr_idx][feature_cols],
            y_tr=train.iloc[tr_idx][TARGET],
            X_val=train.iloc[val_idx][feature_cols],
            y_val=train.iloc[val_idx][TARGET],
            X_test=test[feature_cols],
            categorical_features=categorical_features,
        )

        oof[val_idx] = valid_pred
        test_pred += fold_test_pred / len(splits)
        fold_mae = mean_absolute_error(train.iloc[val_idx][TARGET], valid_pred)
        elapsed = time.time() - started_at

        report = {
            "model_key": model_key,
            "model_family": model_family,
            "objective": objective["name"],
            "model_seed": model_seed,
            "split_seed": model_seed,
            "fold": fold,
            "valid_rows": int(len(val_idx)),
            "mae": to_float(fold_mae),
            "best_iteration": best_iteration,
            "elapsed_sec": to_float(elapsed),
        }
        fold_reports.append(report)
        fold_iter.set_postfix(
            {
                "mae": f"{fold_mae:.5f}",
                "best": best_iteration,
                "time": format_duration(elapsed),
            }
        )
        log(
            f"[train] {model_key} fold={fold} "
            f"mae={fold_mae:.5f} best_iter={best_iteration} "
            f"elapsed={format_duration(elapsed)}"
        )

    if np.isnan(oof).any():
        raise ValueError(f"OOF prediction has NaN values for {model_key}.")

    overall_mae = mean_absolute_error(train[TARGET], oof)
    model_elapsed = time.time() - model_started_at
    log(
        f"[train] done {model_key} "
        f"oof_mae={overall_mae:.5f} elapsed={format_duration(model_elapsed)}"
    )
    return oof, test_pred, fold_reports


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    weight_sum = float(normalized.sum())
    if weight_sum <= 0:
        return np.full(len(normalized), 1.0 / len(normalized), dtype=np.float64)
    return normalized / weight_sum


def make_ensemble_weight_starts(model_keys: list[str]) -> list[tuple[str, np.ndarray]]:
    n_models = len(model_keys)
    starts: list[tuple[str, np.ndarray]] = [
        ("uniform", np.full(n_models, 1.0 / n_models, dtype=np.float64))
    ]

    for idx, model_key in enumerate(model_keys):
        start = np.zeros(n_models, dtype=np.float64)
        start[idx] = 1.0
        starts.append((f"single:{model_key}", start))

    for model_family in MODEL_FAMILIES:
        indices = [
            idx
            for idx, model_key in enumerate(model_keys)
            if get_model_key_family(model_key) == model_family
        ]
        if indices and len(indices) < n_models:
            start = np.zeros(n_models, dtype=np.float64)
            start[indices] = 1.0 / len(indices)
            starts.append((f"family:{model_family}", start))

    for seed in MODEL_SEEDS:
        suffix = f"_seed{seed}"
        indices = [
            idx
            for idx, model_key in enumerate(model_keys)
            if model_key.endswith(suffix)
        ]
        if indices and len(indices) < n_models:
            start = np.zeros(n_models, dtype=np.float64)
            start[indices] = 1.0 / len(indices)
            starts.append((f"seed:{seed}", start))

    return starts


def optimize_ensemble_weights(
    model_oofs: dict[str, np.ndarray],
    y: pd.Series,
) -> dict[str, Any]:
    model_keys = list(model_oofs.keys())
    if not model_keys:
        raise ValueError("No model predictions available for ensemble.")

    y_values = y.to_numpy(dtype=np.float64)
    oof_matrix = np.column_stack([model_oofs[key] for key in model_keys]).astype(np.float64)
    n_models = len(model_keys)

    if n_models == 1:
        weights = np.array([1.0], dtype=np.float64)
        mae = mean_absolute_error(y_values, oof_matrix[:, 0])
        return {
            "type": "single_model",
            "model_keys": model_keys,
            "weights": {model_keys[0]: 1.0},
            "weight_values": weights.tolist(),
            "mae": to_float(mae),
            "simple_average_mae": to_float(mae),
            "optimizer": {"enabled": False, "reason": "single_model"},
            "candidates": [],
        }

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.abs(y_values - oof_matrix @ weights)))

    simple_weights = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    best_weights = simple_weights.copy()
    best_mae = objective(best_weights)
    candidate_reports: list[dict[str, Any]] = [
        {
            "start": "uniform",
            "success": True,
            "message": "initial uniform average",
            "mae": to_float(best_mae),
            "weights": {
                key: to_float(weight)
                for key, weight in zip(model_keys, best_weights, strict=True)
            },
        }
    ]

    constraints = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    bounds = [(0.0, 1.0)] * n_models
    starts = make_ensemble_weight_starts(model_keys)

    for start_name, start_weights in starts:
        result = minimize(
            objective,
            normalize_weights(start_weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        weights = normalize_weights(result.x if result.x is not None else start_weights)
        mae = objective(weights)
        candidate_reports.append(
            {
                "start": start_name,
                "success": bool(result.success),
                "message": str(result.message),
                "mae": to_float(mae),
                "weights": {
                    key: to_float(weight)
                    for key, weight in zip(model_keys, weights, strict=True)
                },
            }
        )
        if mae < best_mae:
            best_mae = mae
            best_weights = weights

    cleaned_weights = best_weights.copy()
    cleaned_weights[cleaned_weights < 1e-8] = 0.0
    cleaned_weights = normalize_weights(cleaned_weights)
    cleaned_mae = objective(cleaned_weights)
    if cleaned_mae <= best_mae + 1e-10:
        best_weights = cleaned_weights
        best_mae = cleaned_mae

    return {
        "type": "mae_optimized_nonnegative_weighted_average",
        "model_keys": model_keys,
        "weights": {
            key: to_float(weight)
            for key, weight in zip(model_keys, best_weights, strict=True)
        },
        "weight_values": [to_float(weight) for weight in best_weights],
        "mae": to_float(best_mae),
        "simple_average_mae": to_float(objective(simple_weights)),
        "optimizer": {
            "enabled": True,
            "method": "scipy.optimize.minimize:SLSQP",
            "bounds": "0 <= weight <= 1",
            "constraint": "sum(weights) == 1",
            "start_count": int(len(starts)),
        },
        "candidates": candidate_reports,
    }


def apply_ensemble_weights(
    model_predictions: dict[str, np.ndarray],
    ensemble_report: dict[str, Any],
) -> np.ndarray:
    model_keys = list(ensemble_report["model_keys"])
    missing_keys = sorted(set(model_keys) - set(model_predictions))
    if missing_keys:
        raise ValueError(f"Missing model predictions for ensemble: {missing_keys}")

    weights = np.array(
        [ensemble_report["weights"][key] for key in model_keys],
        dtype=np.float64,
    )
    prediction_matrix = np.column_stack([model_predictions[key] for key in model_keys])
    return prediction_matrix @ weights


def evaluate_layout_group_risk(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    log("[layout-risk] start GroupKFold by layout_id")
    analysis_seed = MODEL_SEEDS[0]
    splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    splits = list(splitter.split(train, groups=train["layout_id"].to_numpy()))
    model_family = "lightgbm"
    objective = OBJECTIVES[0]

    oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    layout_started_at = time.time()
    fold_iter = tqdm(
        enumerate(splits, start=1),
        total=len(splits),
        desc="layout-risk folds",
        unit="fold",
        leave=False,
        dynamic_ncols=True,
    )
    for fold, (tr_idx, val_idx) in fold_iter:
        started_at = time.time()
        model = LGBMRegressor(**make_lightgbm_params(objective, analysis_seed))
        model.fit(
            train.iloc[tr_idx][feature_cols],
            train.iloc[tr_idx][TARGET],
            eval_set=[(train.iloc[val_idx][feature_cols], train.iloc[val_idx][TARGET])],
            eval_metric="mae",
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        valid_pred = model.predict(
            train.iloc[val_idx][feature_cols],
            num_iteration=model.best_iteration_,
        )
        oof[val_idx] = valid_pred
        fold_mae = mean_absolute_error(train.iloc[val_idx][TARGET], valid_pred)
        elapsed = time.time() - started_at
        best_iteration = int(model.best_iteration_ or model.n_estimators_)
        fold_reports.append(
            {
                "fold": fold,
                "valid_rows": int(len(val_idx)),
                "unique_layouts": int(train.iloc[val_idx]["layout_id"].nunique()),
                "mae": to_float(fold_mae),
                "best_iteration": best_iteration,
                "elapsed_sec": to_float(elapsed),
            }
        )
        fold_iter.set_postfix(
            {
                "mae": f"{fold_mae:.5f}",
                "best": best_iteration,
                "time": format_duration(elapsed),
            }
        )
        log(
            f"[layout-risk] fold={fold} mae={fold_mae:.5f} "
            f"best_iter={best_iteration} elapsed={format_duration(elapsed)}"
        )

    if np.isnan(oof).any():
        raise ValueError("Layout GroupKFold OOF prediction has NaN values.")

    fold_scores = [r["mae"] for r in fold_reports]
    result = {
        "model_family": model_family,
        "objective": objective["name"],
        "model_seed": analysis_seed,
        "split_seed": CV_SEED,
        "mae": to_float(mean_absolute_error(train[TARGET], oof)),
        "fold_mean": to_float(np.mean(fold_scores)),
        "fold_std": to_float(np.std(fold_scores)),
        "folds": fold_reports,
        "elapsed_sec": to_float(time.time() - layout_started_at),
    }
    log(
        f"[layout-risk] done oof_mae={result['mae']:.5f} "
        f"fold_std={result['fold_std']:.5f} "
        f"elapsed={format_duration(result['elapsed_sec'])}"
    )
    return result


def fit_lightgbm_oof_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    objective: dict[str, Any],
    seed: int,
    desc: str,
    *,
    keep_models: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    fitted_folds: list[dict[str, Any]] = []
    fold_iter = tqdm(
        enumerate(splits, start=1),
        total=len(splits),
        desc=desc,
        unit="fold",
        leave=False,
        dynamic_ncols=True,
    )

    for fold, (tr_idx, val_idx) in fold_iter:
        started_at = time.time()
        model = LGBMRegressor(**make_lightgbm_params(objective, seed))
        model.fit(
            train.iloc[tr_idx][feature_cols],
            train.iloc[tr_idx][TARGET],
            eval_set=[(train.iloc[val_idx][feature_cols], train.iloc[val_idx][TARGET])],
            eval_metric="mae",
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        valid_pred = model.predict(
            train.iloc[val_idx][feature_cols],
            num_iteration=model.best_iteration_,
        )
        oof[val_idx] = valid_pred
        fold_mae = mean_absolute_error(train.iloc[val_idx][TARGET], valid_pred)
        elapsed = time.time() - started_at
        best_iteration = int(model.best_iteration_ or model.n_estimators_)
        fold_report = {
            "fold": fold,
            "valid_rows": int(len(val_idx)),
            "mae": to_float(fold_mae),
            "best_iteration": best_iteration,
            "elapsed_sec": to_float(elapsed),
        }
        fold_reports.append(fold_report)
        if keep_models:
            fitted_folds.append(
                {
                    "fold": fold,
                    "model": model,
                    "val_idx": val_idx,
                    "best_iteration": best_iteration,
                }
            )
        fold_iter.set_postfix(
            {
                "mae": f"{fold_mae:.5f}",
                "best": best_iteration,
                "time": format_duration(elapsed),
            }
        )

    if np.isnan(oof).any():
        raise ValueError(f"OOF prediction has NaN values for {desc}.")
    return oof, fold_reports, fitted_folds


def run_group_ablation(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_group_map: dict[str, str],
    baseline_mae: float,
    objective: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    log("[analysis] start feature group ablation")
    started_at = time.time()
    groups: list[dict[str, Any]] = []
    for group in FEATURE_GROUP_NAMES:
        removed_features = [
            col for col in feature_cols if feature_group_map.get(col) == group
        ]
        if not removed_features:
            continue

        ablated_features = [col for col in feature_cols if col not in removed_features]
        ablated_categoricals = [
            col for col in categorical_features if col in ablated_features
        ]
        oof, fold_reports, _ = fit_lightgbm_oof_models(
            train=train,
            feature_cols=ablated_features,
            categorical_features=ablated_categoricals,
            splits=splits,
            objective=objective,
            seed=seed,
            desc=f"ablation {group}",
            keep_models=False,
        )
        mae = mean_absolute_error(train[TARGET], oof)
        result = {
            "group": group,
            "removed_feature_count": int(len(removed_features)),
            "removed_features": removed_features,
            "mae": to_float(mae),
            "delta_mae_vs_baseline": to_float(mae - baseline_mae),
            "folds": fold_reports,
        }
        groups.append(result)
        log(
            f"[analysis] ablation group={group} "
            f"mae={mae:.5f} delta={mae - baseline_mae:.5f}"
        )

    return {
        "enabled": True,
        "baseline_mae": to_float(baseline_mae),
        "groups": groups,
        "elapsed_sec": to_float(time.time() - started_at),
    }


def shuffled_feature_values(series: pd.Series, rng: np.random.Generator) -> Any:
    values = series.to_numpy(copy=True)
    rng.shuffle(values)
    if isinstance(series.dtype, CategoricalDtype):
        return pd.Categorical(
            values,
            categories=series.cat.categories,
            ordered=series.cat.ordered,
        )
    return values


def summarize_permutation_by_group(
    feature_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    group_summary: dict[str, dict[str, Any]] = {}
    for group in FEATURE_GROUP_NAMES:
        group_features = [r for r in feature_reports if r["group"] == group]
        if not group_features:
            continue
        deltas = [r["delta_mae_mean"] for r in group_features]
        group_summary[group] = {
            "feature_count": int(len(group_features)),
            "delta_mae_mean": to_float(np.mean(deltas)),
            "recommended_drop_count": int(
                sum(1 for r in group_features if r["recommended_drop"])
            ),
        }
    return group_summary


def run_permutation_importance(
    train: pd.DataFrame,
    feature_cols: list[str],
    fitted_folds: list[dict[str, Any]],
    feature_group_map: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    if ANALYSIS_PERMUTATION_N_REPEATS < 1:
        raise ValueError("analysis.permutation_importance.n_repeats must be >= 1.")

    log("[analysis] start OOF permutation importance")
    started_at = time.time()
    feature_fold_deltas: dict[str, list[float]] = {col: [] for col in feature_cols}
    feature_repeat_deltas: dict[str, list[float]] = {col: [] for col in feature_cols}
    baseline_fold_reports: list[dict[str, Any]] = []

    for fold_info in fitted_folds:
        fold = int(fold_info["fold"])
        model = fold_info["model"]
        val_idx = fold_info["val_idx"]
        X_val = train.iloc[val_idx][feature_cols]
        y_val = train.iloc[val_idx][TARGET]

        if (
            ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD is not None
            and len(X_val) > ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD
        ):
            rng = np.random.default_rng(seed + fold)
            sample_pos = np.sort(
                rng.choice(
                    len(X_val),
                    size=ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD,
                    replace=False,
                )
            )
            X_eval = X_val.iloc[sample_pos].copy()
            y_eval = y_val.iloc[sample_pos]
        else:
            X_eval = X_val.copy()
            y_eval = y_val

        baseline_pred = model.predict(X_eval, num_iteration=model.best_iteration_)
        baseline_mae = mean_absolute_error(y_eval, baseline_pred)
        baseline_fold_reports.append(
            {
                "fold": fold,
                "rows": int(len(X_eval)),
                "baseline_mae": to_float(baseline_mae),
            }
        )

        for feature in feature_cols:
            fold_deltas: list[float] = []
            for repeat in range(ANALYSIS_PERMUTATION_N_REPEATS):
                rng = np.random.default_rng(seed + fold * 1000 + repeat)
                X_permuted = X_eval.copy()
                X_permuted[feature] = shuffled_feature_values(X_eval[feature], rng)
                permuted_pred = model.predict(
                    X_permuted,
                    num_iteration=model.best_iteration_,
                )
                permuted_mae = mean_absolute_error(y_eval, permuted_pred)
                delta = to_float(permuted_mae - baseline_mae)
                fold_deltas.append(delta)
                feature_repeat_deltas[feature].append(delta)
            feature_fold_deltas[feature].append(to_float(np.mean(fold_deltas)))

    feature_reports: list[dict[str, Any]] = []
    for feature in feature_cols:
        fold_deltas = np.asarray(feature_fold_deltas[feature], dtype=np.float64)
        repeat_deltas = np.asarray(feature_repeat_deltas[feature], dtype=np.float64)
        positive_fold_rate = to_float(np.mean(fold_deltas > 0))
        delta_mean = to_float(np.mean(repeat_deltas))
        delta_std = to_float(np.std(repeat_deltas))
        feature_reports.append(
            {
                "feature": feature,
                "group": feature_group_map.get(feature, "misc"),
                "delta_mae_mean": delta_mean,
                "delta_mae_std": delta_std,
                "positive_fold_rate": positive_fold_rate,
                "recommended_drop": bool(
                    delta_mean <= ANALYSIS_RECOMMENDED_DROP_DELTA_THRESHOLD
                    or positive_fold_rate < ANALYSIS_RECOMMENDED_DROP_POSITIVE_FOLD_RATE
                ),
                "fold_delta_mae_mean": [to_float(value) for value in fold_deltas],
            }
        )

    feature_reports = sorted(
        feature_reports,
        key=lambda r: r["delta_mae_mean"],
    )
    recommended_drop_features = [
        r["feature"] for r in feature_reports if r["recommended_drop"]
    ]
    return {
        "enabled": True,
        "max_rows_per_fold": ANALYSIS_PERMUTATION_MAX_ROWS_PER_FOLD,
        "n_repeats": ANALYSIS_PERMUTATION_N_REPEATS,
        "recommended_drop_delta_threshold": ANALYSIS_RECOMMENDED_DROP_DELTA_THRESHOLD,
        "recommended_drop_positive_fold_rate": ANALYSIS_RECOMMENDED_DROP_POSITIVE_FOLD_RATE,
        "baseline_folds": baseline_fold_reports,
        "recommended_drop_features": recommended_drop_features,
        "features": feature_reports,
        "by_group": summarize_permutation_by_group(feature_reports),
        "elapsed_sec": to_float(time.time() - started_at),
    }


def run_feature_analysis(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_group_map: dict[str, str],
) -> dict[str, Any]:
    if not ANALYSIS_ENABLED:
        return {"enabled": False}

    analysis_seed = MODEL_SEEDS[0]
    objective = OBJECTIVES[0]
    started_at = time.time()
    log(
        "[analysis] enabled "
        f"model_family=lightgbm objective={objective['name']} seed={analysis_seed}"
    )

    baseline_oof, baseline_folds, fitted_folds = fit_lightgbm_oof_models(
        train=train,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        splits=splits,
        objective=objective,
        seed=analysis_seed,
        desc="analysis baseline",
        keep_models=ANALYSIS_PERMUTATION_ENABLED,
    )
    baseline_mae = mean_absolute_error(train[TARGET], baseline_oof)
    result: dict[str, Any] = {
        "enabled": True,
        "model_family": "lightgbm",
        "objective": objective["name"],
        "model_seed": analysis_seed,
        "split_seed": analysis_seed,
        "baseline": {
            "mae": to_float(baseline_mae),
            "folds": baseline_folds,
        },
    }

    if ANALYSIS_ABLATION_ENABLED:
        result["group_ablation"] = run_group_ablation(
            train=train,
            feature_cols=feature_cols,
            categorical_features=categorical_features,
            splits=splits,
            feature_group_map=feature_group_map,
            baseline_mae=to_float(baseline_mae),
            objective=objective,
            seed=analysis_seed,
        )
    else:
        result["group_ablation"] = {"enabled": False}

    if ANALYSIS_PERMUTATION_ENABLED:
        result["permutation_importance"] = run_permutation_importance(
            train=train,
            feature_cols=feature_cols,
            fitted_folds=fitted_folds,
            feature_group_map=feature_group_map,
            seed=analysis_seed,
        )
    else:
        result["permutation_importance"] = {"enabled": False}

    result["elapsed_sec"] = to_float(time.time() - started_at)
    log(f"[analysis] done elapsed={format_duration(result['elapsed_sec'])}")
    return result


def apply_postprocessing(
    pred: np.ndarray,
    steps: pd.Series,
    *,
    lower_clip: bool,
    upper_cap: float | None,
    step_offsets: dict[int, float] | None,
) -> np.ndarray:
    out = pred.astype(np.float64, copy=True)
    if step_offsets is not None:
        offsets = steps.map(step_offsets).fillna(0.0).to_numpy(dtype=np.float64)
        out += offsets
    if lower_clip:
        out = np.maximum(out, 0.0)
    if upper_cap is not None:
        out = np.minimum(out, upper_cap)
    return out


def select_postprocessing(
    train: pd.DataFrame,
    raw_oof: np.ndarray,
    raw_test_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    y = train[TARGET].to_numpy(dtype=np.float64)
    steps = train["scenario_step"]
    raw_residual = y - raw_oof
    step_offsets = (
        pd.DataFrame({"scenario_step": steps, "residual": raw_residual})
        .groupby("scenario_step")["residual"]
        .median()
        .to_dict()
    )
    step_offsets = {int(k): float(v) for k, v in step_offsets.items()}

    cap_options = [
        ("none", None),
        ("target_q995", to_float(train[TARGET].quantile(0.995))),
        ("target_q999", to_float(train[TARGET].quantile(0.999))),
    ]
    candidate_reports: list[dict[str, Any]] = []

    for use_step_offsets in [False, True]:
        for lower_clip in [False, True]:
            for cap_name, upper_cap in cap_options:
                processed = apply_postprocessing(
                    raw_oof,
                    steps,
                    lower_clip=lower_clip,
                    upper_cap=upper_cap,
                    step_offsets=step_offsets if use_step_offsets else None,
                )
                candidate_reports.append(
                    {
                        "use_step_offsets": bool(use_step_offsets),
                        "lower_clip": bool(lower_clip),
                        "upper_cap_name": cap_name,
                        "upper_cap": None if upper_cap is None else to_float(upper_cap),
                        "mae": to_float(mean_absolute_error(y, processed)),
                        "negative_count": int((processed < 0).sum()),
                    }
                )

    valid_candidates = [c for c in candidate_reports if c["lower_clip"]]
    selected = min(valid_candidates, key=lambda c: c["mae"])
    selected = dict(selected)
    selected["step_offsets"] = step_offsets if selected["use_step_offsets"] else {}

    post_oof = apply_postprocessing(
        raw_oof,
        steps,
        lower_clip=selected["lower_clip"],
        upper_cap=selected["upper_cap"],
        step_offsets=step_offsets if selected["use_step_offsets"] else None,
    )
    post_test = apply_postprocessing(
        raw_test_pred,
        train.attrs["test_scenario_step"],
        lower_clip=selected["lower_clip"],
        upper_cap=selected["upper_cap"],
        step_offsets=step_offsets if selected["use_step_offsets"] else None,
    )

    selected["raw_oof_mae"] = to_float(mean_absolute_error(y, raw_oof))
    selected["postprocessed_oof_mae"] = to_float(mean_absolute_error(y, post_oof))
    selected["raw_negative_count"] = int((raw_oof < 0).sum())
    selected["postprocessed_negative_count"] = int((post_oof < 0).sum())
    return post_oof, post_test, selected, candidate_reports


def build_oof_frame(
    train: pd.DataFrame,
    fold_ids_by_seed: dict[int, np.ndarray],
    model_oofs: dict[str, np.ndarray],
    ensemble_raw_oof: np.ndarray,
    ensemble_post_oof: np.ndarray,
) -> pd.DataFrame:
    oof_df = train[ID_COLS + ["scenario_step", TARGET]].copy()
    first_seed = MODEL_SEEDS[0]
    oof_df["scenario_cv_fold"] = fold_ids_by_seed[first_seed]
    for seed, fold_ids in fold_ids_by_seed.items():
        oof_df[f"scenario_cv_fold_seed{seed}"] = fold_ids

    for key, values in model_oofs.items():
        oof_df[f"oof_{key}"] = values

    for model_family in MODEL_FAMILIES:
        cols = [
            values
            for key, values in model_oofs.items()
            if get_model_key_family(key) == model_family
        ]
        oof_df[f"oof_family_{model_family}"] = np.mean(cols, axis=0)

    for objective in [o["name"] for o in OBJECTIVES]:
        cols = [
            values
            for key, values in model_oofs.items()
            if f"_{objective}_seed" in key
        ]
        oof_df[f"oof_objective_{objective}"] = np.mean(cols, axis=0)

    for seed in MODEL_SEEDS:
        cols = [
            values
            for key, values in model_oofs.items()
            if key.endswith(f"_seed{seed}")
        ]
        oof_df[f"oof_seed{seed}"] = np.mean(cols, axis=0)

    oof_df["oof_ensemble_raw"] = ensemble_raw_oof
    oof_df["oof_ensemble_postprocessed"] = ensemble_post_oof
    return oof_df


def save_submission(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    test_pred: np.ndarray,
) -> pd.DataFrame:
    submission = sample_submission.copy()
    if not submission["ID"].equals(test["ID"]):
        raise ValueError("sample_submission ID order does not match test ID order.")

    submission[TARGET] = test_pred
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test row count.")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains NaN predictions.")
    if (submission[TARGET] < 0).any():
        raise ValueError("Submission contains negative predictions.")
    if list(submission.columns) != ["ID", TARGET]:
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")

    submission.to_csv(SUBMISSION_PATH, index=False)
    return submission


def summarize_scores(model_oofs: dict[str, np.ndarray], y: pd.Series) -> dict[str, Any]:
    by_model = {
        key: to_float(mean_absolute_error(y, values))
        for key, values in model_oofs.items()
    }
    by_family = {}
    for model_family in MODEL_FAMILIES:
        cols = [
            values
            for key, values in model_oofs.items()
            if get_model_key_family(key) == model_family
        ]
        by_family[model_family] = to_float(mean_absolute_error(y, np.mean(cols, axis=0)))

    by_objective = {}
    for objective in [o["name"] for o in OBJECTIVES]:
        cols = [
            values
            for key, values in model_oofs.items()
            if f"_{objective}_seed" in key
        ]
        by_objective[objective] = to_float(mean_absolute_error(y, np.mean(cols, axis=0)))

    by_seed = {}
    for seed in MODEL_SEEDS:
        cols = [
            values
            for key, values in model_oofs.items()
            if key.endswith(f"_seed{seed}")
        ]
        by_seed[str(seed)] = to_float(mean_absolute_error(y, np.mean(cols, axis=0)))

    return {
        "by_model": by_model,
        "by_family": by_family,
        "by_objective": by_objective,
        "by_seed": by_seed,
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    resolved_cfg = apply_hydra_config(cfg)
    setup_logging()
    started_at = time.time()
    phase_timings: dict[str, float] = {}

    phase_started_at = time.time()
    log("[start] load data")
    train_raw, test_raw, layout, sample_submission = load_data()
    phase_timings["load_data_sec"] = to_float(time.time() - phase_started_at)
    log(f"[data] train={train_raw.shape} test={test_raw.shape} layout={layout.shape}")

    phase_started_at = time.time()
    log("[features] build features")
    train, test, feature_cols, categorical_features, feature_group_map, pruning_report = (
        build_features(train_raw, test_raw, layout)
    )
    train.attrs["test_scenario_step"] = test["scenario_step"]
    phase_timings["feature_build_sec"] = to_float(time.time() - phase_started_at)
    log(
        f"[features] total={len(feature_cols)} "
        f"categorical={categorical_features} "
        f"elapsed={format_duration(phase_timings['feature_build_sec'])}"
    )

    phase_started_at = time.time()
    log("[cv] build seed-specific StratifiedGroupKFold by scenario_id")
    scenario_splits_by_seed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    fold_ids_by_seed: dict[int, np.ndarray] = {}
    for seed in MODEL_SEEDS:
        scenario_splits, fold_ids = make_stratified_group_splits(train, seed=seed)
        scenario_splits_by_seed[seed] = scenario_splits
        fold_ids_by_seed[seed] = fold_ids
    phase_timings["scenario_cv_split_sec"] = to_float(time.time() - phase_started_at)

    model_oofs: dict[str, np.ndarray] = {}
    model_test_preds: dict[str, np.ndarray] = {}
    fold_reports: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []

    train_started_at = time.time()
    training_tasks = [
        (seed, model_family, objective)
        for model_family in MODEL_FAMILIES
        for seed in get_model_family_seeds(model_family)
        for objective in OBJECTIVES
    ]
    model_iter = tqdm(
        training_tasks,
        total=len(training_tasks),
        desc="scenario models",
        unit="model",
        dynamic_ncols=True,
    )
    for seed, model_family, objective in model_iter:
        key = make_model_key(model_family, objective, seed)
        model_started_at = time.time()
        oof, test_pred, reports = train_one_cv(
            model_family=model_family,
            train=train,
            test=test,
            feature_cols=feature_cols,
            categorical_features=categorical_features,
            splits=scenario_splits_by_seed[seed],
            objective=objective,
            model_seed=seed,
        )
        model_elapsed = time.time() - model_started_at
        model_mae = mean_absolute_error(train[TARGET], oof)
        model_oofs[key] = oof
        model_test_preds[key] = test_pred
        fold_reports.extend(reports)
        model_reports.append(
            {
                "model_key": key,
                "model_family": model_family,
                "objective": objective["name"],
                "model_seed": seed,
                "split_seed": seed,
                "mae": to_float(model_mae),
                "elapsed_sec": to_float(model_elapsed),
            }
        )
        model_iter.set_postfix(
            {
                "last": key,
                "mae": f"{model_mae:.5f}",
                "time": format_duration(model_elapsed),
            }
        )
    phase_timings["scenario_training_sec"] = to_float(time.time() - train_started_at)
    log(
        "[train] scenario model training total "
        f"elapsed={format_duration(phase_timings['scenario_training_sec'])}"
    )

    phase_started_at = time.time()
    log(f"[ensemble] optimize non-negative weights for {len(model_oofs)} model predictions")
    ensemble_weighting = optimize_ensemble_weights(model_oofs, train[TARGET])
    ensemble_raw_oof = apply_ensemble_weights(model_oofs, ensemble_weighting)
    ensemble_raw_test_pred = apply_ensemble_weights(model_test_preds, ensemble_weighting)
    phase_timings["ensemble_sec"] = to_float(time.time() - phase_started_at)
    log(
        "[ensemble] "
        f"simple_mae={ensemble_weighting['simple_average_mae']:.5f} "
        f"weighted_mae={ensemble_weighting['mae']:.5f} "
        f"elapsed={format_duration(phase_timings['ensemble_sec'])}"
    )

    phase_started_at = time.time()
    log("[postprocess] evaluate OOF postprocessing candidates")
    ensemble_post_oof, ensemble_post_test_pred, selected_postprocessing, postprocess_reports = (
        select_postprocessing(train, ensemble_raw_oof, ensemble_raw_test_pred)
    )
    phase_timings["postprocess_sec"] = to_float(time.time() - phase_started_at)

    phase_started_at = time.time()
    log("[layout-risk] evaluate unseen layout risk")
    layout_risk = evaluate_layout_group_risk(
        train=train,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
    )
    phase_timings["layout_risk_sec"] = to_float(time.time() - phase_started_at)

    phase_started_at = time.time()
    feature_analysis = run_feature_analysis(
        train=train,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        splits=scenario_splits_by_seed[MODEL_SEEDS[0]],
        feature_group_map=feature_group_map,
    )
    phase_timings["feature_analysis_sec"] = to_float(time.time() - phase_started_at)

    phase_started_at = time.time()
    log("[output] write OOF, submission, report")
    oof_df = build_oof_frame(
        train=train,
        fold_ids_by_seed=fold_ids_by_seed,
        model_oofs=model_oofs,
        ensemble_raw_oof=ensemble_raw_oof,
        ensemble_post_oof=ensemble_post_oof,
    )
    oof_df.to_csv(OOF_PATH, index=False)

    submission = save_submission(sample_submission, test_raw, ensemble_post_test_pred)
    phase_timings["write_outputs_sec"] = to_float(time.time() - phase_started_at)

    y = train[TARGET]
    score_summary = summarize_scores(model_oofs, y)
    report = {
        "config": {
            "run_date": RUN_DATE,
            "run_timestamp": RUN_TIMESTAMP,
            "run_id": RUN_ID,
            "n_splits": N_SPLITS,
            "cv_seed": CV_SEED,
            "model_seed": MODEL_SEED,
            "model_seeds": MODEL_SEEDS,
            "model_families": MODEL_FAMILIES,
            "model_family_seeds": MODEL_FAMILY_SEEDS,
            "model_params": MODEL_PARAMS,
            "objectives": OBJECTIVES,
            "lag_rolling_cols": LAG_ROLLING_COLS,
            "hydra": resolved_cfg,
            "outputs": {
                "output_dir": path_str(OUTPUT_DIR),
                "submission": path_str(SUBMISSION_PATH),
                "oof": path_str(OOF_PATH),
                "report": path_str(REPORT_PATH),
                "log_file": path_str(LOG_PATH),
                "hydra_run_dir": path_str(HYDRA_RUN_DIR),
                "hydra_log_file": path_str(HYDRA_LOG_PATH),
            },
        },
        "data": {
            "train_shape": list(train_raw.shape),
            "test_shape": list(test_raw.shape),
            "layout_shape": list(layout.shape),
            "feature_count": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "categorical_features": categorical_features,
            "temporal_source_columns_excluded": TEMPORAL_SOURCE_COLS,
            "feature_pruning": pruning_report,
            "feature_groups": {
                "counts": count_feature_groups(feature_group_map),
                "by_feature": feature_group_map,
            },
            "scenario_count": int(train["scenario_id"].nunique()),
            "layout_count": int(train["layout_id"].nunique()),
            "test_layout_count": int(test["layout_id"].nunique()),
        },
        "scenario_group_cv": {
            "folds": fold_reports,
            "models": model_reports,
            "scores": score_summary,
            "ensemble_type": ensemble_weighting["type"],
            "ensemble_members": list(model_oofs.keys()),
            "ensemble_weighting": ensemble_weighting,
            "ensemble_raw_mae": to_float(mean_absolute_error(y, ensemble_raw_oof)),
            "ensemble_postprocessed_mae": to_float(mean_absolute_error(y, ensemble_post_oof)),
            "raw_negative_count": int((ensemble_raw_oof < 0).sum()),
            "postprocessed_negative_count": int((ensemble_post_oof < 0).sum()),
        },
        "layout_group_risk": layout_risk,
        "postprocessing": {
            "candidates": postprocess_reports,
            "selected": selected_postprocessing,
        },
        "feature_analysis": feature_analysis,
        "submission_checks": {
            "rows": int(len(submission)),
            "target_min": to_float(submission[TARGET].min()),
            "target_max": to_float(submission[TARGET].max()),
            "target_mean": to_float(submission[TARGET].mean()),
            "nan_count": int(submission[TARGET].isna().sum()),
            "negative_count": int((submission[TARGET] < 0).sum()),
            "id_order_matches_sample": bool(submission["ID"].equals(sample_submission["ID"])),
        },
        "phase_timings_sec": phase_timings,
        "elapsed_sec": to_float(time.time() - started_at),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(
        "[done] "
        f"raw_oof_mae={report['scenario_group_cv']['ensemble_raw_mae']:.5f} "
        f"post_oof_mae={report['scenario_group_cv']['ensemble_postprocessed_mae']:.5f} "
        f"elapsed={format_duration(report['elapsed_sec'])}"
    )


if __name__ == "__main__":
    main()
