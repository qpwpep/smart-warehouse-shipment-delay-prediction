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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm


warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
MODEL_SEED = 42
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
    global OBJECTIVES

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    TARGET = str(cfg.target)
    ID_COLS = [str(col) for col in cfg.id_cols]
    MODEL_SEED = int(cfg.model_seed)
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

    LAG_ROLLING_COLS = [str(col) for col in cfg.features.lag_rolling_cols]
    MODEL_FAMILIES = [str(model_family) for model_family in cfg.models.families]
    OBJECTIVES = list(OmegaConf.to_container(cfg.objectives, resolve=True))

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


def add_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("scenario_id", sort=False)
    df["scenario_step"] = grouped.cumcount().astype(np.int16)

    for col in LAG_ROLLING_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing lag/rolling source column: {col}")
        lag_col = f"{col}_lag1"
        df[lag_col] = grouped[col].shift(1)
        df[f"{col}_delta1"] = df[col] - df[lag_col]
        df[f"{col}_rolling3_mean"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[f"{col}_rolling5_mean"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )

    return df


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    robot_total = df["robot_total"]
    order_inflow = df["order_inflow_15m"]

    df["inflow_per_active_robot"] = safe_divide(order_inflow, df["robot_active"] + 1)
    df["sku_per_order"] = safe_divide(df["unique_sku_15m"], order_inflow + 1)
    df["charge_pressure"] = (
        df["robot_charging"]
        + df["charge_queue_length"]
        + df["low_battery_ratio"] * robot_total
    )
    df["congestion_pressure"] = (
        df["congestion_score"]
        + 20 * df["max_zone_density"]
        + 2 * df["blocked_path_15m"]
    )
    df["charger_per_robot"] = safe_divide(df["charger_count"], robot_total)
    df["pack_station_per_order"] = safe_divide(df["pack_station_count"], order_inflow + 1)
    df["area_per_robot"] = safe_divide(df["floor_area_sqm"], robot_total)
    df["intersection_per_area"] = safe_divide(df["intersection_count"], df["floor_area_sqm"])
    df["robot_total_gap"] = robot_total - (
        df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    )
    df["active_robot_share"] = safe_divide(df["robot_active"], robot_total)
    return df


def build_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    layout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    layout_type_dtype = CategoricalDtype(
        categories=sorted(layout["layout_type"].dropna().unique()),
        ordered=False,
    )

    train = add_layout_features(train_raw, layout, layout_type_dtype)
    test = add_layout_features(test_raw, layout, layout_type_dtype)

    train = add_scenario_features(train)
    test = add_scenario_features(test)

    train = add_pressure_features(train)
    test = add_pressure_features(test)

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    test_feature_cols = [c for c in test.columns if c not in ID_COLS]
    if feature_cols != test_feature_cols:
        missing_in_test = sorted(set(feature_cols) - set(test_feature_cols))
        extra_in_test = sorted(set(test_feature_cols) - set(feature_cols))
        raise ValueError(
            "Train/test feature mismatch: "
            f"missing_in_test={missing_in_test}, extra_in_test={extra_in_test}"
        )

    object_features = train[feature_cols].select_dtypes(include="object").columns.tolist()
    if object_features:
        raise ValueError(f"Unexpected object feature columns: {object_features}")

    categorical_features = ["layout_type"]
    return train, test, feature_cols, categorical_features


def make_stratified_group_splits(train: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    scenario_target_mean = train.groupby("scenario_id")[TARGET].mean()
    scenario_bins = pd.qcut(scenario_target_mean, q=10, labels=False, duplicates="drop")
    row_bins = train["scenario_id"].map(scenario_bins).astype(int).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SEED,
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


def make_model_key(model_family: str, objective: dict[str, Any]) -> str:
    return f"{model_family}_{objective['name']}"


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
    params.update(objective["params"]["lightgbm"])
    return params


def make_xgboost_params(objective: dict[str, Any], seed: int) -> dict[str, Any]:
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

    if model_family == "xgboost":
        model = XGBRegressor(**make_xgboost_params(objective, seed))
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
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model_key = make_model_key(model_family, objective)
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
            seed=MODEL_SEED,
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
            "model_seed": MODEL_SEED,
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


def evaluate_layout_group_risk(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    log("[layout-risk] start GroupKFold by layout_id")
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
        model = LGBMRegressor(**make_lightgbm_params(objective, MODEL_SEED))
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
        "model_seed": MODEL_SEED,
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
    fold_ids: np.ndarray,
    model_oofs: dict[str, np.ndarray],
    ensemble_raw_oof: np.ndarray,
    ensemble_post_oof: np.ndarray,
) -> pd.DataFrame:
    oof_df = train[ID_COLS + ["scenario_step", TARGET]].copy()
    oof_df["scenario_cv_fold"] = fold_ids

    for key, values in model_oofs.items():
        oof_df[f"oof_{key}"] = values

    for model_family in MODEL_FAMILIES:
        cols = [
            values
            for key, values in model_oofs.items()
            if key.startswith(f"{model_family}_")
        ]
        oof_df[f"oof_family_{model_family}"] = np.mean(cols, axis=0)

    for objective in [o["name"] for o in OBJECTIVES]:
        cols = [
            values
            for key, values in model_oofs.items()
            if key.endswith(f"_{objective}")
        ]
        oof_df[f"oof_objective_{objective}"] = np.mean(cols, axis=0)

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
            if key.startswith(f"{model_family}_")
        ]
        by_family[model_family] = to_float(mean_absolute_error(y, np.mean(cols, axis=0)))

    by_objective = {}
    for objective in [o["name"] for o in OBJECTIVES]:
        cols = [
            values
            for key, values in model_oofs.items()
            if key.endswith(f"_{objective}")
        ]
        by_objective[objective] = to_float(mean_absolute_error(y, np.mean(cols, axis=0)))

    return {
        "by_model": by_model,
        "by_family": by_family,
        "by_objective": by_objective,
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
    train, test, feature_cols, categorical_features = build_features(train_raw, test_raw, layout)
    train.attrs["test_scenario_step"] = test["scenario_step"]
    phase_timings["feature_build_sec"] = to_float(time.time() - phase_started_at)
    log(
        f"[features] total={len(feature_cols)} "
        f"categorical={categorical_features} "
        f"elapsed={format_duration(phase_timings['feature_build_sec'])}"
    )

    phase_started_at = time.time()
    log("[cv] build StratifiedGroupKFold by scenario_id")
    scenario_splits, fold_ids = make_stratified_group_splits(train)
    phase_timings["scenario_cv_split_sec"] = to_float(time.time() - phase_started_at)

    model_oofs: dict[str, np.ndarray] = {}
    model_test_preds: dict[str, np.ndarray] = {}
    fold_reports: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []

    train_started_at = time.time()
    training_tasks = [
        (model_family, objective)
        for model_family in MODEL_FAMILIES
        for objective in OBJECTIVES
    ]
    model_iter = tqdm(
        training_tasks,
        total=len(training_tasks),
        desc="scenario models",
        unit="model",
        dynamic_ncols=True,
    )
    for model_family, objective in model_iter:
        key = make_model_key(model_family, objective)
        model_started_at = time.time()
        oof, test_pred, reports = train_one_cv(
            model_family=model_family,
            train=train,
            test=test,
            feature_cols=feature_cols,
            categorical_features=categorical_features,
            splits=scenario_splits,
            objective=objective,
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
                "model_seed": MODEL_SEED,
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
    log(f"[ensemble] average {len(model_oofs)} model predictions")
    ensemble_raw_oof = np.mean(list(model_oofs.values()), axis=0)
    ensemble_raw_test_pred = np.mean(list(model_test_preds.values()), axis=0)
    phase_timings["ensemble_sec"] = to_float(time.time() - phase_started_at)

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
    log("[output] write OOF, submission, report")
    oof_df = build_oof_frame(
        train=train,
        fold_ids=fold_ids,
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
            "model_families": MODEL_FAMILIES,
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
            "categorical_features": categorical_features,
            "scenario_count": int(train["scenario_id"].nunique()),
            "layout_count": int(train["layout_id"].nunique()),
            "test_layout_count": int(test["layout_id"].nunique()),
        },
        "scenario_group_cv": {
            "folds": fold_reports,
            "models": model_reports,
            "scores": score_summary,
            "ensemble_type": "model_average",
            "ensemble_members": list(model_oofs.keys()),
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
