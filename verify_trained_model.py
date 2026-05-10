"""Verify the trained PGNN-ASCF photovoltaic forecasting model.

The script loads the released checkpoints, reconstructs the chronological test
set, and writes CSV files only. PG-GRU-ASCF is retained as a recurrent reference
model because it is used as a comparison in the manuscript.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PV_RATED_KW = 438.7
ROOT = Path(__file__).resolve().parent
SEQ_LEN = 16
BASE_FEATURES = [
    "ghi_wh_m2",
    "dni_wh_m2",
    "dhi_wh_m2",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "precipitation_mm",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
    "day_of_week",
    "is_weekend",
    "solar_elevation_deg",
    "solar_elevation_pos_sin",
    "is_daytime",
    "ghi_norm",
    "temp_derate",
    "pv_potential_kw",
    "pv_lag_1",
    "pv_lag_2",
    "pv_lag_4",
    "pv_lag_8",
    "pv_lag_96",
    "pv_roll_mean_4",
    "pv_roll_std_4",
    "pv_roll_mean_16",
    "pv_roll_std_16",
    "pv_ramp_1",
    "pv_ramp_4",
    "ghi_lag_1",
    "ghi_lag_4",
    "ghi_roll_mean_4",
    "ghi_roll_std_4",
    "day_type_enc",
]
SEQ_FEATURES = [
    "pv_power_kw",
    "ghi_wh_m2",
    "dni_wh_m2",
    "dhi_wh_m2",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "solar_elevation_deg",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "pv_potential_kw",
]


@dataclass
class TabularSplit:
    x_test: pd.DataFrame
    y_test: pd.Series
    idx_test: pd.DatetimeIndex


@dataclass
class SequenceSplit:
    x_scaled: np.ndarray
    x_raw: np.ndarray
    y_true: np.ndarray
    idx: pd.DatetimeIndex


class PGNN(nn.Module):
    def __init__(self, n_features: int, potential_idx: int) -> None:
        super().__init__()
        self.potential_idx = potential_idx
        self.correction = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.04),
            nn.Linear(32, 1),
        )

    def forward(self, x_scaled: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        potential = x_raw[:, self.potential_idx : self.potential_idx + 1].clamp(0.0, PV_RATED_KW)
        correction = torch.sigmoid(self.correction(x_scaled))
        return potential * correction


class PhysicsGuidedGRU(nn.Module):
    def __init__(self, n_features: int, *, use_envelope: bool = True) -> None:
        super().__init__()
        self.use_envelope = use_envelope
        self.rnn = nn.GRU(n_features, 56, batch_first=True)
        self.head = nn.Sequential(nn.Linear(56, 32), nn.ReLU(), nn.Linear(32, 1))
        self.potential_idx = SEQ_FEATURES.index("pv_potential_kw")

    def forward(self, x_scaled: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_scaled)
        z = self.head(out[:, -1, :])
        if not self.use_envelope:
            return z
        potential = x_raw[:, -1, self.potential_idx : self.potential_idx + 1].clamp(0.0, PV_RATED_KW)
        return potential * torch.sigmoid(z)


def build_features(df: pd.DataFrame, day_type_path: Path | None = None) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)
    out["solar_elevation_pos_sin"] = np.sin(np.radians(out["solar_elevation_deg"].clip(lower=0)))
    out["ghi_norm"] = out["ghi_wh_m2"] / max(float(out["ghi_wh_m2"].max()), 1.0)
    temp_excess = (out["temperature_c"] - 25.0).clip(lower=0)
    out["temp_derate"] = (1.0 - 0.004 * temp_excess).clip(lower=0.80, upper=1.05)
    out["pv_potential_kw"] = (
        PV_RATED_KW * out["ghi_norm"].clip(0, 1.2) * out["temp_derate"]
    ).clip(lower=0, upper=PV_RATED_KW)

    pv = out["pv_power_kw"]
    for lag in [1, 2, 4, 8, 96]:
        out[f"pv_lag_{lag}"] = pv.shift(lag)
    for window in [4, 16]:
        out[f"pv_roll_mean_{window}"] = pv.rolling(window, min_periods=1).mean().shift(1)
        out[f"pv_roll_std_{window}"] = pv.rolling(window, min_periods=2).std().shift(1)
    out["pv_ramp_1"] = pv.diff(1)
    out["pv_ramp_4"] = pv.diff(4)

    ghi = out["ghi_wh_m2"]
    for lag in [1, 4]:
        out[f"ghi_lag_{lag}"] = ghi.shift(lag)
    out["ghi_roll_mean_4"] = ghi.rolling(4, min_periods=1).mean().shift(1)
    out["ghi_roll_std_4"] = ghi.rolling(4, min_periods=2).std().shift(1)

    if day_type_path is not None and day_type_path.exists():
        daily = pd.read_csv(day_type_path, index_col=0, parse_dates=True)
        day_map = {idx.date(): value for idx, value in daily["day_type"].items()}
        dtype_enc = {"clear": 0, "partly_variable": 1, "high_variability": 2, "night": 3}
        out["day_type_enc"] = [dtype_enc.get(day_map.get(ts.date(), "night"), 3) for ts in out.index]
    else:
        out["day_type_enc"] = 3
    return out


def make_tabular_test_split(features: pd.DataFrame, horizon: int) -> TabularSplit:
    work = features.copy()
    work["target"] = work["pv_power_kw"].shift(-horizon)
    work = work.dropna(subset=BASE_FEATURES + ["target"])
    test = work.loc["2025-01-01":]
    return TabularSplit(test[BASE_FEATURES], test["target"], test.index)


def make_sequence_test_split(features: pd.DataFrame, horizon: int, scaler) -> SequenceSplit:
    work = features[SEQ_FEATURES].dropna(subset=SEQ_FEATURES).copy()
    target = work["pv_power_kw"].shift(-horizon)
    valid_end = len(work) - horizon
    raw_values = work[SEQ_FEATURES].iloc[:valid_end].values.astype(np.float32)
    target_values = target.iloc[:valid_end].values.astype(np.float32)
    input_index = work.index[:valid_end]
    scaled_values = scaler.transform(raw_values).astype(np.float32)

    x_scaled, x_raw, y_true, idx = [], [], [], []
    for end_pos in range(SEQ_LEN - 1, valid_end):
        timestamp = input_index[end_pos]
        if timestamp < pd.Timestamp("2025-01-01"):
            continue
        x_scaled.append(scaled_values[end_pos - SEQ_LEN + 1 : end_pos + 1])
        x_raw.append(raw_values[end_pos - SEQ_LEN + 1 : end_pos + 1])
        y_true.append(target_values[end_pos])
        idx.append(timestamp)
    return SequenceSplit(
        np.stack(x_scaled).astype(np.float32),
        np.stack(x_raw).astype(np.float32),
        np.asarray(y_true, dtype=np.float32),
        pd.DatetimeIndex(idx),
    )


def state_windows_for_indices(features: pd.DataFrame, indices: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    work = features[SEQ_FEATURES].dropna(subset=SEQ_FEATURES).copy()
    valid_index = work.index[: len(work) - horizon]
    values = work[SEQ_FEATURES].iloc[: len(work) - horizon].values.astype(np.float32)
    pos_by_time = {ts: pos for pos, ts in enumerate(valid_index)}
    windows = []
    for ts in indices:
        pos = pos_by_time[ts]
        windows.append(values[pos - SEQ_LEN + 1 : pos + 1])
    return np.stack(windows).astype(np.float32)


def predict_pgnn(model: PGNN, scaler, x_test: pd.DataFrame) -> np.ndarray:
    model.eval()
    x_raw = x_test.values.astype(np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)
    with torch.no_grad():
        pred = model(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(x_raw, dtype=torch.float32),
        ).numpy().ravel()
    return np.clip(pred, 0.0, PV_RATED_KW * 1.15)


def predict_pg_gru(model: PhysicsGuidedGRU, split: SequenceSplit) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(split.x_scaled, dtype=torch.float32),
            torch.tensor(split.x_raw, dtype=torch.float32),
        ).numpy().ravel()
    return np.clip(pred, 0.0, PV_RATED_KW * 1.15)


def adaptive_physical_state_correction(y_raw: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    pv_idx = SEQ_FEATURES.index("pv_power_kw")
    ghi_idx = SEQ_FEATURES.index("ghi_wh_m2")
    potential_idx = SEQ_FEATURES.index("pv_potential_kw")
    is_day_idx = SEQ_FEATURES.index("is_daytime")
    pv_now = x_raw[:, -1, pv_idx]
    ghi_now = x_raw[:, -1, ghi_idx]
    ghi_prev = x_raw[:, -5, ghi_idx]
    pv_prev = x_raw[:, -5, pv_idx]
    potential = x_raw[:, -1, potential_idx]
    is_day = x_raw[:, -1, is_day_idx] > 0.5

    ghi_innovation = np.abs(ghi_now - ghi_prev) / max(float(np.nanmax(x_raw[:, :, ghi_idx])), 1.0)
    pv_innovation = np.abs(pv_now - pv_prev) / PV_RATED_KW
    stability = np.exp(-3.0 * ghi_innovation - 1.5 * pv_innovation)
    adaptive_gain = np.clip(0.05 + 0.45 * stability, 0.05, 0.85)
    corrected = adaptive_gain * y_raw + (1.0 - adaptive_gain) * pv_now
    envelope_margin = np.clip(potential + 0.08 * PV_RATED_KW, 0.0, PV_RATED_KW)
    corrected = np.minimum(corrected, envelope_margin)
    corrected = np.where(is_day, corrected, 0.0)
    return np.clip(corrected, 0.0, PV_RATED_KW * 1.15)


def state_consistent_fusion(
    y_structural: np.ndarray,
    y_state: np.ndarray,
    x_raw: np.ndarray,
    weight_state: float,
) -> np.ndarray:
    potential_idx = SEQ_FEATURES.index("pv_potential_kw")
    is_day_idx = SEQ_FEATURES.index("is_daytime")
    fused = weight_state * y_state + (1.0 - weight_state) * y_structural
    potential = x_raw[:, -1, potential_idx]
    is_day = x_raw[:, -1, is_day_idx] > 0.5
    envelope_margin = np.clip(potential + 0.08 * PV_RATED_KW, 0.0, PV_RATED_KW)
    fused = np.minimum(fused, envelope_margin)
    fused = np.where(is_day, fused, 0.0)
    return np.clip(fused, 0.0, PV_RATED_KW * 1.15)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, model: str) -> dict[str, float | int | str]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "horizon_steps": horizon,
        "horizon_min": horizon * 15,
        "model": model,
        "n": int(len(y_true)),
        "mae_kw": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kw": rmse,
        "nrmse": rmse / PV_RATED_KW,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def verify_horizon(features: pd.DataFrame, model_dir: Path, output_dir: Path, horizon: int) -> list[dict[str, float | int | str]]:
    tabular = make_tabular_test_split(features, horizon)
    pgnn_scaler = joblib.load(model_dir / f"pgnn_scaler_h{horizon}.joblib")
    pgnn = PGNN(len(BASE_FEATURES), BASE_FEATURES.index("pv_potential_kw"))
    pgnn.load_state_dict(torch.load(model_dir / f"pgnn_h{horizon}.pt", map_location="cpu"))
    y_pgnn = predict_pgnn(pgnn, pgnn_scaler, tabular.x_test)
    pgnn_state_raw = state_windows_for_indices(features, tabular.idx_test, horizon)
    y_pgnn_apsc = adaptive_physical_state_correction(y_pgnn, pgnn_state_raw)
    pgnn_weight = json.loads((model_dir / f"pgnn_ascf_weight_h{horizon}.json").read_text(encoding="utf-8"))
    y_pgnn_ascf = state_consistent_fusion(y_pgnn, y_pgnn_apsc, pgnn_state_raw, float(pgnn_weight["weight_state"]))

    sequence_scaler = joblib.load(model_dir / f"sequence_scaler_h{horizon}.joblib")
    seq = make_sequence_test_split(features, horizon, sequence_scaler)
    structural_model = PhysicsGuidedGRU(seq.x_scaled.shape[2], use_envelope=False)
    state_model = PhysicsGuidedGRU(seq.x_scaled.shape[2], use_envelope=True)
    structural_model.load_state_dict(torch.load(model_dir / f"pg_gru_no_envelope_h{horizon}.pt", map_location="cpu"))
    state_model.load_state_dict(torch.load(model_dir / f"pg_gru_h{horizon}.pt", map_location="cpu"))
    y_pg_gru_structural = predict_pg_gru(structural_model, seq)
    y_pg_gru_raw = predict_pg_gru(state_model, seq)
    y_pg_gru_apsc = adaptive_physical_state_correction(y_pg_gru_raw, seq.x_raw)
    pg_gru_weight = json.loads((model_dir / f"pg_gru_ascf_weight_h{horizon}.json").read_text(encoding="utf-8"))
    y_pg_gru_ascf = state_consistent_fusion(
        y_pg_gru_structural,
        y_pg_gru_apsc,
        seq.x_raw,
        float(pg_gru_weight["weight_state"]),
    )

    pred = pd.DataFrame(
        {
            "timestamp": tabular.idx_test,
            "target_timestamp": tabular.idx_test + pd.to_timedelta(horizon * 15, unit="min"),
            "y_true_kw": tabular.y_test.values,
            "pgnn_kw": y_pgnn,
            "pgnn_apsc_kw": y_pgnn_apsc,
            "pgnn_ascf_kw": y_pgnn_ascf,
            "pg_gru_ascf_reference_kw": y_pg_gru_ascf,
        }
    )
    pred.to_csv(output_dir / f"pgnn_ascf_predictions_h{horizon}.csv", index=False)
    return [
        metric_row(tabular.y_test.values, y_pgnn, horizon, "PGNN"),
        metric_row(tabular.y_test.values, y_pgnn_apsc, horizon, "PGNN-APSC"),
        metric_row(tabular.y_test.values, y_pgnn_ascf, horizon, "PGNN-ASCF"),
        metric_row(seq.y_true, y_pg_gru_ascf, horizon, "PG-GRU-ASCF reference"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify trained PGNN-ASCF checkpoints and write CSV outputs.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "merged_dataset_15min.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    args = parser.parse_args()

    torch.set_num_threads(1)
    args.outputs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    features = build_features(df, args.data.parent / "daily_classification.csv")
    rows = []
    for horizon in args.horizons:
        rows.extend(verify_horizon(features, args.models, args.outputs, horizon))
    pd.DataFrame(rows).to_csv(args.outputs / "model_comparison_metrics.csv", index=False)
    print(f"Wrote CSV outputs to {args.outputs.resolve()}")


if __name__ == "__main__":
    main()
