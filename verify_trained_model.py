"""Verify the trained PG-GRU-ASCF photovoltaic forecasting model.

The script loads the released checkpoints, reconstructs the 15-minute test
sequences, and writes CSV files only. No figures are generated.
"""
from __future__ import annotations

import argparse
import json
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)
    out["ghi_norm"] = out["ghi_wh_m2"] / max(float(out["ghi_wh_m2"].max()), 1.0)
    temp_excess = (out["temperature_c"] - 25.0).clip(lower=0)
    out["temp_derate"] = (1.0 - 0.004 * temp_excess).clip(lower=0.80, upper=1.05)
    out["pv_potential_kw"] = (
        PV_RATED_KW * out["ghi_norm"].clip(0, 1.2) * out["temp_derate"]
    ).clip(lower=0, upper=PV_RATED_KW)
    return out


def make_test_sequences(features: pd.DataFrame, horizon: int, scaler) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    work = features[SEQ_FEATURES].dropna(subset=SEQ_FEATURES).copy()
    target = work["pv_power_kw"].shift(-horizon)
    valid_end = len(work) - horizon
    raw_values = work[SEQ_FEATURES].iloc[:valid_end].values.astype(np.float32)
    target_values = target.iloc[:valid_end].values.astype(np.float32)
    input_index = work.index[:valid_end]
    scaled_values = scaler.transform(raw_values).astype(np.float32)

    x_scaled, x_raw, y, idx = [], [], [], []
    for end_pos in range(SEQ_LEN - 1, valid_end):
        timestamp = input_index[end_pos]
        if timestamp < pd.Timestamp("2025-01-01"):
            continue
        x_scaled.append(scaled_values[end_pos - SEQ_LEN + 1 : end_pos + 1])
        x_raw.append(raw_values[end_pos - SEQ_LEN + 1 : end_pos + 1])
        y.append(target_values[end_pos])
        idx.append(timestamp)
    return (
        np.stack(x_scaled).astype(np.float32),
        np.stack(x_raw).astype(np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DatetimeIndex(idx),
    )


def load_pg_gru(model_dir: Path, horizon: int, model_name: str, n_features: int) -> PhysicsGuidedGRU:
    model = PhysicsGuidedGRU(n_features, use_envelope=model_name != "pg_gru_no_envelope")
    state = torch.load(model_dir / f"{model_name}_h{horizon}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_pg_gru(model: PhysicsGuidedGRU, x_scaled: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = model(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(x_raw, dtype=torch.float32),
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


def verify_horizon(data_path: Path, model_dir: Path, output_dir: Path, horizon: int) -> dict[str, float | int | str]:
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    features = build_features(df)
    scaler = joblib.load(model_dir / f"sequence_scaler_h{horizon}.joblib")
    x_scaled, x_raw, y_true, idx = make_test_sequences(features, horizon, scaler)

    structural_model = load_pg_gru(model_dir, horizon, "pg_gru_no_envelope", x_scaled.shape[2])
    state_model = load_pg_gru(model_dir, horizon, "pg_gru", x_scaled.shape[2])
    structural = predict_pg_gru(structural_model, x_scaled, x_raw)
    state_raw = predict_pg_gru(state_model, x_scaled, x_raw)
    apsc = adaptive_physical_state_correction(state_raw, x_raw)
    weight_info = json.loads((model_dir / f"pg_gru_ascf_weight_h{horizon}.json").read_text(encoding="utf-8"))
    ascf = state_consistent_fusion(structural, apsc, x_raw, float(weight_info["weight_state"]))

    pred = pd.DataFrame(
        {
            "timestamp": idx,
            "target_timestamp": idx + pd.to_timedelta(horizon * 15, unit="min"),
            "y_true_kw": y_true,
            "pg_gru_no_envelope_kw": structural,
            "pg_gru_raw_kw": state_raw,
            "pg_gru_apsc_kw": apsc,
            "pg_gru_ascf_kw": ascf,
        }
    )
    pred.to_csv(output_dir / f"pg_gru_ascf_predictions_h{horizon}.csv", index=False)
    return metric_row(y_true, ascf, horizon, "PG-GRU-ASCF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify trained PG-GRU-ASCF checkpoints and write CSV outputs.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "merged_dataset_15min.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    args = parser.parse_args()

    torch.set_num_threads(1)
    args.outputs.mkdir(parents=True, exist_ok=True)
    metrics = [verify_horizon(args.data, args.models, args.outputs, horizon) for horizon in args.horizons]
    pd.DataFrame(metrics).to_csv(args.outputs / "pg_gru_ascf_metrics.csv", index=False)
    print(f"Wrote CSV outputs to {args.outputs.resolve()}")


if __name__ == "__main__":
    main()
