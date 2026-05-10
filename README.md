# PGNN-ASCF PV Forecasting

This repository provides the companion code package for the manuscript
**"Physics Guided Neural Forecasting with Adaptive State-Consistent Fusion for
Real-Time Photovoltaic Power Prediction"**.

The main model in the article is **PGNN-ASCF**, a compact physics-guided neural
network with adaptive state-consistent fusion for real-time photovoltaic (PV)
power forecasting. The repository is prepared as a focused verification package:
it loads the trained models, reconstructs the chronological test set, and writes
CSV prediction and metric outputs. Plotting scripts, exploratory notebooks,
figure generation, and manuscript source files are intentionally excluded.

Author: Shirong Guo  
Contact: shirong.guo@monash.edu

## Model Summary

PGNN-ASCF combines four elements:

- A physical PV potential envelope based on irradiance, temperature, and rated
  PV capacity.
- A compact feedforward PGNN structural forecaster that predicts a bounded
  correction around the physical envelope.
- Adaptive physical state correction (APSC), which uses recent PV and irradiance
  changes to form a deterministic state-updated candidate.
- Adaptive state-consistent fusion (ASCF), which selects the final fusion weight
  on the validation set without using test data.

The manuscript also compares PGNN-ASCF with a higher-latency recurrent reference,
**PG-GRU-ASCF**. This repository therefore retains the PG-GRU-ASCF checkpoints
only as a comparison model; PGNN-ASCF is the primary released model.

## Repository Contents

- `data/merged_dataset_15min.csv`: processed 15-minute PV, weather, and temporal
  source data.
- `data/daily_classification.csv`: daily regime labels used to reconstruct the
  PGNN feature matrix.
- `models/pgnn_*`: trained PGNN checkpoints, scalers, and ASCF fusion weights.
- `models/pg_gru_*` and `models/sequence_scaler_*`: PG-GRU-ASCF reference
  artifacts used for manuscript comparison.
- `verify_trained_model.py`: verification script that writes CSV outputs only.
- `outputs/`: generated prediction and metric CSV files.
- `requirements.txt`: minimal Python dependencies.

## Environment

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python verify_trained_model.py
```

The default run verifies both article horizons:

- `h1`: 15-minute-ahead PV power forecasting.
- `h4`: 60-minute-ahead PV power forecasting.

To verify only one horizon:

```bash
python verify_trained_model.py --horizons 1
python verify_trained_model.py --horizons 4
```

## CSV Outputs

The script writes:

- `outputs/pgnn_ascf_predictions_h1.csv`
- `outputs/pgnn_ascf_predictions_h4.csv`
- `outputs/model_comparison_metrics.csv`

Each prediction CSV contains the input timestamp, forecast target timestamp,
observed PV power, PGNN structural prediction, PGNN-APSC prediction, final
PGNN-ASCF prediction, and the PG-GRU-ASCF reference prediction.

The metric CSV reports MAE, RMSE, normalized RMSE, and R2 for PGNN, PGNN-APSC,
PGNN-ASCF, and the PG-GRU-ASCF reference at each horizon.

## Notes

The chronological test period starts on 2025-01-01. The package is intended to
support verification of the trained article models and the CSV results, not to
retrain models or regenerate figures.

For questions about the repository or manuscript code, please contact Shirong
Guo at shirong.guo@monash.edu.
