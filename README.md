# PG-GRU-ASCF PV Forecasting

This repository provides the companion code package for the manuscript on
physics-guided photovoltaic (PV) power forecasting. It is prepared as a compact
verification repository for reproducing the CSV results from the trained
PG-GRU-ASCF model used in the article.

Author: Shirong Guo  
Contact: shirong.guo@monash.edu

The repository is intentionally limited to model verification and CSV output.
Plotting scripts, figure generation, exploratory analysis, and manuscript source
files are not included.

## Overview

The released code verifies a physics-guided gated recurrent unit model with
adaptive state-consistent fusion (PG-GRU-ASCF) for short-term PV power
forecasting. The model combines sequence learning with physically informed
constraints and state correction so that predictions remain consistent with
irradiance, daytime state, and PV generation capacity.

Two forecasting horizons are included:

- `h1`: 15-minute-ahead PV power forecasting.
- `h4`: 60-minute-ahead PV power forecasting.

The verification workflow loads the trained checkpoints, reconstructs the test
sequences from the processed 15-minute dataset, applies the PG-GRU-ASCF fusion
step, and exports prediction and metric tables as CSV files.

## Contents

- `data/merged_dataset_15min.csv`: processed 15-minute PV, weather, and temporal
  feature source data.
- `models/`: trained PG-GRU checkpoints, sequence scalers, and ASCF fusion
  weights for both horizons.
- `verify_trained_model.py`: verification script that loads the released models
  and writes CSV results.
- `outputs/`: generated prediction and metric CSV files.
- `requirements.txt`: minimal Python dependencies.

## Environment

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python verify_trained_model.py
```

CSV outputs are written to `outputs/`:

- `pg_gru_ascf_predictions_h1.csv`
- `pg_gru_ascf_predictions_h4.csv`
- `pg_gru_ascf_metrics.csv`

To verify only one horizon:

```bash
python verify_trained_model.py --horizons 1
python verify_trained_model.py --horizons 4
```

## Output Description

Each prediction CSV contains:

- `timestamp`: input timestamp.
- `target_timestamp`: forecast target timestamp.
- `y_true_kw`: observed PV power.
- `pg_gru_no_envelope_kw`: structural sequence model prediction.
- `pg_gru_raw_kw`: physics-guided GRU prediction before state correction.
- `pg_gru_apsc_kw`: adaptive physical state correction output.
- `pg_gru_ascf_kw`: final PG-GRU-ASCF prediction.

The metric CSV reports MAE, RMSE, normalized RMSE, and R2 for each forecasting
horizon.

## Notes

The test period starts on 2025-01-01. Each prediction CSV includes the input
timestamp, forecast target timestamp, observed PV power, intermediate model
components, and final PG-GRU-ASCF prediction.

For questions about the repository or manuscript code, please contact Shirong
Guo at shirong.guo@monash.edu.
