# Article Submission Code

This directory contains a minimal, self-contained verification package for the
trained physics-guided PV forecasting model used in the manuscript.

It is intentionally limited to model verification and CSV output. Plotting,
figure generation, exploratory analysis, and manuscript files are excluded.

## Contents

- `data/merged_dataset_15min.csv`: processed 15-minute PV and weather dataset.
- `models/`: trained PG-GRU-ASCF checkpoints and preprocessing/fusion artifacts.
- `verify_trained_model.py`: loads the trained models and writes CSV results.
- `outputs/`: generated prediction and metric CSV files.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python verify_trained_model.py
```

The script verifies both horizons used in the article:

- `h1`: 15-minute-ahead forecasting.
- `h4`: 60-minute-ahead forecasting.

CSV outputs are written to `outputs/`:

- `pg_gru_ascf_predictions_h1.csv`
- `pg_gru_ascf_predictions_h4.csv`
- `pg_gru_ascf_metrics.csv`

To verify only one horizon:

```bash
python verify_trained_model.py --horizons 1
python verify_trained_model.py --horizons 4
```

## Notes

The test period starts on 2025-01-01. Each prediction CSV includes the input
timestamp, forecast target timestamp, observed PV power, intermediate model
components, and final PG-GRU-ASCF prediction.
