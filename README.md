# Physics-Informed GRU for Short-Term Traffic Speed Forecasting

CEE 12-787 Physics-Informed Machine Learning — CMU Spring 2026

**Authors:** Zhaoyang Guo, Junchi Fan

## Overview

This project develops a physics-informed GRU model for 30-minute traffic speed forecasting on I-880 Southbound (San Francisco Bay Area). Three physics-based constraints (acceleration limits, speed bounds, congestion-speed consistency) are incorporated as soft penalties in the training loss.

**Key results:**

| Model | RMSE (mph) | MAE (mph) |
|-------|-----------|----------|
| Baseline GRU | 4.10 | 2.26 |
| Physics v1 (no warmup) | 4.69 | 3.09 |
| Physics v2 (warmup) | **3.98** | **2.15** |

## Setup

1. Install dependencies:
```
pip install torch pandas numpy matplotlib scikit-learn
```

2. Processed data (`data/processed/traffic_clean.csv`) is included in this repo. If you want to reproduce from raw data:
   - Register at https://pems.dot.ca.gov (free)
   - Download: Data Clearinghouse → District 4, Station 5-Minute, January 2025 (31 files)
   - Download: Station Metadata → `d04_text_meta_2025_01_15.txt`
   - Place all files in `data/raw/`
   - Run `python step1_preprocess.py`

## Run

```
python step2_train_baseline.py    # Baseline GRU → outputs_baseline/
python step3_physics_v1.py        # Physics v1 (fixed weights) → outputs_physics_v1/
python step3_physics_v2.py        # Physics v2 (warmup) → outputs_physics_v2/
```

Each script reads from `data/processed/traffic_clean.csv` and saves results (loss curves, prediction plots, metrics) to its output folder.

## Project Structure

```
├── data/
│   └── processed/
│       └── traffic_clean.csv        # Cleaned traffic + weather data (included)
├── step1_preprocess.py              # PeMS parsing + weather download + feature engineering
├── step2_train_baseline.py          # Baseline GRU (MSE loss only)
├── step3_physics_v1.py              # Physics-informed v1 (fixed λ, no warmup)
├── step3_physics_v2.py              # Physics-informed v2 (reduced λ_cong + warmup)
└── README.md
```

## Data

- **Traffic**: Caltrans PeMS District 4, January 2025, 5-minute intervals
- **Stations**: 10 mainline detectors on I-880 Southbound (PM 5.2–17.3)
- **Weather**: Open-Meteo Historical API (lat 37.46, lon -121.94)
- **Split**: 22 days train / 3 days validation / 6 days test (chronological)
- **Features**: avg_speed, total_flow, avg_occupancy, hour_sin/cos, dow_sin/cos, is_weekend, temperature, precipitation, wind_speed, weather_code

## Model

- **Architecture**: 2-layer GRU, hidden size 64, dropout 0.2
- **Input**: 12 time steps (60 min) × 12 features
- **Output**: 6 time steps (30 min) of avg_speed
- **Physics constraints** (v2):
  - Acceleration limit: |Δv| ≤ 15 mph per 5-min step
  - Speed bounds: 0 ≤ v ≤ 85 mph
  - Congestion-speed consistency: if flow > 180 veh/5min, speed ≤ 50 mph
  - Warmup: epochs 0–10 data only, epochs 10–20 linear ramp-up of physics weights


