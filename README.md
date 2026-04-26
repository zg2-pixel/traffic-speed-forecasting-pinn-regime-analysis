# Traffic Speed Forecasting with Physics-Informed Neural Networks: A Regime-Dependent Analysis

A course project for CMU 12-787 (Physics-Informed Machine Learning) investigating when and where physics-informed regularization actually helps a GRU-based traffic speed forecaster, and when it does not.

**Authors:** Zhaoyang Guo, Junchi Fan

## Thesis

Physics-informed losses are often presented as a universal improvement over pure data-driven models. Across four data regimes (dense standard-horizon, event-focused, long-horizon, sparse-sensor), we find that the benefit is strongly regime-dependent: soft physics penalties can be neutral or mildly harmful under abundant data, while carefully chosen regime-specific constraints provide substantial gains where data is thin. The practical takeaway: the form of physics injection must match the data regime, not the other way around.

## Dataset

PeMS 5-minute station data, I-880 Southbound (San Jose -> Fremont), January 2025. After quality filtering, 8 detectors are retained. Hourly weather is pulled from Open-Meteo and joined on the hour. See `preprocessing/raw_cleaning.py` for exact filter thresholds.

The raw PeMS `.txt.gz` files are **not** included in this repository (too large, and subject to PeMS terms of use). The **processed CSVs are included** (`data/processed/`) so the model scripts can be run without going back to PeMS. If you want to regenerate the processed data from scratch (for example, to tune the preprocessing thresholds), download the raw files from the Caltrans PeMS portal and place them in `data/raw/`, then re-run the preprocessing step below.

## Repository Layout

```
.
├── preprocessing/
│   ├── raw_cleaning.py
│   └── event_labeling.py
│
├── baseline/
│   ├── v1_simple.py
│   ├── v2_refactor.py
│   ├── v3_attention.py
│   ├── v4_final.py
│   ├── v5_residual.py
│   └── v6_event_focused.py
│
├── physics/
│   ├── v1_flow_penalty.py
│   ├── v2_occupancy_penalty.py
│   ├── v3_greenshields_fd.py
│   ├── v4_perl.py
│   └── v5_event_focused.py
│
├── regime_study/
│   ├── long_horizon_baseline.py
│   ├── long_horizon_physics.py
│   ├── sparse_sensor_baseline.py
│   └── sparse_sensor_physics.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── traffic_clean.csv
│       ├── weather.csv
│       ├── all_splits_with_regime_labels.csv
│       └── event_labels/
│
├── requirements.txt
├── LICENSE
└── README.md
```

Output folders (`baseline_output/`, `physics_output/`, `regime_study_output/`) are created automatically by each script and gitignored. Each experiment writes `metrics.txt`, a set of PNG plots, and (for the reference seed) `best_model.pt`.

## Installation

```bash
git clone https://github.com/zg2-pixel/traffic-pinn-regime-analysis.git
cd traffic-pinn-regime-analysis
python -m venv venv
source venv/bin/activate            # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.x. A GPU is helpful but not required; all scripts auto-detect CUDA, Apple MPS, or CPU.

## Data Setup

The processed CSVs are already in `data/processed/`, so you can skip directly to "Running the Experiments" below.

**Optional** — regenerate the processed data from raw PeMS files:

1. Download PeMS 5-minute station data for District 4 (Bay Area), January 2025, from the [Caltrans PeMS portal](https://pems.dot.ca.gov/).
2. Place all `d04_text_station_5min_2025_01_*.txt.gz` files in `data/raw/`.
3. Run preprocessing (takes ~1-2 min):
   ```bash
   python preprocessing/raw_cleaning.py
   python preprocessing/event_labeling.py
   ```
   The second script also downloads hourly weather from Open-Meteo and writes `data/processed/all_splits_with_regime_labels.csv`, which every model script reads.

## Running the Experiments

Each script is self-contained. Train times are roughly 5-20 minutes per seed on a laptop GPU, longer on CPU.

**Dense 30-min horizon** (main baseline and its physics variants):
```bash
python baseline/v4_final.py
python physics/v1_flow_penalty.py
python physics/v2_occupancy_penalty.py
python physics/v3_greenshields_fd.py
python physics/v4_perl.py
```

**Event-focused regime** (predicting during congestion onset / peak / recovery):
```bash
python baseline/v6_event_focused.py
python physics/v5_event_focused.py
```

**Long-horizon regime** (120-min forecast):
```bash
python regime_study/long_horizon_baseline.py
python regime_study/long_horizon_physics.py
```

**Sparse-sensor regime** (train on 3 detectors, test on 5 unseen):
```bash
python regime_study/sparse_sensor_baseline.py
python regime_study/sparse_sensor_physics.py
```

Results for each run are written to a matching `*_output/` folder with a `metrics.txt` summary and diagnostic plots.

## Key Findings (Summary)

Full results and analysis are in the project report. In brief:

- **Dense 30-min regime**: physics penalties (v1, v2, v3) are neutral or slightly worse than the tuned baseline. PERL (v4) improves overall RMSE, but a LastSpeed residual baseline matches it without any physics, indicating that the gain comes from the residual formulation rather than the Greenshields term.
- **Event-focused regime**: phase-aware losses (v5) improve congested-phase RMSE substantially over the v6 data-only baseline. This is the clearest case where physics, tailored to the regime, helps.
- **Long-horizon and sparse-sensor regimes**: the `regime_study/` experiments test whether the phase-aware design generalizes to other "data-thin" settings. Numbers are in `regime_study_output/*/metrics.txt` after running the four scripts.

## License

MIT License. See `LICENSE` for the full text.

## Citation

If you reference this codebase, please cite as:

```
Guo, Z., and Fan, J. (2025). Traffic Speed Forecasting with Physics-Informed
Neural Networks: A Regime-Dependent Analysis. Course project, CMU 12-787.
https://github.com/zg2-pixel/traffic-pinn-regime-analysis
```
