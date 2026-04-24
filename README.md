# Traffic Speed Forecasting with Physics-Informed Neural Networks: A Regime-Dependent Analysis

A course project for CMU 12-787 (Physics-Informed Machine Learning) investigating when and where physics-informed regularization actually helps a GRU-based traffic speed forecaster, and when it does not.

**Authors:** Zhaoyang Guo, Junchi Fan

## Thesis

Physics-informed losses are often presented as a universal improvement over pure data-driven models. Across four data regimes (dense standard-horizon, event-focused, long-horizon, sparse-sensor), we find that the benefit is strongly regime-dependent: soft physics penalties can be neutral or mildly harmful under abundant data, while carefully chosen regime-specific constraints provide substantial gains where data is thin. The practical takeaway: the form of physics injection must match the data regime, not the other way around.

## Dataset

PeMS 5-minute station data, I-880 Southbound (San Jose -> Fremont), January 2025. After quality filtering, 8 detectors are retained. Hourly weather is pulled from Open-Meteo and joined on the hour. See `preprocessing/raw_cleaning.py` for exact filter thresholds.

The raw PeMS files are not included in this repository (too large, and subject to PeMS terms of use). They can be downloaded from the Caltrans PeMS portal and placed in `data/raw/`.

## Repository Layout

```
.
├── preprocessing/                      # Stage 1: clean raw data, label events
│   ├── raw_cleaning.py                 # PeMS txt.gz -> traffic_clean.csv
│   └── event_labeling.py               # traffic_clean.csv -> all_splits_with_regime_labels.csv
│                                       #   (adds split column and phase labels)
│
├── baseline/                           # Pure data-driven models (no physics)
│   ├── v1_simple.py                    # plain GRU, last-hidden head
│   ├── v2_refactor.py                  # + early stopping + deepcopy
│   ├── v3_attention.py                 # + temporal attention + LR scheduler
│   ├── v4_final.py                     # tuned, main dense baseline
│   ├── v5_residual.py                  # residual learning (LastSpeed), 5 seeds
│   └── v6_event_focused.py             # event-regime baseline, 2 features, 5 seeds
│
├── physics/                            # Physics-informed variants
│   ├── v1_flow_penalty.py              # soft penalties: acceleration + speed range + flow>450 -> v<=50
│   ├── v2_occupancy_penalty.py         # same framework, occupancy indicator replaces flow
│   ├── v3_greenshields_fd.py           # Greenshields fundamental diagram penalty
│   ├── v4_perl.py                      # PERL: v_hat = v_phys(occ) + GRU_residual
│   └── v5_event_focused.py             # 4 phase-aware losses (trend, onset, depth, recovery)
│
├── regime_study/                       # Cross-regime comparisons (baseline vs physics pairs)
│   ├── long_horizon_baseline.py        # 120-min forecast, pure data
│   ├── long_horizon_physics.py         # 120-min forecast, phase-aware physics
│   ├── sparse_sensor_baseline.py       # train on 3 detectors, test on 5 held-out
│   └── sparse_sensor_physics.py        # same split + phase-aware physics
│
├── data/
│   └── raw/                            # Put PeMS *.txt.gz files here (not in repo)
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
