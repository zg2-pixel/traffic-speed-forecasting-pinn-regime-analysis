"""
sparse_sensor_baseline.py

Sparse-sensor regime baseline: pure data-driven, spatial generalization.
Uses only 3 detectors for training and validation; the remaining 5
unseen detectors are held out for testing. Scaler is fit on the 3
training stations only, so the test set is genuinely out-of-sample
in the spatial dimension. Paired with sparse_sensor_physics.py.
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Config
LABEL_CSV_CANDIDATES = [
    "data/processed/all_splits_with_regime_labels.csv",
    "all_splits_with_regime_labels.csv",
]
OUT_DIR = "regime_study_output/sparse_sensor_baseline_output"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 52, 62, 72, 82]
REF_SEED = 42

INPUT_WINDOW = 3
OUTPUT_WINDOW = 6
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
PATIENCE = 12
MIN_DELTA = 1e-4

FEATURE_COLS = ['avg_speed', 'avg_occupancy']
TARGET_COL = 'avg_speed'
SPEED_IDX = 0

# Sparse-sensor setup: 3 detectors for train/val, rest held out for test
N_TRAIN_STATIONS = 3

PHASE_TO_ID = {'other': 0, 'onset': 1, 'congested': 2, 'recovery': 3}
ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}
PHASE_COLORS = {'other': '#9AA0A6', 'onset': '#FFD54F',
                'congested': '#EF5350', 'recovery': '#81C784'}

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_label_csv():
    for p in LABEL_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"CSV not found: {LABEL_CSV_CANDIDATES}")


class PhaseDataset(Dataset):
    def __init__(self, features, targets, phases, input_window, output_window):
        self.X, self.y, self.ph = [], [], []
        n = len(features)
        for i in range(n - input_window - output_window + 1):
            self.X.append(features[i:i + input_window])
            self.y.append(targets[i + input_window:i + input_window + output_window])
            self.ph.append(phases[i + input_window:i + input_window + output_window])
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))
        self.ph = torch.LongTensor(np.array(self.ph))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ph[idx]


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, h):
        w = torch.softmax(self.score(h), dim=1)
        ctx = (h * w).sum(dim=1)
        return ctx, w.squeeze(-1)


class GRUAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_window, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_window),
        )

    def forward(self, x):
        h_all, _ = self.gru(x)
        ctx, attn_w = self.attention(h_all)
        return self.decoder(ctx), attn_w


def prepare_data():
    path = resolve_label_csv()
    print(f"Loading: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)
    required = ['timestamp', 'station', 'avg_speed', 'avg_occupancy', 'split', 'event_phase']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df['phase_id'] = df['event_phase'].map(PHASE_TO_ID).fillna(0).astype(int)

    all_stations = sorted(df['station'].unique())
    if len(all_stations) < N_TRAIN_STATIONS + 1:
        raise ValueError(f"Need at least {N_TRAIN_STATIONS+1} stations, "
                         f"got {len(all_stations)}")
    train_stations = all_stations[:N_TRAIN_STATIONS]
    test_stations = all_stations[N_TRAIN_STATIONS:]
    print(f"Train/Val stations ({len(train_stations)}): {train_stations}")
    print(f"Test stations     ({len(test_stations)}): {test_stations}")

    # Fit scaler on training stations' train-split only
    train_fit_mask = (df['station'].isin(train_stations)) & (df['split'] == 'train')
    scaler = StandardScaler()
    scaler.fit(df.loc[train_fit_mask, FEATURE_COLS])
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    train_list, val_list, test_list = [], [], []

    # Train + Val: from training stations only
    for sid in train_stations:
        sdf = df_scaled[df_scaled['station'] == sid].reset_index(drop=True)
        feat = sdf[FEATURE_COLS].values
        tgt = sdf[TARGET_COL].values
        ph = sdf['phase_id'].values
        for split_name, lst in [('train', train_list), ('val', val_list)]:
            sel = (sdf['split'] == split_name).values
            f, t, p = feat[sel], tgt[sel], ph[sel]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                lst.append(PhaseDataset(f, t, p, INPUT_WINDOW, OUTPUT_WINDOW))

    # Test: from held-out stations, use their test split only
    for sid in test_stations:
        sdf = df_scaled[df_scaled['station'] == sid].reset_index(drop=True)
        feat = sdf[FEATURE_COLS].values
        tgt = sdf[TARGET_COL].values
        ph = sdf['phase_id'].values
        sel = (sdf['split'] == 'test').values
        f, t, p = feat[sel], tgt[sel], ph[sel]
        if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
            test_list.append(PhaseDataset(f, t, p, INPUT_WINDOW, OUTPUT_WINDOW))

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)
    test_ds = ConcatDataset(test_list)
    print(f"Samples — train: {len(train_ds):,}  val: {len(val_ds):,}  "
          f"test: {len(test_ds):,} (on held-out stations)")

    speed_mean = float(scaler.mean_[SPEED_IDX])
    speed_std = float(scaler.scale_[SPEED_IDX])

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        speed_mean, speed_std,
        train_stations, test_stations,
    )


def train_one_seed(seed, loaders, speed_mean, speed_std, save_best=False):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders

    model = GRUAttention(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_window=OUTPUT_WINDOW,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)
    mse_fn = nn.MSELoss()
    history = {'train_total': [], 'val_total': [], 'lr': []}
    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        running, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred, _ = model(x)
            loss = mse_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        tr_loss = running / n

        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred, _ = model(x)
                running += mse_fn(pred, y).item() * x.size(0)
                n += x.size(0)
        val_loss = running / n

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        history['train_total'].append(tr_loss)
        history['val_total'].append(val_loss)
        history['lr'].append(cur_lr)

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    if save_best:
        torch.save(best_state, os.path.join(OUT_DIR, "best_model.pt"))

    model.eval()
    preds, truths, phases, attns = [], [], [], []
    with torch.no_grad():
        for x, y, ph in test_loader:
            pred, attn_w = model(x.to(DEVICE))
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
            phases.append(ph.numpy())
            attns.append(attn_w.cpu().numpy())
    preds = np.clip(np.concatenate(preds) * speed_std + speed_mean, 0, 100)
    truths = np.concatenate(truths) * speed_std + speed_mean
    phases = np.concatenate(phases)
    attns = np.concatenate(attns)

    rmse = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae = mean_absolute_error(truths.flatten(), preds.flatten())
    per_horizon_rmse = [
        np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        for h in range(OUTPUT_WINDOW)
    ]

    t_flat = truths.flatten()
    p_flat = preds.flatten()
    ph_flat = phases.flatten()
    phase_metrics = {}
    for name, pid in PHASE_TO_ID.items():
        mask = ph_flat == pid
        if mask.sum() > 0:
            phase_metrics[name] = {
                'n': int(mask.sum()),
                'rmse': float(np.sqrt(mean_squared_error(t_flat[mask], p_flat[mask]))),
                'mae': float(mean_absolute_error(t_flat[mask], p_flat[mask])),
            }
        else:
            phase_metrics[name] = {'n': 0, 'rmse': np.nan, 'mae': np.nan}

    phase_horizon = {}
    for name, pid in PHASE_TO_ID.items():
        phase_horizon[name] = []
        for h in range(OUTPUT_WINDOW):
            mask = phases[:, h] == pid
            if mask.sum() > 0:
                phase_horizon[name].append(
                    float(np.sqrt(mean_squared_error(truths[mask, h], preds[mask, h]))))
            else:
                phase_horizon[name].append(np.nan)

    return {
        'seed': seed, 'history': history, 'best_val': best_val,
        'preds': preds, 'truths': truths, 'phases': phases, 'attns': attns,
        'rmse': rmse, 'mae': mae,
        'per_horizon_rmse': per_horizon_rmse,
        'phase_metrics': phase_metrics, 'phase_horizon': phase_horizon,
    }


# Plots (same style as baseline/physics)
TITLE_PREFIX = "Sparse-sensor baseline"


def plot_loss_curve(r):
    h = r['history']
    ep = range(1, len(h['train_total'])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, h['train_total'], label='Train', linewidth=1.5)
    ax1.plot(ep, h['val_total'], label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, h['lr'], color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — LR Schedule')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth(r):
    truths, preds = r['truths'], r['preds']
    n = min(500, len(preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(preds[:n, 0], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(preds[:n, -1], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — Prediction vs Truth (t+30 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_truth.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_rmse(r):
    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(horizons, r['per_horizon_rmse'], width=3, color='steelblue', alpha=0.85)
    for hz, v in zip(horizons, r['per_horizon_rmse']):
        ax.text(hz, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — RMSE by Horizon')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_rmse.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention(r):
    attns = r['attns']
    avg = attns.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_samples = min(200, len(attns))
    idx = np.linspace(0, len(attns)-1, n_samples, dtype=int)
    for i in idx:
        ax.plot(range(1, INPUT_WINDOW+1), attns[i],
                alpha=0.08, linewidth=0.6, color='steelblue')
    ax.plot(range(1, INPUT_WINDOW+1), avg,
            linewidth=2.5, color='red', label='Mean')
    ax.set_xlabel('Input time step (1=oldest)')
    ax.set_ylabel('Attention weight')
    ax.set_title(f'{TITLE_PREFIX} (seed {r["seed"]}) — Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curve_5seed(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    max_len = max(len(r['history']['train_total']) for r in results)
    tr_arr = np.full((len(results), max_len), np.nan)
    val_arr = np.full((len(results), max_len), np.nan)
    for i, r in enumerate(results):
        L = len(r['history']['train_total'])
        tr_arr[i, :L] = r['history']['train_total']
        val_arr[i, :L] = r['history']['val_total']
    epochs = np.arange(1, max_len + 1)
    for r in results:
        L = len(r['history']['train_total'])
        ax1.plot(epochs[:L], r['history']['train_total'], alpha=0.3, linewidth=0.8, color='steelblue')
        ax1.plot(epochs[:L], r['history']['val_total'], alpha=0.3, linewidth=0.8, color='darkorange')
    ax1.plot(epochs, np.nanmean(tr_arr, axis=0), linewidth=2.0, color='steelblue', label='Train (mean)')
    ax1.plot(epochs, np.nanmean(val_arr, axis=0), linewidth=2.0, color='darkorange', label='Val (mean)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title(f'{TITLE_PREFIX} — 5-seed Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for r in results:
        L = len(r['history']['lr'])
        ax2.plot(range(1, L+1), r['history']['lr'], alpha=0.5, linewidth=1.0,
                 label=f'seed {r["seed"]}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'{TITLE_PREFIX} — 5-seed LR Schedules')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth_5seed(results):
    preds_stack = np.stack([r['preds'] for r in results], axis=0)
    mean_preds = preds_stack.mean(axis=0)
    truths = results[0]['truths']
    n = min(500, len(mean_preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(mean_preds[:n, 0], label='Prediction (5-seed mean)',
                 linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title(f'{TITLE_PREFIX} — 5-seed Mean Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(mean_preds[:n, -1], label='Prediction (5-seed mean)',
                 linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title(f'{TITLE_PREFIX} — 5-seed Mean Prediction vs Truth (t+30 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_truth_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_rmse_5seed(results):
    stack = np.stack([r['per_horizon_rmse'] for r in results], axis=0)
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(horizons, means, width=3, color='steelblue', alpha=0.85,
           yerr=stds, capsize=4, ecolor='black')
    for hz, m, s in zip(horizons, means, stds):
        ax.text(hz, m + s + 0.1, f'{m:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title(f'{TITLE_PREFIX} — 5-seed RMSE by Horizon (mean ± std)')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_rmse_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_5seed(results):
    mean_attns = np.stack([r['attns'].mean(axis=0) for r in results], axis=0)
    overall = mean_attns.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        ax.plot(range(1, INPUT_WINDOW+1), mean_attns[i],
                alpha=0.5, linewidth=1.2, label=f'seed {r["seed"]}')
    ax.plot(range(1, INPUT_WINDOW+1), overall,
            linewidth=2.5, color='red', label='5-seed mean')
    ax.set_xlabel('Input time step (1=oldest)')
    ax.set_ylabel('Attention weight')
    ax.set_title(f'{TITLE_PREFIX} — 5-seed Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_rmse_bars(results):
    names = ['other', 'onset', 'congested', 'recovery']
    stack = np.zeros((len(results), len(names)))
    for i, r in enumerate(results):
        for j, p in enumerate(names):
            stack[i, j] = r['phase_metrics'][p]['rmse']
    means = np.nanmean(stack, axis=0)
    stds = np.nanstd(stack, axis=0)
    ns = [results[0]['phase_metrics'][p]['n'] for p in names]
    colors = [PHASE_COLORS[p] for p in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, m, s, nsam in zip(bars, means, stds, ns):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.15,
                f'{m:.2f}\n(n={nsam:,})', ha='center', fontsize=9)
    ax.set_xlabel('Event phase')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title(f'{TITLE_PREFIX} — RMSE by Event Phase (5-seed mean ± std)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase_rmse_bars.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_horizon_rmse(results):
    names = ['other', 'onset', 'congested', 'recovery']
    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in names:
        stack = np.array([r['phase_horizon'][p] for r in results])
        means = np.nanmean(stack, axis=0)
        stds = np.nanstd(stack, axis=0)
        ax.errorbar(horizons, means, yerr=stds, marker='o', linewidth=1.6,
                    capsize=4, label=p, color=PHASE_COLORS[p])
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title(f'{TITLE_PREFIX} — RMSE by Phase × Horizon (5-seed mean ± std)')
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase_horizon_rmse.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(results, train_stations, test_stations):
    rmses = np.array([r['rmse'] for r in results])
    maes = np.array([r['mae'] for r in results])
    stack = np.stack([r['per_horizon_rmse'] for r in results], axis=0)
    ph_mean = stack.mean(axis=0)
    ph_std = stack.std(axis=0)

    with open(os.path.join(OUT_DIR, "metrics.txt"), 'w') as f:
        f.write("Sparse-sensor baseline (spatial generalization, 5 seeds)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train+Val stations ({len(train_stations)}): {train_stations}\n")
        f.write(f"Test stations      ({len(test_stations)}): {test_stations}\n")
        f.write("  (scaler fit on training stations only; "
                "test is spatially out-of-sample)\n\n")
        f.write(f"Test RMSE (overall): {rmses.mean():.4f} +/- {rmses.std():.4f} mph\n")
        f.write(f"Test MAE  (overall): {maes.mean():.4f} +/- {maes.std():.4f} mph\n\n")
        f.write("Per-horizon RMSE (mean +/- std):\n")
        for h in range(OUTPUT_WINDOW):
            f.write(f"  t+{(h+1)*5:>2} min: {ph_mean[h]:.4f} +/- {ph_std[h]:.4f}\n")
        f.write("\nPer-seed overall:\n")
        for r in results:
            f.write(f"  seed={r['seed']:>3}  RMSE={r['rmse']:.4f}  "
                    f"MAE={r['mae']:.4f}  best_val={r['best_val']:.6f}\n")
        f.write("\nConfig:\n")
        f.write(f"  seeds = {SEEDS}  (baseline-v6 architecture, sparse-sensor split)\n")
        f.write(f"  n_train_stations = {N_TRAIN_STATIONS}\n")
        f.write(f"  input_window = {INPUT_WINDOW} (15 min), output_window = {OUTPUT_WINDOW}\n")
        f.write(f"  hidden = {HIDDEN_SIZE}, layers = {NUM_LAYERS}, dropout = {DROPOUT}\n")
        f.write(f"  features = {FEATURE_COLS}\n")
        f.write(f"  batch = {BATCH_SIZE}, epochs = {EPOCHS}, lr = {LR}\n")
        f.write(f"  weight_decay = {WEIGHT_DECAY}\n")
        f.write(f"  early_stopping_patience = {PATIENCE}, min_delta = {MIN_DELTA}\n")


def save_phase_metrics(results):
    names = ['other', 'onset', 'congested', 'recovery']
    with open(os.path.join(OUT_DIR, "phase_metrics.txt"), 'w') as f:
        f.write("Sparse-sensor baseline — Phase-Stratified Metrics\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'Phase':<12} {'n':>10} {'RMSE':>16} {'MAE':>16}\n")
        f.write("-" * 55 + "\n")
        for p in names:
            rmses = np.array([r['phase_metrics'][p]['rmse'] for r in results])
            maes = np.array([r['phase_metrics'][p]['mae'] for r in results])
            n = results[0]['phase_metrics'][p]['n']
            f.write(f"{p:<12} {n:>10,}  {rmses.mean():>7.4f} +/- {rmses.std():.4f}  "
                    f"{maes.mean():>7.4f} +/- {maes.std():.4f}\n")
        f.write("\nPer-seed phase RMSE:\n")
        f.write(f"{'Seed':>6}")
        for p in names:
            f.write(f"  {p:>10}")
        f.write("\n")
        for r in results:
            f.write(f"{r['seed']:>6}")
            for p in names:
                f.write(f"  {r['phase_metrics'][p]['rmse']:>10.4f}")
            f.write("\n")


# Main
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}  (reference seed: {REF_SEED})")
    train_loader, val_loader, test_loader, speed_mean, speed_std, \
        train_stations, test_stations = prepare_data()
    loaders = (train_loader, val_loader, test_loader)

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = train_one_seed(seed, loaders, speed_mean, speed_std,
                           save_best=(seed == REF_SEED))
        results.append(r)
        print(f"  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")

    ref = next(r for r in results if r['seed'] == REF_SEED)
    plot_loss_curve(ref)
    plot_pred_vs_truth(ref)
    plot_horizon_rmse(ref)
    plot_attention(ref)
    plot_loss_curve_5seed(results)
    plot_pred_vs_truth_5seed(results)
    plot_horizon_rmse_5seed(results)
    plot_attention_5seed(results)
    plot_phase_rmse_bars(results)
    plot_phase_horizon_rmse(results)
    save_metrics(results, train_stations, test_stations)
    save_phase_metrics(results)

    rmses = np.array([r['rmse'] for r in results])
    print(f"\n5-seed RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f} mph")
    print(f"Outputs -> {OUT_DIR}/")
