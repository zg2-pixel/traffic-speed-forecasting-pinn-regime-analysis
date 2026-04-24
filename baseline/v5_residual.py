"""
v5_residual.py

Baseline v5: residual learning over 5 seeds. Model predicts the
residual between future speed and the last observed speed; final
prediction = last_speed + residual.

Generates both single-seed plots (seed=42, for comparison with v1-v4)
and 5-seed aggregated plots (for significance analysis).
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
DATA_PATH = "data/processed/traffic_clean.csv"
OUT_DIR = "baseline_output/v5_residual_output"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2025]
REF_SEED = 42

INPUT_WINDOW = 12
OUTPUT_WINDOW = 6
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
PATIENCE = 15
MIN_DELTA = 1e-4

FEATURE_COLS = ['avg_speed', 'total_flow', 'avg_occupancy',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
                'temperature', 'precipitation', 'wind_speed', 'weather_code']
TARGET_COL = 'avg_speed'
SPEED_IDX = 0
TRAIN_DAYS = 22
VAL_DAYS = 3

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


# Dataset: returns X, residual, last-speed baseline
class ResidualDataset(Dataset):
    def __init__(self, features, speed, input_window, output_window):
        self.X, self.y_res, self.v_base = [], [], []
        n = len(features)
        for i in range(n - input_window - output_window + 1):
            x = features[i:i + input_window]
            y_speed = speed[i + input_window:i + input_window + output_window]
            last = speed[i + input_window - 1]
            v_base = np.full(output_window, last)
            self.X.append(x)
            self.y_res.append(y_speed - v_base)
            self.v_base.append(v_base)
        self.X = torch.FloatTensor(np.array(self.X))
        self.y_res = torch.FloatTensor(np.array(self.y_res))
        self.v_base = torch.FloatTensor(np.array(self.v_base))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_res[idx], self.v_base[idx]


# Model
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


# Data
def prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)
    stations = sorted(df['station'].unique())

    dates = sorted(df['timestamp'].dt.date.unique())
    train_end = dates[TRAIN_DAYS - 1]
    val_end = dates[TRAIN_DAYS + VAL_DAYS - 1]

    scaler = StandardScaler()
    scaler.fit(df.loc[df['timestamp'].dt.date <= train_end, FEATURE_COLS])
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    train_list, val_list, test_list = [], [], []

    for sid in stations:
        sdf = df[df['station'] == sid].reset_index(drop=True)
        feat = sdf[FEATURE_COLS].values
        sp = sdf[TARGET_COL].values

        s_train = sdf['timestamp'].dt.date <= train_end
        s_val = (sdf['timestamp'].dt.date > train_end) & (sdf['timestamp'].dt.date <= val_end)
        s_test = sdf['timestamp'].dt.date > val_end

        for mask, lst in [(s_train, train_list), (s_val, val_list), (s_test, test_list)]:
            f, s = feat[mask], sp[mask]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                lst.append(ResidualDataset(f, s, INPUT_WINDOW, OUTPUT_WINDOW))

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)
    test_ds = ConcatDataset(test_list)

    speed_mean = scaler.mean_[SPEED_IDX]
    speed_std = scaler.scale_[SPEED_IDX]

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        speed_mean, speed_std,
    )


# Train + evaluate one seed
def train_one_seed(seed, loaders, speed_mean, speed_std, save_best_model=False):
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
        optimizer, mode='min', factor=0.5, patience=5
    )
    mse_fn = nn.MSELoss()
    train_losses, val_losses, lr_history = [], [], []
    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        running, n = 0.0, 0
        for x, y_res, _ in train_loader:
            x, y_res = x.to(DEVICE), y_res.to(DEVICE)
            pred_res, _ = model(x)
            loss = mse_fn(pred_res, y_res)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / n

        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for x, y_res, _ in val_loader:
                x, y_res = x.to(DEVICE), y_res.to(DEVICE)
                pred_res, _ = model(x)
                running += mse_fn(pred_res, y_res).item() * x.size(0)
                n += x.size(0)
        val_loss = running / n

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    if save_best_model:
        torch.save(best_state, os.path.join(OUT_DIR, "best_model.pt"))

    # Test
    model.eval()
    preds, truths, baselines, attns = [], [], [], []
    with torch.no_grad():
        for x, y_res, v_base in test_loader:
            x_d = x.to(DEVICE)
            pred_res, attn_w = model(x_d)
            pred_res = pred_res.cpu()
            pred_speed = v_base + pred_res
            true_speed = v_base + y_res
            preds.append(pred_speed.numpy() * speed_std + speed_mean)
            truths.append(true_speed.numpy() * speed_std + speed_mean)
            baselines.append(v_base.numpy() * speed_std + speed_mean)
            attns.append(attn_w.cpu().numpy())

    preds = np.clip(np.concatenate(preds), 0, 100)
    truths = np.concatenate(truths)
    baselines = np.concatenate(baselines)
    attns = np.concatenate(attns)

    rmse_model = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae_model = mean_absolute_error(truths.flatten(), preds.flatten())
    rmse_base = np.sqrt(mean_squared_error(truths.flatten(), baselines.flatten()))
    mae_base = mean_absolute_error(truths.flatten(), baselines.flatten())
    per_horizon_rmse = [
        np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        for h in range(OUTPUT_WINDOW)
    ]

    return {
        'seed': seed,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'best_val': best_val,
        'preds': preds,
        'truths': truths,
        'attns': attns,
        'rmse_model': rmse_model, 'mae_model': mae_model,
        'rmse_base': rmse_base, 'mae_base': mae_base,
        'per_horizon_rmse': per_horizon_rmse,
    }


# Plotting (single-seed, identical to v1-v4)
def plot_loss_curve(r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, len(r['train_losses'])+1), r['train_losses'], label='Train', linewidth=1.5)
    ax1.plot(range(1, len(r['val_losses'])+1), r['val_losses'], label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (residual)')
    ax1.set_title(f'v5 (seed {r["seed"]}) — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(r['lr_history'])+1), r['lr_history'], color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'v5 (seed {r["seed"]}) — LR Schedule')
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
    axes[0].set_title(f'v5 (seed {r["seed"]}) — Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(preds[:n, -1], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title(f'v5 (seed {r["seed"]}) — Prediction vs Truth (t+30 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_truth.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_rmse(r):
    per_horizon = r['per_horizon_rmse']
    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(horizons, per_horizon, width=3, color='steelblue', alpha=0.85)
    for h, v in zip(horizons, per_horizon):
        ax.text(h, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title(f'v5 (seed {r["seed"]}) — RMSE by Horizon')
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
    ax.set_xlabel('Input time step (1=oldest, 12=newest)')
    ax.set_ylabel('Attention weight')
    ax.set_title(f'v5 (seed {r["seed"]}) — Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention.png"), dpi=150, bbox_inches='tight')
    plt.close()


# Plotting (5-seed aggregated)
def plot_loss_curve_5seed(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    max_len = max(len(r['train_losses']) for r in all_results)
    train_arr = np.full((len(all_results), max_len), np.nan)
    val_arr = np.full((len(all_results), max_len), np.nan)
    for i, r in enumerate(all_results):
        L = len(r['train_losses'])
        train_arr[i, :L] = r['train_losses']
        val_arr[i, :L] = r['val_losses']

    epochs = np.arange(1, max_len + 1)
    for i, r in enumerate(all_results):
        L = len(r['train_losses'])
        ax1.plot(epochs[:L], r['train_losses'], alpha=0.3, linewidth=0.8, color='steelblue')
        ax1.plot(epochs[:L], r['val_losses'], alpha=0.3, linewidth=0.8, color='darkorange')
    ax1.plot(epochs, np.nanmean(train_arr, axis=0), linewidth=2.0, color='steelblue', label='Train (mean)')
    ax1.plot(epochs, np.nanmean(val_arr, axis=0), linewidth=2.0, color='darkorange', label='Val (mean)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (residual)')
    ax1.set_title(f'v5 — 5-seed Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for r in all_results:
        L = len(r['lr_history'])
        ax2.plot(range(1, L+1), r['lr_history'], alpha=0.5, linewidth=1.0,
                 label=f'seed {r["seed"]}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('v5 — 5-seed LR Schedules')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth_5seed(all_results):
    # Average predictions across seeds
    preds_stack = np.stack([r['preds'] for r in all_results], axis=0)
    mean_preds = preds_stack.mean(axis=0)
    truths = all_results[0]['truths']

    n = min(500, len(mean_preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(mean_preds[:n, 0], label='Prediction (5-seed mean)',
                 linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title('v5 — 5-seed Mean Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(mean_preds[:n, -1], label='Prediction (5-seed mean)',
                 linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title('v5 — 5-seed Mean Prediction vs Truth (t+30 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_truth_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_rmse_5seed(all_results):
    per_horizon_stack = np.stack([r['per_horizon_rmse'] for r in all_results], axis=0)
    means = per_horizon_stack.mean(axis=0)
    stds = per_horizon_stack.std(axis=0)

    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(horizons, means, width=3, color='steelblue', alpha=0.85,
           yerr=stds, capsize=4, ecolor='black')
    for h, m, s in zip(horizons, means, stds):
        ax.text(h, m + s + 0.08, f'{m:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title('v5 — 5-seed RMSE by Horizon (mean ± std)')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_rmse_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_5seed(all_results):
    mean_attns = np.stack([r['attns'].mean(axis=0) for r in all_results], axis=0)
    overall_mean = mean_attns.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(all_results):
        ax.plot(range(1, INPUT_WINDOW+1), mean_attns[i],
                alpha=0.5, linewidth=1.2, label=f'seed {r["seed"]}')
    ax.plot(range(1, INPUT_WINDOW+1), overall_mean,
            linewidth=2.5, color='red', label='5-seed mean')
    ax.set_xlabel('Input time step (1=oldest, 12=newest)')
    ax.set_ylabel('Attention weight')
    ax.set_title('v5 — 5-seed Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


# v5-specific: significance bars (LastSpeed baseline vs residual model)
def plot_significance_bars(all_results):
    seeds = [r['seed'] for r in all_results]
    base_rmse = [r['rmse_base'] for r in all_results]
    model_rmse = [r['rmse_model'] for r in all_results]
    base_mae = [r['mae_base'] for r in all_results]
    model_mae = [r['mae_model'] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(seeds))
    w = 0.35

    ax1.bar(x - w/2, base_rmse, w, label='Last-speed baseline', color='gray', alpha=0.8)
    ax1.bar(x + w/2, model_rmse, w, label='Residual model', color='steelblue', alpha=0.85)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('RMSE (mph)')
    ax1.set_title('v5 — RMSE by Seed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x - w/2, base_mae, w, label='Last-speed baseline', color='gray', alpha=0.8)
    ax2.bar(x + w/2, model_mae, w, label='Residual model', color='steelblue', alpha=0.85)
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('MAE (mph)')
    ax2.set_title('v5 — MAE by Seed')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "significance_bars.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(all_results):
    rmses = np.array([r['rmse_model'] for r in all_results])
    maes = np.array([r['mae_model'] for r in all_results])
    per_horizon_stack = np.stack([r['per_horizon_rmse'] for r in all_results], axis=0)
    ph_mean = per_horizon_stack.mean(axis=0)
    ph_std = per_horizon_stack.std(axis=0)

    with open(os.path.join(OUT_DIR, "metrics.txt"), 'w') as f:
        f.write("Baseline v5 (residual learning, 5 seeds)\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Test RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f} mph\n")
        f.write(f"Test MAE:  {maes.mean():.4f} +/- {maes.std():.4f} mph\n\n")
        f.write("Per-horizon RMSE (mean +/- std):\n")
        for h in range(OUTPUT_WINDOW):
            f.write(f"  t+{(h+1)*5:>2} min: {ph_mean[h]:.4f} +/- {ph_std[h]:.4f}\n")

        f.write("\nPer-seed results:\n")
        for r in all_results:
            f.write(f"  seed={r['seed']:>5}  RMSE={r['rmse_model']:.4f}  "
                    f"MAE={r['mae_model']:.4f}  best_val={r['best_val']:.6f}\n")

        f.write("\nConfig:\n")
        f.write(f"  seeds = {SEEDS}\n")
        f.write(f"  input_window = {INPUT_WINDOW}, output_window = {OUTPUT_WINDOW}\n")
        f.write(f"  hidden = {HIDDEN_SIZE}, layers = {NUM_LAYERS}, dropout = {DROPOUT}\n")
        f.write(f"  batch = {BATCH_SIZE}, epochs = {EPOCHS}, lr = {LR}\n")
        f.write(f"  weight_decay = {WEIGHT_DECAY}\n")
        f.write(f"  early_stopping_patience = {PATIENCE}, min_delta = {MIN_DELTA}\n")
        f.write(f"  scheduler = ReduceLROnPlateau(factor=0.5, patience=5)\n")
        f.write(f"  features = {len(FEATURE_COLS)}\n")
        f.write(f"  target = residual (future_speed - last_observed_speed)\n")


def save_significance_table(all_results):
    base_rmse = np.array([r['rmse_base'] for r in all_results])
    model_rmse = np.array([r['rmse_model'] for r in all_results])
    base_mae = np.array([r['mae_base'] for r in all_results])
    model_mae = np.array([r['mae_model'] for r in all_results])
    wins_rmse = int((model_rmse < base_rmse).sum())
    wins_mae = int((model_mae < base_mae).sum())

    with open(os.path.join(OUT_DIR, "significance_table.txt"), 'w') as f:
        f.write("v5 Significance Test: Residual Model vs Last-Speed Baseline\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Seed':>6} {'BASE_RMSE':>12} {'MODEL_RMSE':>12} "
                f"{'BASE_MAE':>12} {'MODEL_MAE':>12}\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            f.write(f"{r['seed']:>6} {r['rmse_base']:>12.4f} {r['rmse_model']:>12.4f} "
                    f"{r['mae_base']:>12.4f} {r['mae_model']:>12.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Mean':>6} {base_rmse.mean():>12.4f} {model_rmse.mean():>12.4f} "
                f"{base_mae.mean():>12.4f} {model_mae.mean():>12.4f}\n")
        f.write(f"{'Std':>6} {base_rmse.std():>12.4f} {model_rmse.std():>12.4f} "
                f"{base_mae.std():>12.4f} {model_mae.std():>12.4f}\n\n")
        f.write(f"Model wins on RMSE: {wins_rmse}/{len(SEEDS)}\n")
        f.write(f"Model wins on MAE:  {wins_mae}/{len(SEEDS)}\n")


# Main
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}  (reference seed for single-seed plots: {REF_SEED})")

    train_loader, val_loader, test_loader, speed_mean, speed_std = prepare_data()
    loaders = (train_loader, val_loader, test_loader)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        save_model = (seed == REF_SEED)
        r = train_one_seed(seed, loaders, speed_mean, speed_std,
                           save_best_model=save_model)
        all_results.append(r)
        print(f"  RMSE (model): {r['rmse_model']:.4f}  "
              f"RMSE (last-speed baseline): {r['rmse_base']:.4f}")

    # Reference seed for single-seed plots
    ref = next(r for r in all_results if r['seed'] == REF_SEED)

    # Single-seed plots (match v1-v4 naming)
    plot_loss_curve(ref)
    plot_pred_vs_truth(ref)
    plot_horizon_rmse(ref)
    plot_attention(ref)

    # 5-seed aggregated plots
    plot_loss_curve_5seed(all_results)
    plot_pred_vs_truth_5seed(all_results)
    plot_horizon_rmse_5seed(all_results)
    plot_attention_5seed(all_results)

    # v5-specific
    plot_significance_bars(all_results)
    save_significance_table(all_results)

    # Aggregated metrics
    save_metrics(all_results)

    rmses = np.array([r['rmse_model'] for r in all_results])
    print(f"\n5-seed RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f} mph")
    print(f"Outputs -> {OUT_DIR}/")
