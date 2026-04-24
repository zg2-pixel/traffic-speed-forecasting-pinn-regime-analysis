"""
v3_greenshields_fd.py

Physics v3: three penalties with the congestion threshold replaced by
a continuous Greenshields fundamental-diagram reference,
  v_FD(occ) = V_f * (1 - occ / occ_jam),
fitted from training data (V_f = 72.39 mph, occ_jam = 0.4366).

Built on baseline v6 (INPUT=3, speed+occupancy only). 5 seeds.
Generates single-seed plots (seed=42) and 5-seed aggregated plots,
matching baseline v6 output layout.
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
OUT_DIR = "physics_output/v3_greenshields_fd_output"
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
OCC_IDX = 1

# Greenshields parameters (fitted from training data)
VF = 72.39
OCC_JAM = 0.4366

# Physics constraints
ACC_LIMIT = 15.0
SPEED_MIN = 0.0
SPEED_MAX = 85.0

LAMBDA_ACC = 0.01
LAMBDA_SPEED = 0.01
LAMBDA_FD = 0.0005

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
    raise FileNotFoundError(
        f"Could not find all_splits_with_regime_labels.csv. Candidates: {LABEL_CSV_CANDIDATES}")


# Dataset: returns X, y, future occupancy (original scale, for FD penalty)
class PhysicsDataset(Dataset):
    def __init__(self, features, targets, occ_orig, input_window, output_window):
        self.X, self.y, self.occ = [], [], []
        n = len(features)
        for i in range(n - input_window - output_window + 1):
            self.X.append(features[i:i + input_window])
            self.y.append(targets[i + input_window:i + input_window + output_window])
            self.occ.append(occ_orig[i + input_window:i + input_window + output_window])
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))
        self.occ = torch.FloatTensor(np.array(self.occ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.occ[idx]


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


# Physics losses
def greenshields_mph(occ_orig):
    return torch.clamp(VF * (1.0 - occ_orig / OCC_JAM), min=0.0, max=VF)


def compute_physics_losses(pred_scaled, occ_orig, speed_mean, speed_std):
    pred_mph = pred_scaled * speed_std + speed_mean

    if pred_mph.shape[1] > 1:
        dv = torch.abs(pred_mph[:, 1:] - pred_mph[:, :-1])
        L_acc = torch.clamp(dv - ACC_LIMIT, min=0).mean()
    else:
        L_acc = torch.tensor(0.0, device=pred_mph.device)

    below = torch.clamp(SPEED_MIN - pred_mph, min=0)
    above = torch.clamp(pred_mph - SPEED_MAX, min=0)
    L_speed = (below + above).mean()

    # Greenshields FD penalty: encourage v ~ v_FD(occ)
    v_fd = greenshields_mph(occ_orig)
    L_fd = ((pred_mph - v_fd) ** 2).mean()

    return L_acc, L_speed, L_fd


# Data
def prepare_data():
    path = resolve_label_csv()
    print(f"Loading: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)
    required = ['timestamp', 'station', 'avg_speed', 'avg_occupancy', 'split']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    occ_orig_all = df['avg_occupancy'].values.copy()

    scaler = StandardScaler()
    scaler.fit(df.loc[df['split'] == 'train', FEATURE_COLS])
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    train_list, val_list, test_list = [], [], []

    for sid in sorted(df['station'].unique()):
        mask = df['station'] == sid
        sdf = df_scaled[mask].reset_index(drop=True)
        occ_sid = occ_orig_all[mask.values]
        feat = sdf[FEATURE_COLS].values
        tgt = sdf[TARGET_COL].values

        for split_name, lst in [('train', train_list), ('val', val_list), ('test', test_list)]:
            sel = (sdf['split'] == split_name).values
            f, t, o = feat[sel], tgt[sel], occ_sid[sel]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                lst.append(PhysicsDataset(f, t, o, INPUT_WINDOW, OUTPUT_WINDOW))

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)
    test_ds = ConcatDataset(test_list)

    speed_mean = float(scaler.mean_[SPEED_IDX])
    speed_std = float(scaler.scale_[SPEED_IDX])

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        speed_mean, speed_std,
    )


# Train + evaluate one seed
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
        optimizer, mode='min', factor=0.5, patience=4
    )
    mse_fn = nn.MSELoss()
    history = {'train_total': [], 'val_total': [], 'lr': [],
               'train_data': [], 'train_acc': [], 'train_speed': [], 'train_fd': []}
    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        sums = {'total': 0.0, 'data': 0.0, 'acc': 0.0, 'speed': 0.0, 'fd': 0.0}
        n = 0
        for x, y, occ in train_loader:
            x, y, occ = x.to(DEVICE), y.to(DEVICE), occ.to(DEVICE)
            pred, _ = model(x)
            L_data = mse_fn(pred, y)
            L_acc, L_speed, L_fd = compute_physics_losses(pred, occ, speed_mean, speed_std)
            total = L_data + LAMBDA_ACC * L_acc + LAMBDA_SPEED * L_speed + LAMBDA_FD * L_fd

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            bs = x.size(0)
            sums['total'] += total.item() * bs
            sums['data'] += L_data.item() * bs
            sums['acc'] += L_acc.item() * bs
            sums['speed'] += L_speed.item() * bs
            sums['fd'] += L_fd.item() * bs
            n += bs
        tr = {k: v / n for k, v in sums.items()}

        model.eval()
        vsum, vn = 0.0, 0
        with torch.no_grad():
            for x, y, occ in val_loader:
                x, y, occ = x.to(DEVICE), y.to(DEVICE), occ.to(DEVICE)
                pred, _ = model(x)
                L_data = mse_fn(pred, y)
                L_acc, L_speed, L_fd = compute_physics_losses(pred, occ, speed_mean, speed_std)
                total = L_data + LAMBDA_ACC * L_acc + LAMBDA_SPEED * L_speed + LAMBDA_FD * L_fd
                vsum += total.item() * x.size(0)
                vn += x.size(0)
        val_total = vsum / vn

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_total)

        history['train_total'].append(tr['total'])
        history['val_total'].append(val_total)
        history['lr'].append(cur_lr)
        history['train_data'].append(tr['data'])
        history['train_acc'].append(tr['acc'])
        history['train_speed'].append(tr['speed'])
        history['train_fd'].append(tr['fd'])

        if val_total < best_val - MIN_DELTA:
            best_val = val_total
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    if save_best:
        torch.save(best_state, os.path.join(OUT_DIR, "best_model.pt"))

    # Test
    model.eval()
    preds, truths, occs, attns = [], [], [], []
    with torch.no_grad():
        for x, y, occ in test_loader:
            pred, attn_w = model(x.to(DEVICE))
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
            occs.append(occ.numpy())
            attns.append(attn_w.cpu().numpy())
    preds = np.clip(np.concatenate(preds) * speed_std + speed_mean, 0, 100)
    truths = np.concatenate(truths) * speed_std + speed_mean
    occs = np.concatenate(occs)
    attns = np.concatenate(attns)

    rmse = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae = mean_absolute_error(truths.flatten(), preds.flatten())
    per_horizon_rmse = [
        np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        for h in range(OUTPUT_WINDOW)
    ]

    # FD deviation on test set
    v_fd_test = np.clip(VF * (1.0 - occs / OCC_JAM), 0, VF)
    fd_gap = np.abs(preds - v_fd_test).mean()

    return {
        'seed': seed,
        'history': history,
        'best_val': best_val,
        'preds': preds,
        'truths': truths,
        'occs': occs,
        'attns': attns,
        'rmse': rmse, 'mae': mae,
        'per_horizon_rmse': per_horizon_rmse,
        'fd_gap_mean': float(fd_gap),
    }


# Single-seed plots (aligned with baseline v4/v6 naming)
def plot_loss_curve(r):
    h = r['history']
    ep = range(1, len(h['train_total'])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, h['train_total'], label='Train', linewidth=1.5)
    ax1.plot(ep, h['val_total'], label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title(f'v3 (seed {r["seed"]}) — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, h['lr'], color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'v3 (seed {r["seed"]}) — LR Schedule')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_physics_losses(r):
    h = r['history']
    ep = range(1, len(h['train_data'])+1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, h['train_data'], label='Data loss (MSE)', linewidth=1.6)
    ax.plot(ep, h['train_acc'], label='Acceleration', linewidth=1.2)
    ax.plot(ep, h['train_speed'], label='Speed range', linewidth=1.2)
    ax.plot(ep, h['train_fd'], label='Greenshields FD', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Unweighted component loss')
    ax.set_title(f'v3 (seed {r["seed"]}) — Physics Loss Components')
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "physics_losses.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth(r):
    truths, preds = r['truths'], r['preds']
    n = min(500, len(preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(preds[:n, 0], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title(f'v3 (seed {r["seed"]}) — Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(preds[:n, -1], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title(f'v3 (seed {r["seed"]}) — Prediction vs Truth (t+30 min)')
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
    ax.set_title(f'v3 (seed {r["seed"]}) — RMSE by Horizon')
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
    ax.set_title(f'v3 (seed {r["seed"]}) — Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention.png"), dpi=150, bbox_inches='tight')
    plt.close()


# 5-seed aggregated plots
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
    ax1.set_ylabel('Total Loss')
    ax1.set_title('v3 — 5-seed Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for r in results:
        L = len(r['history']['lr'])
        ax2.plot(range(1, L+1), r['history']['lr'], alpha=0.5, linewidth=1.0,
                 label=f'seed {r["seed"]}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('v3 — 5-seed LR Schedules')
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
    axes[0].set_title('v3 — 5-seed Mean Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(mean_preds[:n, -1], label='Prediction (5-seed mean)',
                 linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title('v3 — 5-seed Mean Prediction vs Truth (t+30 min)')
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
    ax.set_title('v3 — 5-seed RMSE by Horizon (mean ± std)')
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
    ax.set_title('v3 — 5-seed Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention_5seed.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(results):
    rmses = np.array([r['rmse'] for r in results])
    maes = np.array([r['mae'] for r in results])
    fd_gaps = np.array([r['fd_gap_mean'] for r in results])
    stack = np.stack([r['per_horizon_rmse'] for r in results], axis=0)
    ph_mean = stack.mean(axis=0)
    ph_std = stack.std(axis=0)

    with open(os.path.join(OUT_DIR, "metrics.txt"), 'w') as f:
        f.write("Physics v3 (Greenshields FD as penalty, 5 seeds)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f} mph\n")
        f.write(f"Test MAE:  {maes.mean():.4f} +/- {maes.std():.4f} mph\n")
        f.write(f"Mean |pred - v_FD(occ)|: {fd_gaps.mean():.4f} mph\n\n")
        f.write("Per-horizon RMSE (mean +/- std):\n")
        for h in range(OUTPUT_WINDOW):
            f.write(f"  t+{(h+1)*5:>2} min: {ph_mean[h]:.4f} +/- {ph_std[h]:.4f}\n")
        f.write("\nPer-seed results:\n")
        for r in results:
            f.write(f"  seed={r['seed']:>3}  RMSE={r['rmse']:.4f}  "
                    f"MAE={r['mae']:.4f}  fd_gap={r['fd_gap_mean']:.4f}  "
                    f"best_val={r['best_val']:.6f}\n")
        f.write("\nConfig:\n")
        f.write(f"  seeds = {SEEDS}  (baseline-v6 architecture)\n")
        f.write(f"  input_window = {INPUT_WINDOW} (15 min), output_window = {OUTPUT_WINDOW}\n")
        f.write(f"  hidden = {HIDDEN_SIZE}, layers = {NUM_LAYERS}, dropout = {DROPOUT}\n")
        f.write(f"  features = {FEATURE_COLS}\n")
        f.write(f"  batch = {BATCH_SIZE}, epochs = {EPOCHS}, lr = {LR}\n")
        f.write(f"  weight_decay = {WEIGHT_DECAY}\n")
        f.write(f"  early_stopping_patience = {PATIENCE}, min_delta = {MIN_DELTA}\n\n")
        f.write("Physics constraints:\n")
        f.write(f"  acc_limit = {ACC_LIMIT} mph/5min  (lambda = {LAMBDA_ACC})\n")
        f.write(f"  speed_range = [{SPEED_MIN}, {SPEED_MAX}] mph  (lambda = {LAMBDA_SPEED})\n")
        f.write(f"  Greenshields FD: v = {VF} * (1 - occ/{OCC_JAM})  "
                f"(lambda = {LAMBDA_FD})\n")


def save_violation_stats(results):
    dv_list = []
    speed_viol_list = []
    fd_gap_list = []
    for r in results:
        p = r['preds']
        dv = np.abs(p[:, 1:] - p[:, :-1])
        dv_list.append(float((dv > ACC_LIMIT).mean()))
        speed_viol_list.append(float(((p < SPEED_MIN) | (p > SPEED_MAX)).mean()))
        v_fd = np.clip(VF * (1.0 - r['occs'] / OCC_JAM), 0, VF)
        fd_gap_list.append(float(np.abs(p - v_fd).mean()))

    with open(os.path.join(OUT_DIR, "violation_stats.txt"), 'w') as f:
        f.write("v3 — Test-set Constraint Stats (5-seed mean)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Acceleration violations  (|dv| > {ACC_LIMIT} mph): "
                f"{np.mean(dv_list)*100:.3f}% +/- {np.std(dv_list)*100:.3f}%\n")
        f.write(f"Speed-range violations   (v<{SPEED_MIN} or v>{SPEED_MAX}): "
                f"{np.mean(speed_viol_list)*100:.3f}% +/- {np.std(speed_viol_list)*100:.3f}%\n")
        f.write(f"Mean |pred - v_FD(occ)|: "
                f"{np.mean(fd_gap_list):.4f} +/- {np.std(fd_gap_list):.4f} mph\n")


# Main
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}  (reference seed: {REF_SEED})")
    train_loader, val_loader, test_loader, speed_mean, speed_std = prepare_data()
    loaders = (train_loader, val_loader, test_loader)

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = train_one_seed(seed, loaders, speed_mean, speed_std,
                           save_best=(seed == REF_SEED))
        results.append(r)
        print(f"  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  fd_gap={r['fd_gap_mean']:.4f}")

    ref = next(r for r in results if r['seed'] == REF_SEED)

    plot_loss_curve(ref)
    plot_physics_losses(ref)
    plot_pred_vs_truth(ref)
    plot_horizon_rmse(ref)
    plot_attention(ref)

    plot_loss_curve_5seed(results)
    plot_pred_vs_truth_5seed(results)
    plot_horizon_rmse_5seed(results)
    plot_attention_5seed(results)

    save_metrics(results)
    save_violation_stats(results)

    rmses = np.array([r['rmse'] for r in results])
    print(f"\n5-seed RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f} mph")
    print(f"Outputs -> {OUT_DIR}/")
