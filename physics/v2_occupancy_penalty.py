"""
v2_occupancy_penalty.py

Physics v2: same three-penalty scheme as v1, but congestion is
detected by occupancy (avg_occupancy > 0.15) instead of flow.
This ablation isolates whether the congestion signal itself is
the issue, independent of which indicator defines congestion.
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
OUT_DIR = "physics_output/v2_occupancy_penalty_output"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

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
PATIENCE = 12
MIN_DELTA = 1e-4

FEATURE_COLS = ['avg_speed', 'total_flow', 'avg_occupancy',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
                'temperature', 'precipitation', 'wind_speed', 'weather_code']
TARGET_COL = 'avg_speed'
SPEED_IDX = 0
OCC_IDX = 2
TRAIN_DAYS = 22
VAL_DAYS = 3

# Physics constraints
ACC_LIMIT = 15.0
SPEED_MIN = 0.0
SPEED_MAX = 85.0
OCC_CONG = 0.15
V_CONG = 50.0

LAMBDA_ACC = 0.01
LAMBDA_SPEED = 0.01
LAMBDA_CONG = 0.001

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


# Dataset: returns X, y, and future occupancy (original scale, for penalty)
class StationDataset(Dataset):
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

    # Occupancy-based congestion: occ > OCC_CONG => v <= V_CONG
    is_cong = (occ_orig > OCC_CONG).float()
    excess = torch.clamp(pred_mph - V_CONG, min=0) * is_cong
    L_cong = excess.mean()

    return L_acc, L_speed, L_cong


# Data
def prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)
    stations = sorted(df['station'].unique())

    dates = sorted(df['timestamp'].dt.date.unique())
    train_end = dates[TRAIN_DAYS - 1]
    val_end = dates[TRAIN_DAYS + VAL_DAYS - 1]

    occ_orig_all = df['avg_occupancy'].values.copy()

    scaler = StandardScaler()
    scaler.fit(df.loc[df['timestamp'].dt.date <= train_end, FEATURE_COLS])
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    train_list, val_list, test_list = [], [], []

    for sid in stations:
        mask = df['station'] == sid
        sdf = df[mask].reset_index(drop=True)
        sdf_occ = occ_orig_all[mask.values]
        feat = sdf[FEATURE_COLS].values
        tgt = sdf[TARGET_COL].values

        s_train = sdf['timestamp'].dt.date <= train_end
        s_val = (sdf['timestamp'].dt.date > train_end) & (sdf['timestamp'].dt.date <= val_end)
        s_test = sdf['timestamp'].dt.date > val_end

        for sel, lst in [(s_train, train_list), (s_val, val_list), (s_test, test_list)]:
            f, t, o = feat[sel], tgt[sel], sdf_occ[sel]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                lst.append(StationDataset(f, t, o, INPUT_WINDOW, OUTPUT_WINDOW))

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


# Training
def train_model(model, train_loader, val_loader, speed_mean, speed_std):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    mse_fn = nn.MSELoss()
    history = {'train_total': [], 'val_total': [], 'lr': [],
               'train_data': [], 'train_acc': [], 'train_speed': [], 'train_cong': []}
    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        sums = {'total': 0.0, 'data': 0.0, 'acc': 0.0, 'speed': 0.0, 'cong': 0.0}
        n_samples = 0
        for x, y, occ in train_loader:
            x, y, occ = x.to(DEVICE), y.to(DEVICE), occ.to(DEVICE)
            pred, _ = model(x)
            L_data = mse_fn(pred, y)
            L_acc, L_speed, L_cong = compute_physics_losses(pred, occ, speed_mean, speed_std)
            total = L_data + LAMBDA_ACC * L_acc + LAMBDA_SPEED * L_speed + LAMBDA_CONG * L_cong

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            bs = x.size(0)
            sums['total'] += total.item() * bs
            sums['data'] += L_data.item() * bs
            sums['acc'] += L_acc.item() * bs
            sums['speed'] += L_speed.item() * bs
            sums['cong'] += L_cong.item() * bs
            n_samples += bs
        tr = {k: v / n_samples for k, v in sums.items()}

        model.eval()
        val_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y, occ in val_loader:
                x, y, occ = x.to(DEVICE), y.to(DEVICE), occ.to(DEVICE)
                pred, _ = model(x)
                L_data = mse_fn(pred, y)
                L_acc, L_speed, L_cong = compute_physics_losses(pred, occ, speed_mean, speed_std)
                total = L_data + LAMBDA_ACC * L_acc + LAMBDA_SPEED * L_speed + LAMBDA_CONG * L_cong
                val_sum += total.item() * x.size(0)
                n += x.size(0)
        val_total = val_sum / n

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_total)

        history['train_total'].append(tr['total'])
        history['val_total'].append(val_total)
        history['lr'].append(cur_lr)
        history['train_data'].append(tr['data'])
        history['train_acc'].append(tr['acc'])
        history['train_speed'].append(tr['speed'])
        history['train_cong'].append(tr['cong'])

        improved = val_total < best_val - MIN_DELTA
        if improved:
            best_val = val_total
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            tag = " *"
        else:
            wait += 1
            tag = ""

        if (epoch + 1) % 5 == 0 or improved:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"train={tr['total']:.5f}  val={val_total:.5f}  "
                  f"data={tr['data']:.5f}  acc={tr['acc']:.5f}  "
                  f"speed={tr['speed']:.5f}  cong={tr['cong']:.5f}{tag}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, "best_model.pt"))
    return history, best_val


# Evaluation
def evaluate(model, test_loader, speed_mean, speed_std):
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

    dv = np.abs(preds[:, 1:] - preds[:, :-1])
    acc_viol = float((dv > ACC_LIMIT).mean())
    speed_viol = float(((preds < SPEED_MIN) | (preds > SPEED_MAX)).mean())
    is_cong = (occs > OCC_CONG)
    cong_viol = float(((preds > V_CONG) & is_cong).sum() / max(is_cong.sum(), 1))
    viol = {
        'acc_violation_rate': acc_viol,
        'speed_violation_rate': speed_viol,
        'cong_violation_rate': cong_viol,
        'n_congested_samples': int(is_cong.sum()),
    }
    return preds, truths, attns, rmse, mae, per_horizon_rmse, viol


# Plots (same structure as v1)
def plot_loss_curve(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history['train_total'])+1)
    ax1.plot(ep, history['train_total'], label='Train', linewidth=1.5)
    ax1.plot(ep, history['val_total'], label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('v2 (occupancy penalty) — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history['lr'], color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('v2 — LR Schedule')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_physics_losses(history):
    ep = range(1, len(history['train_data'])+1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, history['train_data'], label='Data loss (MSE)', linewidth=1.6)
    ax.plot(ep, history['train_acc'], label='Acceleration', linewidth=1.2)
    ax.plot(ep, history['train_speed'], label='Speed range', linewidth=1.2)
    ax.plot(ep, history['train_cong'], label='Occupancy-based congestion', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Unweighted component loss')
    ax.set_title('v2 — Physics Loss Components (training)')
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "physics_losses.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth(truths, preds):
    n = min(500, len(preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(preds[:n, 0], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title('v2 — Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(preds[:n, -1], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title('v2 — Prediction vs Truth (t+30 min)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_truth.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_horizon_rmse(per_horizon_rmse):
    horizons = [(h+1)*5 for h in range(OUTPUT_WINDOW)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(horizons, per_horizon_rmse, width=3, color='steelblue', alpha=0.85)
    for h, r in zip(horizons, per_horizon_rmse):
        ax.text(h, r + 0.05, f'{r:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('Prediction Horizon (min)')
    ax.set_ylabel('RMSE (mph)')
    ax.set_title('v2 — RMSE by Horizon')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_rmse.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention(attns):
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
    ax.set_title('v2 — Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(rmse, mae, per_horizon_rmse, history, best_val, viol):
    with open(os.path.join(OUT_DIR, "metrics.txt"), 'w') as f:
        f.write("Physics v2 (occupancy-based 3-penalty)\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Test RMSE: {rmse:.4f} mph\n")
        f.write(f"Test MAE:  {mae:.4f} mph\n\n")
        f.write("Per-horizon RMSE:\n")
        for h in range(OUTPUT_WINDOW):
            f.write(f"  t+{(h+1)*5:>2} min: {per_horizon_rmse[h]:.4f}\n")
        f.write(f"\nBest val total loss: {best_val:.6f}\n")
        f.write(f"Trained epochs: {len(history['train_total'])}/{EPOCHS}\n\n")
        f.write("Config:\n")
        f.write(f"  seed = {SEED}  (baseline-v4 architecture)\n")
        f.write(f"  input_window = {INPUT_WINDOW}, output_window = {OUTPUT_WINDOW}\n")
        f.write(f"  hidden = {HIDDEN_SIZE}, layers = {NUM_LAYERS}, dropout = {DROPOUT}\n")
        f.write(f"  features = {len(FEATURE_COLS)}\n")
        f.write(f"  batch = {BATCH_SIZE}, epochs = {EPOCHS}, lr = {LR}\n")
        f.write(f"  weight_decay = {WEIGHT_DECAY}\n")
        f.write(f"  early_stopping_patience = {PATIENCE}, min_delta = {MIN_DELTA}\n\n")
        f.write("Physics constraints:\n")
        f.write(f"  acc_limit = {ACC_LIMIT} mph/5min  (lambda = {LAMBDA_ACC})\n")
        f.write(f"  speed_range = [{SPEED_MIN}, {SPEED_MAX}] mph  (lambda = {LAMBDA_SPEED})\n")
        f.write(f"  occ_congestion: occ > {OCC_CONG} => v <= {V_CONG} mph  (lambda = {LAMBDA_CONG})\n\n")
        f.write("Test-set constraint violation rates:\n")
        f.write(f"  acc:  {viol['acc_violation_rate']*100:.3f}% of step-pairs\n")
        f.write(f"  range:{viol['speed_violation_rate']*100:.3f}% of predictions\n")
        f.write(f"  cong: {viol['cong_violation_rate']*100:.3f}% of congested samples "
                f"(n_cong={viol['n_congested_samples']:,})\n")


def save_violation_stats(viol):
    with open(os.path.join(OUT_DIR, "violation_stats.txt"), 'w') as f:
        f.write("v2 — Test-set Constraint Violation Stats\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Acceleration violations  (|dv| > {ACC_LIMIT} mph): "
                f"{viol['acc_violation_rate']*100:.3f}%\n")
        f.write(f"Speed-range violations   (v<{SPEED_MIN} or v>{SPEED_MAX}): "
                f"{viol['speed_violation_rate']*100:.3f}%\n")
        f.write(f"Occ-congestion violations (occ>{OCC_CONG} & v>{V_CONG}): "
                f"{viol['cong_violation_rate']*100:.3f}%\n")
        f.write(f"  (of {viol['n_congested_samples']:,} congested samples)\n")


# Main
if __name__ == "__main__":
    set_seed(SEED)
    print(f"Device: {DEVICE}  Seed: {SEED}")
    train_loader, val_loader, test_loader, speed_mean, speed_std = prepare_data()

    model = GRUAttention(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_window=OUTPUT_WINDOW,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history, best_val = train_model(model, train_loader, val_loader, speed_mean, speed_std)
    preds, truths, attns, rmse, mae, per_horizon_rmse, viol = evaluate(
        model, test_loader, speed_mean, speed_std
    )

    plot_loss_curve(history)
    plot_physics_losses(history)
    plot_pred_vs_truth(truths, preds)
    plot_horizon_rmse(per_horizon_rmse)
    plot_attention(attns)
    save_metrics(rmse, mae, per_horizon_rmse, history, best_val, viol)
    save_violation_stats(viol)

    print(f"\nTest RMSE: {rmse:.4f} mph")
    print(f"Test MAE:  {mae:.4f} mph")
    print(f"Outputs -> {OUT_DIR}/")
