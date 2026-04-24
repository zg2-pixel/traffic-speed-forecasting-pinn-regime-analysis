"""
v3_attention.py

Baseline v3: GRU + temporal attention + weight decay + LR scheduler.
Dropout raised to 0.3. Per-station sliding windows retained.
"""

import os
import copy
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
OUT_DIR = "baseline_output/v3_attention_output"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_WINDOW = 12
OUTPUT_WINDOW = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
PATIENCE = 15

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


# Dataset
class StationDataset(Dataset):
    def __init__(self, features, targets, input_window, output_window):
        self.X, self.y = [], []
        n = len(features)
        for i in range(n - input_window - output_window + 1):
            self.X.append(features[i:i + input_window])
            self.y.append(targets[i + input_window:i + input_window + output_window])
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        tgt = sdf[TARGET_COL].values

        s_train = sdf['timestamp'].dt.date <= train_end
        s_val = (sdf['timestamp'].dt.date > train_end) & (sdf['timestamp'].dt.date <= val_end)
        s_test = sdf['timestamp'].dt.date > val_end

        for mask, lst in [(s_train, train_list), (s_val, val_list), (s_test, test_list)]:
            f, t = feat[mask], tgt[mask]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                lst.append(StationDataset(f, t, INPUT_WINDOW, OUTPUT_WINDOW))

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


# Training
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    mse_fn = nn.MSELoss()
    train_losses, val_losses, lr_history = [], [], []
    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred, _ = model(x)
            loss = mse_fn(pred, y)
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
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred, _ = model(x)
                running += mse_fn(pred, y).item() * x.size(0)
                n += x.size(0)
        val_loss = running / n

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            tag = " *"
        else:
            wait += 1
            tag = ""

        if (epoch + 1) % 5 == 0 or tag:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={current_lr:.2e}{tag}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, "best_model.pt"))
    return train_losses, val_losses, lr_history, best_val


# Evaluation
def evaluate(model, test_loader, speed_mean, speed_std):
    model.eval()
    preds, truths, attns = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred, attn_w = model(x.to(DEVICE))
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
            attns.append(attn_w.cpu().numpy())
    preds = np.clip(np.concatenate(preds) * speed_std + speed_mean, 0, 100)
    truths = np.concatenate(truths) * speed_std + speed_mean
    attns = np.concatenate(attns)

    rmse = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae = mean_absolute_error(truths.flatten(), preds.flatten())
    per_horizon_rmse = [
        np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        for h in range(OUTPUT_WINDOW)
    ]
    return preds, truths, attns, rmse, mae, per_horizon_rmse


# Plotting
def plot_loss_curve(train_losses, val_losses, lr_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(train_losses)+1), train_losses, label='Train', linewidth=1.5)
    ax1.plot(range(1, len(val_losses)+1), val_losses, label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('v3 — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(lr_history)+1), lr_history, color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('v3 — LR Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_truth(truths, preds):
    n = min(500, len(preds))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(truths[:n, 0], label='Ground truth', linewidth=1.2)
    axes[0].plot(preds[:n, 0], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title('v3 — Prediction vs Truth (t+5 min)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(truths[:n, -1], label='Ground truth', linewidth=1.2)
    axes[1].plot(preds[:n, -1], label='Prediction', linewidth=1.2, alpha=0.85)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Speed (mph)')
    axes[1].set_title('v3 — Prediction vs Truth (t+30 min)')
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
    ax.set_title('v3 — RMSE by Horizon')
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
    ax.set_xlabel('Input time step (1=oldest, 12=newest)')
    ax.set_ylabel('Attention weight')
    ax.set_title('v3 — Temporal Attention')
    ax.set_xticks(range(1, INPUT_WINDOW+1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attention.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(rmse, mae, per_horizon_rmse, train_losses, val_losses, best_val):
    with open(os.path.join(OUT_DIR, "metrics.txt"), 'w') as f:
        f.write("Baseline v3 (GRU + temporal attention + regularization)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Test RMSE: {rmse:.4f} mph\n")
        f.write(f"Test MAE:  {mae:.4f} mph\n\n")
        f.write("Per-horizon RMSE:\n")
        for h in range(OUTPUT_WINDOW):
            f.write(f"  t+{(h+1)*5:>2} min: {per_horizon_rmse[h]:.4f}\n")
        f.write(f"\nBest val loss: {best_val:.6f}\n")
        f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final val loss: {val_losses[-1]:.6f}\n")
        f.write(f"Trained epochs: {len(train_losses)}/{EPOCHS}\n\n")
        f.write("Config:\n")
        f.write(f"  input_window = {INPUT_WINDOW}, output_window = {OUTPUT_WINDOW}\n")
        f.write(f"  hidden = {HIDDEN_SIZE}, layers = {NUM_LAYERS}, dropout = {DROPOUT}\n")
        f.write(f"  batch = {BATCH_SIZE}, epochs = {EPOCHS}, lr = {LR}\n")
        f.write(f"  weight_decay = {WEIGHT_DECAY}\n")
        f.write(f"  early_stopping_patience = {PATIENCE}\n")
        f.write(f"  scheduler = ReduceLROnPlateau(factor=0.5, patience=7)\n")
        f.write(f"  features = {len(FEATURE_COLS)}\n")


# Main
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader, speed_mean, speed_std = prepare_data()

    model = GRUAttention(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_window=OUTPUT_WINDOW,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_losses, val_losses, lr_history, best_val = train_model(
        model, train_loader, val_loader
    )
    preds, truths, attns, rmse, mae, per_horizon_rmse = evaluate(
        model, test_loader, speed_mean, speed_std
    )

    plot_loss_curve(train_losses, val_losses, lr_history)
    plot_pred_vs_truth(truths, preds)
    plot_horizon_rmse(per_horizon_rmse)
    plot_attention(attns)
    save_metrics(rmse, mae, per_horizon_rmse, train_losses, val_losses, best_val)

    print(f"\nTest RMSE: {rmse:.4f} mph")
    print(f"Test MAE:  {mae:.4f} mph")
    print(f"Outputs -> {OUT_DIR}/")
