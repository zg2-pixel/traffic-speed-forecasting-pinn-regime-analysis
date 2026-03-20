"""
Step 3: Physics-Informed GRU Model (v2 - with warmup + rebalanced weights)
============================================================================
Same GRU architecture as baseline, but with three physics constraints
added to the loss function with warmup scheduling:
  1. Acceleration constraint: |v(t) - v(t-1)| <= a_max
  2. Speed range constraint:  0 <= v(t) <= v_max
  3. Congestion-speed consistency: high flow => low speed

Changes from v1:
  - Reduced LAMBDA_CONG from 0.05 to 0.005
  - Added warmup: first 10 epochs data-only, then ramp up physics over 10 epochs

Outputs in outputs_physics_v2/:
  - loss_curve_physics_v2.png
  - prediction_vs_truth_physics_v2*.png
  - metrics_physics_v2.txt
  - constraint_losses_v2.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration

DATA_PATH = "data/processed/traffic_clean.csv"
OUT_DIR = "outputs_physics_v2"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_WINDOW = 12
OUTPUT_WINDOW = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

FEATURE_COLS = [
    'avg_speed', 'total_flow', 'avg_occupancy',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'temperature', 'precipitation', 'wind_speed', 'weather_code'
]
TARGET_COL = 'avg_speed'
FLOW_COL = 'total_flow'

TRAIN_DAYS = 22
VAL_DAYS = 3
TEST_DAYS = 6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Physics constraint parameters

A_MAX_MPH = 15.0
V_MAX_MPH = 85.0
V_MIN_MPH = 0.0
FLOW_CONGESTION = 180
V_CONGESTION_MPH = 50.0

# Rebalanced weights (v2: LAMBDA_CONG reduced from 0.05 to 0.005)
LAMBDA_ACC = 0.1
LAMBDA_SPEED = 0.1
LAMBDA_CONG = 0.005

# Warmup schedule
WARMUP_EPOCHS = 10    # first 10 epochs: data loss only
RAMPUP_EPOCHS = 10    # epochs 11-20: gradually increase physics weight from 0 to 1


# Dataset

class TrafficDatasetPhysics(Dataset):
    def __init__(self, features, targets, flow, input_window, output_window):
        self.features = features
        self.targets = targets
        self.flow = flow
        self.input_window = input_window
        self.output_window = output_window
        self.total_len = len(features) - input_window - output_window + 1

    def __len__(self):
        return max(0, self.total_len)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.input_window]
        y = self.targets[idx + self.input_window: idx + self.input_window + self.output_window]
        f = self.flow[idx + self.input_window: idx + self.input_window + self.output_window]
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(f)


# Model

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_window, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_window)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Physics-informed loss functions

def compute_physics_losses(pred, flow, speed_mean, speed_std):
    pred_mph = pred * speed_std + speed_mean

    # Constraint 1: Acceleration limit
    if pred_mph.shape[1] > 1:
        speed_diff = torch.abs(pred_mph[:, 1:] - pred_mph[:, :-1])
        acc_violation = torch.clamp(speed_diff - A_MAX_MPH, min=0)
        L_acc = acc_violation.mean()
    else:
        L_acc = torch.tensor(0.0, device=pred.device)

    # Constraint 2: Speed range
    below_min = torch.clamp(V_MIN_MPH - pred_mph, min=0)
    above_max = torch.clamp(pred_mph - V_MAX_MPH, min=0)
    L_speed = (below_min + above_max).mean()

    # Constraint 3: Congestion-speed consistency
    is_congested = (flow > FLOW_CONGESTION).float()
    cong_violation = torch.clamp(pred_mph - V_CONGESTION_MPH, min=0) * is_congested
    L_cong = cong_violation.mean()

    return L_acc, L_speed, L_cong


# Data preparation

def prepare_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)

    stations = sorted(df['station'].unique())
    print(f"Stations: {stations}")

    dates = sorted(df['timestamp'].dt.date.unique())
    train_end = dates[TRAIN_DAYS - 1]
    val_end = dates[TRAIN_DAYS + VAL_DAYS - 1]

    original_flow = df[FLOW_COL].values.copy()

    train_mask = df['timestamp'].dt.date <= train_end
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_COLS])

    speed_idx = FEATURE_COLS.index(TARGET_COL)
    speed_mean = scaler.mean_[speed_idx]
    speed_std = scaler.scale_[speed_idx]

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    train_datasets, val_datasets, test_datasets = [], [], []
    test_meta = []

    for sid in stations:
        sdf = df[df['station'] == sid].copy().reset_index(drop=True)
        sdf_idx = df[df['station'] == sid].index

        features = sdf[FEATURE_COLS].values
        targets = sdf[TARGET_COL].values
        flow_orig = original_flow[sdf_idx]

        train_idx = sdf['timestamp'].dt.date <= train_end
        val_idx = (sdf['timestamp'].dt.date > train_end) & (sdf['timestamp'].dt.date <= val_end)
        test_idx = sdf['timestamp'].dt.date > val_end

        for mask, ds_list in [(train_idx, train_datasets), (val_idx, val_datasets), (test_idx, test_datasets)]:
            f = features[mask]
            t = targets[mask]
            fl = flow_orig[mask]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                ds_list.append(TrafficDatasetPhysics(f, t, fl, INPUT_WINDOW, OUTPUT_WINDOW))

        test_f = features[test_idx]
        test_t = targets[test_idx]
        test_fl = flow_orig[test_idx]
        test_ts = sdf.loc[test_idx, 'timestamp'].values
        test_meta.append({
            'station': sid, 'features': test_f, 'targets': test_t,
            'flow': test_fl, 'timestamps': test_ts
        })

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_meta, speed_mean, speed_std


# Training with physics constraints + warmup

def train_model(model, train_loader, val_loader, speed_mean, speed_std):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    train_physics = {'acc': [], 'speed': [], 'cong': []}
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # ── Warmup schedule ──
        if epoch < WARMUP_EPOCHS:
            physics_weight = 0.0
        else:
            physics_weight = min(1.0, (epoch - WARMUP_EPOCHS) / float(RAMPUP_EPOCHS))

        # Train
        model.train()
        epoch_data_loss = 0
        epoch_acc_loss = 0
        epoch_speed_loss = 0
        epoch_cong_loss = 0
        n_samples = 0

        for x, y, flow in train_loader:
            x, y, flow = x.to(DEVICE), y.to(DEVICE), flow.to(DEVICE)

            pred = model(x)

            # Data loss
            L_data = criterion(pred, y)

            # Physics losses
            L_acc, L_speed, L_cong = compute_physics_losses(pred, flow, speed_mean, speed_std)

            # Total loss with warmup
            loss = L_data + physics_weight * (
                LAMBDA_ACC * L_acc + LAMBDA_SPEED * L_speed + LAMBDA_CONG * L_cong
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x.size(0)
            epoch_data_loss += L_data.item() * bs
            epoch_acc_loss += L_acc.item() * bs
            epoch_speed_loss += L_speed.item() * bs
            epoch_cong_loss += L_cong.item() * bs
            n_samples += bs

        train_loss = epoch_data_loss / n_samples
        train_losses.append(train_loss)
        train_physics['acc'].append(epoch_acc_loss / n_samples)
        train_physics['speed'].append(epoch_speed_loss / n_samples)
        train_physics['cong'].append(epoch_cong_loss / n_samples)

        # Validate (data loss only)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, flow in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model_physics_v2.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Data: {train_loss:.6f}, "
                  f"Acc: {epoch_acc_loss/n_samples:.4f}, "
                  f"Cong: {epoch_cong_loss/n_samples:.4f}, "
                  f"Val: {val_loss:.6f}, "
                  f"PhysW: {physics_weight:.2f}")

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_physics_v2.pt"), weights_only=True))

    return train_losses, val_losses, train_physics


# Evaluation

def evaluate_model(model, test_loader, speed_mean, speed_std):
    model.eval()
    all_preds, all_truths = [], []

    with torch.no_grad():
        for x, y, flow in test_loader:
            x = x.to(DEVICE)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_truths.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_truths, axis=0)

    preds = preds * speed_std + speed_mean
    truths = truths * speed_std + speed_mean
    preds = np.clip(preds, 0, 100)

    rmse = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae = mean_absolute_error(truths.flatten(), preds.flatten())

    print(f"\n{'Horizon':<12} {'RMSE':>8} {'MAE':>8}")
    print("-" * 30)
    for h in range(OUTPUT_WINDOW):
        h_rmse = np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        h_mae = mean_absolute_error(truths[:, h], preds[:, h])
        print(f"  t+{(h+1)*5:>2} min   {h_rmse:8.2f} {h_mae:8.2f}")
    print("-" * 30)
    print(f"  Overall    {rmse:8.2f} {mae:8.2f}")

    # Count constraint violations
    speed_diff = np.abs(np.diff(preds, axis=1))
    n_acc_violations = np.sum(speed_diff > A_MAX_MPH)
    n_speed_violations = np.sum((preds < V_MIN_MPH) | (preds > V_MAX_MPH))

    print(f"\nConstraint violations on test set:")
    print(f"  Acceleration violations (>{A_MAX_MPH} mph change): {n_acc_violations}")
    print(f"  Speed range violations: {n_speed_violations}")

    return preds, truths, rmse, mae


# Plotting

def plot_loss_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss (data only)', linewidth=1.5)
    ax.plot(val_losses, label='Validation Loss', linewidth=1.5)
    ax.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Physics warmup start')
    ax.axvline(x=WARMUP_EPOCHS + RAMPUP_EPOCHS, color='gray', linestyle=':', alpha=0.5, label='Physics fully active')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Physics-Informed GRU (v2) — Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve_physics_v2.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_curve_physics_v2.png")


def plot_physics_losses(train_physics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(train_physics['acc'], color='#e74c3c', linewidth=1.5)
    axes[0].set_title('Acceleration penalty')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.5)

    axes[1].plot(train_physics['speed'], color='#3498db', linewidth=1.5)
    axes[1].set_title('Speed range penalty')
    axes[1].set_xlabel('Epoch')
    axes[1].axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.5)

    axes[2].plot(train_physics['cong'], color='#2ecc71', linewidth=1.5)
    axes[2].set_title('Congestion penalty')
    axes[2].set_xlabel('Epoch')
    axes[2].axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.5)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Physics Constraint Losses During Training (v2, dashed = warmup end)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "constraint_losses_v2.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: constraint_losses_v2.png")


def plot_predictions(test_meta, model, speed_mean, speed_std):
    model.eval()

    for meta in test_meta[:3]:
        sid = meta['station']
        features = meta['features']
        targets = meta['targets']
        timestamps = meta['timestamps']

        preds_list, truth_list, time_list = [], [], []
        n = len(features)
        step = OUTPUT_WINDOW

        for i in range(0, n - INPUT_WINDOW - OUTPUT_WINDOW + 1, step):
            x = torch.FloatTensor(features[i:i + INPUT_WINDOW]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(x).cpu().numpy()[0]

            pred = pred * speed_std + speed_mean
            truth = targets[i + INPUT_WINDOW: i + INPUT_WINDOW + OUTPUT_WINDOW]
            truth = truth * speed_std + speed_mean
            ts = timestamps[i + INPUT_WINDOW: i + INPUT_WINDOW + OUTPUT_WINDOW]

            preds_list.extend(pred)
            truth_list.extend(truth)
            time_list.extend(ts)

        if not time_list:
            continue

        n_points = min(len(time_list), 288 * 2)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_list[:n_points], truth_list[:n_points],
                label='Ground Truth', color='#1f77b4', linewidth=1)
        ax.plot(time_list[:n_points], preds_list[:n_points],
                label='Physics-Informed (v2)', color='#ff7f0e', linewidth=1, alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Speed (mph)')
        ax.set_title(f'Station {sid} — Physics v2 Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"prediction_vs_truth_physics_{sid}_v2.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: prediction_vs_truth_physics_{sid}_v2.png")


def save_metrics(rmse, mae, train_losses, val_losses):
    with open(os.path.join(OUT_DIR, "metrics_physics_v2.txt"), 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Physics-Informed GRU Model (v2) — Test Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: GRU ({NUM_LAYERS} layers, hidden={HIDDEN_SIZE})\n")
        f.write(f"Input window: {INPUT_WINDOW} steps ({INPUT_WINDOW * 5} min)\n")
        f.write(f"Output window: {OUTPUT_WINDOW} steps ({OUTPUT_WINDOW * 5} min)\n\n")
        f.write("Physics constraints:\n")
        f.write(f"  Acceleration: a_max = {A_MAX_MPH} mph/5min, lambda = {LAMBDA_ACC}\n")
        f.write(f"  Speed range: [{V_MIN_MPH}, {V_MAX_MPH}] mph, lambda = {LAMBDA_SPEED}\n")
        f.write(f"  Congestion: flow > {FLOW_CONGESTION} => speed <= {V_CONGESTION_MPH} mph, lambda = {LAMBDA_CONG}\n\n")
        f.write(f"Warmup: {WARMUP_EPOCHS} epochs data-only, then ramp up over {RAMPUP_EPOCHS} epochs\n\n")
        f.write(f"Test RMSE: {rmse:.4f} mph\n")
        f.write(f"Test MAE:  {mae:.4f} mph\n\n")
        f.write(f"Final train loss (data): {train_losses[-1]:.6f}\n")
        f.write(f"Final val loss:          {val_losses[-1]:.6f}\n")
        f.write(f"Best val loss:           {min(val_losses):.6f}\n")
    print("Saved: metrics_physics_v2.txt")


# Main

if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Train Physics-Informed GRU (v2 - warmup + rebalanced)")
    print("=" * 60)

    print(f"\nPhysics constraint parameters:")
    print(f"  Acceleration limit: {A_MAX_MPH} mph per 5-min step")
    print(f"  Speed range: [{V_MIN_MPH}, {V_MAX_MPH}] mph")
    print(f"  Congestion: flow > {FLOW_CONGESTION} => speed <= {V_CONGESTION_MPH} mph")
    print(f"  Weights: lambda_acc={LAMBDA_ACC}, lambda_speed={LAMBDA_SPEED}, lambda_cong={LAMBDA_CONG}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs data-only, ramp up over {RAMPUP_EPOCHS} epochs")

    # 1. Prepare data
    print("\n[1/4] Preparing data...")
    train_loader, val_loader, test_loader, test_meta, speed_mean, speed_std = prepare_data()

    # 2. Train
    print(f"\n[2/4] Training Physics-Informed GRU v2 ({EPOCHS} epochs)...")
    model = GRUModel(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_window=OUTPUT_WINDOW,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    train_losses, val_losses, train_physics = train_model(
        model, train_loader, val_loader, speed_mean, speed_std
    )

    # 3. Evaluate
    print("\n[3/4] Evaluating on test set...")
    preds, truths, rmse, mae = evaluate_model(model, test_loader, speed_mean, speed_std)

    # 4. Plots
    print("\n[4/4] Generating plots...")
    plot_loss_curve(train_losses, val_losses)
    plot_physics_losses(train_physics)
    plot_predictions(test_meta, model, speed_mean, speed_std)
    save_metrics(rmse, mae, train_losses, val_losses)

    print("\n" + "=" * 60)
    print("Done! Check outputs_physics_v2/ for results.")
    print("Compare with baseline results in outputs_baseline/")
    print("=" * 60)