"""
Step 2: Train GRU Baseline Model

Loads preprocessed data, trains a GRU model to predict speed,
and generates all plots/metrics required for the milestone.

Outputs in outputs_baseline/:
  - loss_curve_baseline.png
  - prediction_vs_truth_baseline_XXXXX.png (multiple)
  - metrics_baseline.txt
  - data_overview_baseline.png
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
OUT_DIR = "outputs_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

# Model hyperparameters
INPUT_WINDOW = 12    # past 12 steps (1 hour) as input
OUTPUT_WINDOW = 6    # predict next 6 steps (30 min)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# Features used as input
FEATURE_COLS = [
    'avg_speed', 'total_flow', 'avg_occupancy',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'temperature', 'precipitation', 'wind_speed', 'weather_code'
]
TARGET_COL = 'avg_speed'

# Train/val/test split by days (22/3/6 days)
TRAIN_DAYS = 22
VAL_DAYS = 3
TEST_DAYS = 6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# Dataset

class TrafficDataset(Dataset):
    """Sliding window dataset for one station."""
    def __init__(self, features, targets, input_window, output_window):
        self.features = features
        self.targets = targets
        self.input_window = input_window
        self.output_window = output_window
        self.total_len = len(features) - input_window - output_window + 1

    def __len__(self):
        return max(0, self.total_len)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.input_window]
        y = self.targets[idx + self.input_window: idx + self.input_window + self.output_window]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# Model

class GRUModel(nn.Module):
    """GRU-based sequence-to-sequence model for speed prediction."""
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
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take last hidden state
        out = self.fc(out)   # (batch, output_window)
        return out


# Data preparation

def prepare_data():
    """Load CSV, split by time, create datasets per station."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    # Sort
    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)

    stations = sorted(df['station'].unique())
    print(f"Stations: {stations}")

    # Determine split dates
    dates = sorted(df['timestamp'].dt.date.unique())
    train_end = dates[TRAIN_DAYS - 1]
    val_end = dates[TRAIN_DAYS + VAL_DAYS - 1]

    print(f"Train: up to {train_end} ({TRAIN_DAYS} days)")
    print(f"Val:   up to {val_end} ({VAL_DAYS} days)")
    print(f"Test:  remaining ({TEST_DAYS} days)")

    # Feature scaling (fit on training data only)
    train_mask = df['timestamp'].dt.date <= train_end
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_COLS])

    # Save original data for plotting before scaling
    df_original = df.copy()

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    # Also need to track the speed scaler separately for inverse transform
    speed_idx = FEATURE_COLS.index(TARGET_COL)
    speed_mean = scaler.mean_[speed_idx]
    speed_std = scaler.scale_[speed_idx]

    # Build datasets per station, then combine
    train_datasets, val_datasets, test_datasets = [], [], []
    test_meta = []  # store (station, timestamps) for plotting

    for sid in stations:
        sdf = df[df['station'] == sid].copy().reset_index(drop=True)
        features = sdf[FEATURE_COLS].values
        targets = sdf[TARGET_COL].values

        # Split indices by date
        train_idx = sdf['timestamp'].dt.date <= train_end
        val_idx = (sdf['timestamp'].dt.date > train_end) & (sdf['timestamp'].dt.date <= val_end)
        test_idx = sdf['timestamp'].dt.date > val_end

        for mask, ds_list in [(train_idx, train_datasets), (val_idx, val_datasets), (test_idx, test_datasets)]:
            f = features[mask]
            t = targets[mask]
            if len(f) > INPUT_WINDOW + OUTPUT_WINDOW:
                ds_list.append(TrafficDataset(f, t, INPUT_WINDOW, OUTPUT_WINDOW))

        # Save test info for plotting
        test_f = features[test_idx]
        test_t = targets[test_idx]
        test_ts = sdf.loc[test_idx, 'timestamp'].values
        test_meta.append({
            'station': sid,
            'features': test_f,
            'targets': test_t,
            'timestamps': test_ts
        })

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_meta, speed_mean, speed_std, df_original


# Training

def train_model(model, train_loader, val_loader):
    """Train the GRU model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model_baseline.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_baseline.pt"), weights_only=True))

    return train_losses, val_losses


# Evaluation

def evaluate_model(model, test_loader, speed_mean, speed_std):
    """Evaluate on test set, return predictions and ground truth (original scale)."""
    model.eval()
    all_preds, all_truths = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_truths.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_truths, axis=0)

    # Inverse transform (speed was standardized)
    preds = preds * speed_std + speed_mean
    truths = truths * speed_std + speed_mean

    # Clip predictions to valid range
    preds = np.clip(preds, 0, 100)

    # Metrics (average over all horizons)
    rmse = np.sqrt(mean_squared_error(truths.flatten(), preds.flatten()))
    mae = mean_absolute_error(truths.flatten(), preds.flatten())

    # Per-horizon metrics
    print(f"\n{'Horizon':<12} {'RMSE':>8} {'MAE':>8}")
    print("-" * 30)
    for h in range(OUTPUT_WINDOW):
        h_rmse = np.sqrt(mean_squared_error(truths[:, h], preds[:, h]))
        h_mae = mean_absolute_error(truths[:, h], preds[:, h])
        print(f"  t+{(h+1)*5:>2} min   {h_rmse:8.2f} {h_mae:8.2f}")
    print("-" * 30)
    print(f"  Overall    {rmse:8.2f} {mae:8.2f}")

    return preds, truths, rmse, mae


# Plotting

def plot_data_overview(df):
    """Plot representative data figure (required for milestone)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Pick one station, one week
    sid = df['station'].unique()[0]
    sdf = df[df['station'] == sid].copy()
    week = sdf[(sdf['timestamp'] >= '2025-01-06') & (sdf['timestamp'] < '2025-01-13')]

    axes[0].plot(week['timestamp'], week['avg_speed'], color='#1f77b4', linewidth=0.6)
    axes[0].set_ylabel('Speed (mph)')
    axes[0].set_title(f'Station {sid} — One week of traffic data (Jan 6-12, 2025)')

    axes[1].plot(week['timestamp'], week['total_flow'], color='#ff7f0e', linewidth=0.6)
    axes[1].set_ylabel('Flow (veh/5min)')

    axes[2].plot(week['timestamp'], week['avg_occupancy'], color='#2ca02c', linewidth=0.6)
    axes[2].set_ylabel('Occupancy')
    axes[2].set_xlabel('Time')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "data_overview_baseline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: data_overview_baseline.png")


def plot_loss_curve(train_losses, val_losses):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss', linewidth=1.5)
    ax.plot(val_losses, label='Validation Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Baseline GRU — Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve_baseline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: loss_curve_baseline.png")


def plot_predictions(test_meta, model, speed_mean, speed_std):
    """Plot prediction vs ground truth for several stations."""
    model.eval()

    for meta in test_meta[:3]:  # plot first 3 stations
        sid = meta['station']
        features = meta['features']
        targets = meta['targets']
        timestamps = meta['timestamps']

        # Generate predictions for a continuous segment
        preds_list = []
        truth_list = []
        time_list = []

        n = len(features)
        step = OUTPUT_WINDOW  # non-overlapping predictions
        for i in range(0, n - INPUT_WINDOW - OUTPUT_WINDOW + 1, step):
            x = torch.FloatTensor(features[i:i + INPUT_WINDOW]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(x).cpu().numpy()[0]

            # Inverse transform
            pred = pred * speed_std + speed_mean
            truth = targets[i + INPUT_WINDOW: i + INPUT_WINDOW + OUTPUT_WINDOW]
            truth = truth * speed_std + speed_mean
            ts = timestamps[i + INPUT_WINDOW: i + INPUT_WINDOW + OUTPUT_WINDOW]

            preds_list.extend(pred)
            truth_list.extend(truth)
            time_list.extend(ts)

        if not time_list:
            continue

        # Plot first 2 days
        n_points = min(len(time_list), 288 * 2)  # 2 days
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_list[:n_points], truth_list[:n_points],
                label='Ground Truth', color='#1f77b4', linewidth=1)
        ax.plot(time_list[:n_points], preds_list[:n_points],
                label='Prediction', color='#ff7f0e', linewidth=1, alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Speed (mph)')
        ax.set_title(f'Station {sid} — Baseline Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"prediction_vs_truth_baseline_{sid}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: prediction_vs_truth_baseline_{sid}.png")


def save_metrics(rmse, mae, train_losses, val_losses):
    """Save metrics to text file."""
    with open(os.path.join(OUT_DIR, "metrics_baseline.txt"), 'w') as f:
        f.write("=" * 40 + "\n")
        f.write("Baseline GRU Model — Test Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: GRU ({NUM_LAYERS} layers, hidden={HIDDEN_SIZE})\n")
        f.write(f"Input window: {INPUT_WINDOW} steps ({INPUT_WINDOW * 5} min)\n")
        f.write(f"Output window: {OUTPUT_WINDOW} steps ({OUTPUT_WINDOW * 5} min)\n")
        f.write(f"Optimizer: Adam (lr={LEARNING_RATE})\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n\n")
        f.write(f"Test RMSE: {rmse:.4f} mph\n")
        f.write(f"Test MAE:  {mae:.4f} mph\n\n")
        f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final val loss:   {val_losses[-1]:.6f}\n")
        f.write(f"Best val loss:    {min(val_losses):.6f}\n")
    print("Saved: metrics_baseline.txt")


# Main

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Train GRU Baseline")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/5] Preparing data...")
    train_loader, val_loader, test_loader, test_meta, speed_mean, speed_std, df = prepare_data()

    # 2. Plot data overview
    print("\n[2/5] Plotting data overview...")
    plot_data_overview(df)

    # 3. Train model
    print(f"\n[3/5] Training GRU model ({EPOCHS} epochs)...")
    model = GRUModel(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_window=OUTPUT_WINDOW,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    train_losses, val_losses = train_model(model, train_loader, val_loader)

    # 4. Evaluate
    print("\n[4/5] Evaluating on test set...")
    preds, truths, rmse, mae = evaluate_model(model, test_loader, speed_mean, speed_std)

    # 5. Generate plots
    print("\n[5/5] Generating plots...")
    plot_loss_curve(train_losses, val_losses)
    plot_predictions(test_meta, model, speed_mean, speed_std)
    save_metrics(rmse, mae, train_losses, val_losses)

    print("\n" + "=" * 60)
    print("Done! Check the outputs_baseline/ folder for all results.")
    print("=" * 60)
