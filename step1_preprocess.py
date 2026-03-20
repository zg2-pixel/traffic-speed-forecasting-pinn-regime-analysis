"""
Step 1: Preprocess PeMS raw data + download weather data

Reads all .txt.gz files from data/raw/, filters to 10 selected I-880 S stations,
downloads hourly weather from Open-Meteo, merges everything into a clean CSV.

Output: data/processed/traffic_clean.csv
"""

import os
import glob
import gzip
import csv
import json
import urllib.request
import ssl
from datetime import datetime

import pandas as pd
import numpy as np

# Configuration

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# 10 selected stations on I-880 Southbound (San Jose → Fremont)
SELECTED_STATIONS = {
    '402121', '423148', '400238', '402291', '401637',
    '407221', '400409', '400363', '400468', '400576'
}

# Weather API (Open-Meteo, no key needed)
WEATHER_LAT = 37.46
WEATHER_LON = -121.94
WEATHER_START = "2025-01-01"
WEATHER_END = "2025-01-31"

# Column names for PeMS Station 5-Minute data (first 12 fields)
PEMS_COLUMNS = [
    'timestamp', 'station', 'district', 'freeway', 'direction',
    'lane_type', 'station_length', 'samples', 'pct_observed',
    'total_flow', 'avg_occupancy', 'avg_speed'
]



# 1. Parse PeMS raw data

def parse_pems_files():
    """Read all .txt.gz files and extract data for selected stations."""
    # Support both .txt.gz and .txt files
    gz_files = sorted(glob.glob(os.path.join(RAW_DIR, "d04_text_station_5min_2025_01_*.txt.gz")))
    txt_files = sorted(glob.glob(os.path.join(RAW_DIR, "d04_text_station_5min_2025_01_*.txt")))
    data_files = gz_files if gz_files else txt_files

    if not data_files:
        print(f"ERROR: No data files found in {RAW_DIR}/")
        print("Please download PeMS data and place files in data/raw/")
        return None

    is_gzip = bool(gz_files)
    print(f"Found {len(data_files)} data files ({'gzip' if is_gzip else 'plain text'})")
    all_rows = []

    for fpath in data_files:
        fname = os.path.basename(fpath)
        count = 0
        opener = gzip.open(fpath, 'rt', encoding='utf-8', errors='replace') if is_gzip \
                 else open(fpath, 'r', encoding='utf-8', errors='replace')
        with opener as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12:
                    continue
                station_id = row[1].strip()
                if station_id in SELECTED_STATIONS:
                    all_rows.append(row[:12])
                    count += 1
        print(f"  {fname}: {count} records for selected stations")

    df = pd.DataFrame(all_rows, columns=PEMS_COLUMNS)

    # Type conversions
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip(), format='%m/%d/%Y %H:%M:%S')
    df['station'] = df['station'].str.strip().astype(int)
    for col in ['total_flow', 'samples']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    for col in ['avg_speed', 'avg_occupancy', 'pct_observed', 'station_length']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

    df = df.sort_values(['station', 'timestamp']).reset_index(drop=True)
    print(f"\nTotal records: {len(df)}")
    print(f"Stations: {sorted(df['station'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


# 2. Download weather data

def download_weather():
    """Download hourly weather from Open-Meteo API."""
    weather_file = os.path.join(OUT_DIR, "weather.csv")

    if os.path.exists(weather_file):
        print(f"Weather file already exists: {weather_file}")
        return pd.read_csv(weather_file, parse_dates=['datetime'])

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
        f"&start_date={WEATHER_START}&end_date={WEATHER_END}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,"
        f"snowfall,weather_code,wind_speed_10m"
        f"&timezone=America/Los_Angeles"
    )

    print(f"Downloading weather data from Open-Meteo...")

    # Handle SSL certificate issues on Mac
    try:
        context = ssl.create_default_context()
        response = urllib.request.urlopen(url, context=context)
    except ssl.SSLCertVerificationError:
        print("  SSL certificate issue detected, using unverified context...")
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(url, context=context)

    data = json.loads(response.read().decode())
    hourly = data["hourly"]

    weather_df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'temperature': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'precipitation': hourly['precipitation'],
        'rain': hourly['rain'],
        'snowfall': hourly['snowfall'],
        'weather_code': hourly['weather_code'],
        'wind_speed': hourly['wind_speed_10m'],
    })

    weather_df.to_csv(weather_file, index=False)
    print(f"  Saved {len(weather_df)} hourly records to {weather_file}")

    return weather_df


# 3. Add time features and merge weather

def add_features(df, weather_df):
    """Add temporal features and merge weather data."""

    # Time features (cyclical encoding)
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = df['timestamp'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['is_weekend'] = (dow >= 5).astype(int)

    # Merge weather (align by hour)
    df['hour_key'] = df['timestamp'].dt.floor('h')
    weather_df['hour_key'] = weather_df['datetime'].dt.floor('h')

    weather_cols = ['hour_key', 'temperature', 'humidity', 'precipitation',
                    'rain', 'snowfall', 'weather_code', 'wind_speed']
    df = df.merge(weather_df[weather_cols], on='hour_key', how='left')
    df = df.drop(columns=['hour_key'])

    # Fill any missing weather values (forward fill then back fill)
    weather_fill_cols = ['temperature', 'humidity', 'precipitation', 'rain',
                         'snowfall', 'weather_code', 'wind_speed']
    df[weather_fill_cols] = df[weather_fill_cols].ffill().bfill()

    return df


# 4. Data quality checks

def clean_data(df):
    """Basic data quality filtering."""
    before = len(df)

    # Remove rows where speed is 0 or negative (sensor error)
    df = df[df['avg_speed'] > 0].copy()

    # Remove rows where observed% is very low (mostly imputed)
    df = df[df['pct_observed'] >= 50].copy()

    # Cap speed at reasonable max (100 mph for freeway)
    df['avg_speed'] = df['avg_speed'].clip(upper=100)

    after = len(df)
    print(f"Data cleaning: {before} → {after} records ({before - after} removed)")

    return df


# Main

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Preprocessing PeMS + Weather Data")
    print("=" * 60)

    # 1. Parse PeMS
    print("\n[1/4] Parsing PeMS raw data...")
    df = parse_pems_files()
    if df is None:
        exit(1)

    # 2. Download weather
    print("\n[2/4] Getting weather data...")
    weather_df = download_weather()

    # 3. Add features & merge
    print("\n[3/4] Adding features and merging weather...")
    df = add_features(df, weather_df)

    # 4. Clean
    print("\n[4/4] Cleaning data...")
    df = clean_data(df)

    # Save
    out_path = os.path.join(OUT_DIR, "traffic_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Stations: {sorted(df['station'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSpeed stats:")
    print(df['avg_speed'].describe())
    print(f"\nFlow stats:")
    print(df['total_flow'].describe())
