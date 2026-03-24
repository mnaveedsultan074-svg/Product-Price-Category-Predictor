#!/bin/bash
set -e

echo "=== Price Predictor ML Pipeline ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Downloading / preparing data..."
python download_data.py

echo "[2/3] Preprocessing..."
python preprocess.py

echo "[3/3] Training models..."
python train.py

echo "=== Pipeline complete. Starting prediction service... ==="
python predict_service.py
