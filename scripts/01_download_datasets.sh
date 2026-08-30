#!/usr/bin/env bash
# Download all three datasets to data/. Idempotent (skips already-present trees).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

# 1) PlantVillage — already present locally at /home/csdf/Apple/Apple leaf dataset/color
#    (4 classes: Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy
#     with 630/621/275/1645 images respectively — matches paper-reported counts).
LOCAL_PV="/home/csdf/Apple/Apple leaf dataset/color"
if [ -d "$LOCAL_PV/Apple___Apple_scab" ]; then
  echo "[1/3] PlantVillage found at $LOCAL_PV (skipping download)."
elif [ ! -d "$DATA/plantvillage" ]; then
  echo "[1/3] Downloading PlantVillage from Mendeley..."
  curl -L -o "$DATA/pv.zip" "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
  unzip -q "$DATA/pv.zip" -d "$DATA/plantvillage"
  rm "$DATA/pv.zip"
else
  echo "[1/3] PlantVillage already present at $DATA/plantvillage, skipping."
fi

# 2) Plant Pathology 2021 (Kaggle competition).
#    Requires kaggle CLI authenticated (~/.kaggle/kaggle.json present).
if [ ! -d "$DATA/plant-pathology-2021" ]; then
  echo "[2/3] Downloading Plant Pathology 2021 via kaggle CLI..."
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: kaggle CLI not installed. Run: pip install kaggle, then place kaggle.json in ~/.kaggle/."
    exit 1
  fi
  kaggle competitions download -c plant-pathology-2021-fgvc8 -p "$DATA"
  unzip -q "$DATA/plant-pathology-2021-fgvc8.zip" -d "$DATA/plant-pathology-2021"
  rm "$DATA/plant-pathology-2021-fgvc8.zip"
else
  echo "[2/3] Plant Pathology 2021 already present, skipping."
fi

# 3) AppleLeaf9.
if [ ! -d "$DATA/appleleaf9" ]; then
  echo "[3/3] Cloning AppleLeaf9..."
  git clone --depth 1 https://github.com/JasonYangCode/AppleLeaf9.git "$DATA/appleleaf9"
else
  echo "[3/3] AppleLeaf9 already present, skipping."
fi

echo "Done. Datasets under: $DATA"
