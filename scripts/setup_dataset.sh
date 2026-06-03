#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_URL="${1:-}"
DEST="data/raw/guitar-techs"

if [ -z "$DOWNLOAD_URL" ]; then
  echo "Usage: bash scripts/setup_dataset.sh <GUITAR_TECHS_DOWNLOAD_URL>"
  echo "Get the URL from: https://guitar-techs.github.io/"
  exit 1
fi

mkdir -p "$DEST"

echo "Downloading Guitar-TECHS dataset..."
curl -L "$DOWNLOAD_URL" -o "$DEST/guitar-techs.zip"

echo "Extracting..."
unzip -q "$DEST/guitar-techs.zip" -d "$DEST"
rm "$DEST/guitar-techs.zip"

echo "Generating annotations CSV..."
python data/download.py --data-dir "$DEST"

echo "Done. Annotations written to $DEST/annotations.csv"
