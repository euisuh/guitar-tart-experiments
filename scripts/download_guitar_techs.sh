#!/usr/bin/env bash
# Downloads Guitar-TECHS technique + singlenote files from Zenodo.
# Only fetches files needed for technique classification (~950 MB total).
set -euo pipefail

DEST="data/raw/guitar-techs"
mkdir -p "$DEST"

BASE="https://zenodo.org/records/14963133/files"

FILES=(
  "P1_techniques.zip"
  "P2_techniques.zip"
  "P1_singlenotes.zip"
  "P2_singlenotes.zip"
)

for f in "${FILES[@]}"; do
  if [ -d "$DEST/${f%.zip}" ]; then
    echo "Already extracted: ${f%.zip}, skipping."
    continue
  fi
  echo "Downloading $f..."
  curl -L --progress-bar "$BASE/$f?download=1" -o "$DEST/$f"
  echo "Extracting $f..."
  unzip -q "$DEST/$f" -d "$DEST"
  rm "$DEST/$f"
  echo "Done: $f"
done

echo ""
echo "Generating annotations CSV..."
/opt/homebrew/Caskroom/miniconda/base/bin/python data/download.py --data-dir "$DEST"
echo ""
echo "Setup complete. Dataset ready at $DEST/"
