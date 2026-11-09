#!/usr/bin/env bash
set -euo pipefail

REPO="/content/los80"
DRIVE_ROOT="/content/drive/MyDrive/LOS80"
CACHE_ROOT="$DRIVE_ROOT/CACHE"

if [ ! -d "$DRIVE_ROOT" ]; then
    echo "ERROR: Mount Google Drive first."
    exit 1
fi

mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

cd "$REPO"
chmod +x colab_bootstrap.sh

echo "=== LOS80 resume ==="
echo "Persistent model cache: $CACHE_ROOT"

./colab_bootstrap.sh

echo
echo "=== Starting/resuming production pipeline ==="
exec env PYTHONPATH="$REPO" python run.py --config config/colab-production.yaml
