#!/usr/bin/env bash
set -euo pipefail

cd /content/los80

echo "=== LOS80 Colab bootstrap ==="

if [ ! -d "/content/drive/MyDrive/LOS80" ]; then
    echo "ERROR: Google Drive LOS80 directory is not mounted."
    exit 1
fi

echo "[1/4] Installing Python runtime dependencies..."
python -m pip install -q \
    faster-whisper \
    transformers \
    sentencepiece \
    accelerate

echo "[2/4] Checking production config..."
test -f config/colab-production.yaml

echo "[3/4] Checking NVIDIA/NVENC..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc; then
    echo "ERROR: FFmpeg HEVC NVENC is unavailable."
    exit 1
fi

echo "[4/4] Checking LOS80..."
PYTHONPATH=/content/los80 python - <<'PY'
from los80.encoder import FFmpegBackend

encoder = FFmpegBackend().detect_hardware_encoder("auto")
print("LOS80 encoder:", encoder)

if encoder != "hevc_nvenc":
    raise SystemExit(f"Expected hevc_nvenc, got {encoder}")
PY

echo
echo "✅ LOS80 COLAB READY"
echo "Run:"
echo "PYTHONPATH=/content/los80 python run.py --config config/colab-production.yaml"
