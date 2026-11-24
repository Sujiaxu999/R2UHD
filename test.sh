#!/usr/bin/env bash
# Test R2UHD: whole-frame inference, plus PSNR/SSIM/LPIPS when GT is given.
#
# Every argument is forwarded to inference_r2uhd.py, so both styles work:
#
#   bash test.sh -w /path/to/r2uhd_haze.pth -i /path/to/input -g /path/to/gt -o results/haze
#   WEIGHT=/path/to/r2uhd_haze.pth INPUT=/path/to/input GT=/path/to/gt bash test.sh
#
# Optional environment variables (command-line flags win because they come last):
#   WEIGHT  -w   checkpoint
#   INPUT   -i   input images or directory
#   GT      -g   ground truth for metrics
#   CONFIG  -c   training YAML (default: the single YAML beside the checkpoint)
#   OUTPUT  -o   result folder
#   DEVICE       cuda (default) or cpu
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON_BIN:-python}"
ARGS=()
if [ -n "${WEIGHT:-}" ]; then ARGS+=(-w "$WEIGHT"); fi
if [ -n "${INPUT:-}" ]; then ARGS+=(-i "$INPUT"); fi
if [ -n "${GT:-}" ]; then ARGS+=(-g "$GT"); fi
if [ -n "${CONFIG:-}" ]; then ARGS+=(-c "$CONFIG"); fi
if [ -n "${OUTPUT:-}" ]; then ARGS+=(-o "$OUTPUT"); fi
if [ -n "${DEVICE:-}" ]; then ARGS+=(--device "$DEVICE"); fi

exec "$PYTHON" inference_r2uhd.py "${ARGS[@]}" "$@"
