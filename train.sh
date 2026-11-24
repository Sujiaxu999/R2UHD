#!/usr/bin/env bash
# Train R2UHD.
#
#   bash train.sh                                          # default haze recipe, one GPU
#   bash train.sh options/train_r2uhd_snow.yml
#   bash train.sh options/train_r2uhd_snow.yml --auto_resume
#   bash train.sh --auto_resume                            # default recipe, resume
#   CONFIG=options/train_r2uhd_ll.yml bash train.sh --debug
#   NPROC_PER_NODE=4 bash train.sh options/train_r2uhd_blur.yml
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh options/train_r2uhd_ll.yml
#
# The first non-option argument is the option file; everything else is
# forwarded to basicsr.train (--auto_resume, --debug, --force_yml ...).
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-options/train_r2uhd_haze.yml}"
if [ "$#" -gt 0 ] && [ "${1:0:1}" != "-" ]; then
    CONFIG="$1"
    shift
fi

PYTHON="${PYTHON_BIN:-python}"
NPROC="${NPROC_PER_NODE:-1}"

if [ "$NPROC" -gt 1 ]; then
    exec "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NPROC" \
        --master_port="${MASTER_PORT:-4398}" \
        --module basicsr.train \
        -opt "$CONFIG" --launcher pytorch "$@"
fi

exec "$PYTHON" -m basicsr.train -opt "$CONFIG" --launcher none "$@"
