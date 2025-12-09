#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate method-comp
cd "$ROOT/third_party/3DPPE"
python tools/train.py projects/configs/petrv2_depth/3dppe_r50_mini.py --work-dir "$ROOT/outputs/3dppe"

