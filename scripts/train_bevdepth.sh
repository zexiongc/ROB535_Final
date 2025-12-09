#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate method-comp
cd "$ROOT/third_party/BEVDepth"
python tools/train.py configs/bevdepth_mini.py --work-dir "$ROOT/outputs/bevdepth"

