#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate method-comp
cd "$ROOT/third_party/PETR"
python tools/train.py projects/configs/petrv2/petrv2_vovnet_eval_mini.py --work-dir "$ROOT/outputs/petrv2"

