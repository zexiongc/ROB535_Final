#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate method-comp
mkdir -p "$ROOT/third_party" "$ROOT/checkpoints" "$ROOT/outputs"
cd "$ROOT/third_party"
if [ ! -d PETR ]; then
  git clone https://github.com/megvii-research/PETR.git
fi
if [ ! -d 3DPPE ]; then
  git clone https://github.com/drilistbox/3DPPE.git
fi
if [ ! -d BEVDepth ]; then
  git clone https://github.com/Megvii-BaseDetection/BEVDepth.git
fi
cp "$ROOT/configs/petrv2_vovnet_eval_mini.py" "$ROOT/third_party/PETR/projects/configs/petrv2/petrv2_vovnet_eval_mini.py"
cp "$ROOT/configs/3dppe_r50_mini.py" "$ROOT/third_party/3DPPE/projects/configs/petrv2_depth/3dppe_r50_mini.py"
cp "$ROOT/configs/bevdepth_mini.py" "$ROOT/third_party/BEVDepth/configs/bevdepth_mini.py"

