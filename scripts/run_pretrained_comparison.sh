#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate method-comp
python "$ROOT/scripts/generate_pretrained_comparison.py"

