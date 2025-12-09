#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.."; pwd)
DATA_SRC=${1:-"$ROOT/data-mini"}
mkdir -p "$ROOT"
ln -sfn "$DATA_SRC" "$ROOT/data-mini"
ln -sfn "$DATA_SRC" "$ROOT/third_party/PETR/data-mini"
ln -sfn "$DATA_SRC" "$ROOT/third_party/3DPPE/data-mini"
ln -sfn "$DATA_SRC" "$ROOT/third_party/BEVDepth/data-mini"

