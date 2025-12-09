#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=method-comp
source ~/miniconda3/etc/profile.d/conda.sh
if conda info --envs | grep -q "^${ENV_NAME} "; then
  conda remove -n ${ENV_NAME} --all -y
fi
conda create -n ${ENV_NAME} python=3.8 -y
conda activate ${ENV_NAME}
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full==1.6.0
pip install mmdet==2.25.0 mmdet3d==1.0.0rc6
pip install nuscenes-devkit pyquaternion einops timm yapf addict opencv-python pillow scipy tqdm matplotlib seaborn pandas
pip install mmsegmentation==0.29.0 mmcls==0.25.0
pip install pytorch-lightning==1.5.10

