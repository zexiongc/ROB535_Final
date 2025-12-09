# Quickstart (Method Comparison, 3D Position Encoding)

## 1) Clone
```bash
git clone <your_repo_url> method-comparison
cd method-comparison
```

## 2) Prepare data
- Download nuScenes mini to any path (contains `samples`, `sweeps`, `nuscenes_infos_val.pkl`, etc.)
- Link it:
```bash
./scripts/setup_data.sh /path/to/nuscenes_mini
```

## 3) Create environment
```bash
./scripts/create_env.sh
```

## 4) Bootstrap code (clone upstream repos and copy configs)
```bash
./scripts/bootstrap.sh
```

## 5) Checkpoints
- Put official pretrain weights into `checkpoints/`:
  - `petrv2_vovnet.pth`
  - `3dppe_vovnet.pth`
- (Optional) `bevdepth_r50.pth` if you have it

## 6) Train (optional, uses nuScenes mini)
```bash
./scripts/train_petrv2.sh
./scripts/train_3dppe.sh
./scripts/train_bevdepth.sh
```

## 7) Generate qualitative comparison + accuracy table
```bash
./scripts/run_pretrained_comparison.sh
```
Outputs land in `outputs/`:
- `pe_comparison_pretrained.png`
- `accuracy_summary.txt`

## 8) Where things live
- `third_party/` upstream code (PETR, 3DPPE, BEVDepth)
- `checkpoints/` pretrained weights
- `data-mini/` symlink to your nuScenes mini
- `outputs/` results and figures

## Notes
- Use `conda activate method-comp` before training or running scripts.
- If you want full nuScenes, update `DATA_ROOT` env and run the same steps.
