import os

_base_ = ['configs/bevdepth/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.py']

data_root = os.environ.get('DATA_ROOT', os.path.join(os.path.dirname(__file__), '..', 'data-mini'))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_train.pkl'),
    ),
    val=dict(
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
    ),
    test=dict(
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
    ),
)

