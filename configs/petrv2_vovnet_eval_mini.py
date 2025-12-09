import os

_base_ = ['./petrv2_vovnet_gridmask_p4_800x320.py']

data_root = os.environ.get('DATA_ROOT', os.path.join(os.path.dirname(__file__), '..', 'data-mini'))

data = dict(
    val=dict(
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
    ),
    test=dict(
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
    ),
)

