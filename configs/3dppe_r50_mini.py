import os

_base_ = ['./petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py']

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

