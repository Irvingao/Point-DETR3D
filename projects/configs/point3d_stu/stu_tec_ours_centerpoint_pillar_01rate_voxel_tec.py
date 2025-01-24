_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/models/centerpoint_02pillar_second_secfpn_nus.py',
    '../../../mmdetection3d/configs/_base_/schedules/cyclic_20e.py', 
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

model = dict(
    type='WSStudentCenterPointV3',    # ws student type
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2]),
        obj_contrast=False, 
        feat_contrast=True, 
        loss_feat_contrast=dict(type='ContrastiveNegCosLoss', reduction='mean', loss_weight=0.2),
        point_guide=True,
        point_size_factor=2,
        type='WSCenterHead'),
    loss_type="V5",
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range,
        # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),    # original
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0., 0.])),    # set vel to zero.
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

# spnds
without_vel = True

sample_rate = 0.1
rate_str = str(sample_rate).replace(".", "")

ann_file=f'/mnt/lustrenew/gaohongzhi/work_dirs/point_objDGCNN/point3d_tec/tec_voxel_ours_{rate_str}rate/nuscenes_gt_pseudo_dbinfos_train_{rate_str}rate.pkl'

dataset_type = 'WSStudentSampledNuScenesDataset'
data_root = 'data/nuscenes/'
# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/nuscenes/': 'czh:s3://czhBucket/nuscenes/',
        'data/nuscenes/': 'czh:s3://czhBucket/nuscenes/',
    }))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + f'nuscenes_sampled_{rate_str}ratio_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

strong_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.85, 1.15],
                translation_std=[0.5, 0.5, 0.5]),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointShuffle'),
            dict(type="ExtraAttrs", tag="view_1"),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                # meta_keys=['tag']
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'tag', 'label_flag')
            )
        ]
    )
]

weak_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointShuffle'),
            dict(type="ExtraAttrs", tag="view_2"),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'tag', 'label_flag')
            )
        ]
    )
]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type="MultiBranch", view_1=strong_pipeline, view_2=weak_pipeline,
    ),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_file,
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            without_vel=without_vel, 
            gt_ratio=sample_rate,
            load_interval=1,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names),
    test=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names))

evaluation = dict(interval=20, pipeline=eval_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
