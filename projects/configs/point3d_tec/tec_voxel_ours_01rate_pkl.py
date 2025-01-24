_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]    # voxel
# voxel_size = [0.2, 0.2, 8]    # pillar

# imagenet setting
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

use_LiDAR=True
use_Cam=True

sample_rate = 0.1
rate_str = str(sample_rate).replace(".", "")

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# velocity ×
without_vel = True
# GT point init
point_init = True

# point noise
point_noise=True
noise_density=1/3
norm_std=0.3

model = dict(
    type='MMPointObjDGCNNV4',
    use_LiDAR=use_LiDAR,
    use_Cam=use_Cam,
    use_grid_mask=use_Cam,
    # camera
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    # lidar voxel refer: mmdetection3d/configs/_base_/models/centerpoint_01voxel_second_secfpn_nus.py
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size, max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        num_outs=4),
    # head
    pts_bbox_head=dict(
        type='MMPointDGCNN3DHeadV4',
        num_query=300,
        num_classes=10,
        in_channels=256,
        point_init=point_init,
        without_vel=without_vel,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        point_encoder=dict(
            type='RotPoint3DEncoder',
            num_classes=10,
            ),
        transformer=dict(
            type='PointFUTR3DTransformerV4',
            point_init=point_init,
            use_encoder=False,
            use_LiDAR=use_LiDAR,
            use_Cam=use_Cam,
            # only for lidar
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=2,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='FUTR3DTransformerDecoderV4',
                num_layers=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='FUTR3DDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='Deform3DRoIWiseMultiModalCrossAttnV2',
                            pc_range=point_cloud_range,
                            use_LiDAR=use_LiDAR,
                            use_Cam=use_Cam,
                            img_roi=True,
                            pts_roi=True,
                            img_roi_offset=True,
                            pts_roi_offset=True,
                            combine_lvl_dim=True,
                            rotate_bev_grid=True,
                            num_cam_points=16,
                            num_lid_points=16,
                            embed_dims=256,
                            ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm'))
                )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)), # For DETR compatibility.
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[1024, 1024, 40],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=8,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'PointNuScenesDataset'
data_root = 'data/nuscenes/'

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

train_pipeline = [
    # l1
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFilesV2', 
         to_float32=True,
         file_client_args=file_client_args),  # c1
    
    # l2
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),    # l3
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),    # l4
    dict(type='ObjectNameFilter', classes=class_names),         # l5
    # LoadPointAnnotations
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),

    dict(type='PointShuffle'),      # l6
    
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),           # c3
    dict(type='PadMultiViewImage', size_divisor=32),                # c4
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='MMPointCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','point_coord','labels','img'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFilesV2', 
         to_float32=True,
        file_client_args=file_client_args),  # c1
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    # ------------------------------------
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # LoadPointAnnotations
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),
    # ------------------------------------
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),           # c3
    dict(type='PadMultiViewImage', size_divisor=32),                # c4
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='MMPointCollect3D', keys=['points','point_coord','labels','img'])
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
    dict(type='LoadMultiViewImageFromFilesV2', 
         to_float32=True,
         file_client_args=file_client_args),  # c1
    
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),           # c3
    dict(type='PadMultiViewImage', size_divisor=32),                # c4
    
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

    # ------------------------------------
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # LoadPointAnnotations
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),
    # ------------------------------------

    dict(type='MMPointCollect3D', keys=['points','point_coord','labels','img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            without_vel=without_vel, 
            gt_ratio=sample_rate, # partial data
            load_interval=1,
            box_type_3d='LiDAR',
    ),
    val=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names, modality=input_modality,ann_file=data_root + 'nuscenes_infos_train.pkl',),
    test=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names, modality=input_modality,ann_file=data_root + 'nuscenes_infos_train.pkl',))

evaluation = dict(interval=20, pipeline=eval_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=20)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_voxel_encoder': dict(lr_mult=0.1),
            'SECOND': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

load_from = f'ckpts/voxel_R50_{rate_str}.pth'   # voxel_R50_01
checkpoint_config = dict(interval=5, max_keep_ckpts=1)
find_unused_parameters = True