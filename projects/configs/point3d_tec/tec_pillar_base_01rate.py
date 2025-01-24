_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/schedules/cyclic_20e.py', 
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

use_LiDAR=True
use_Cam=False

sample_rate = 0.1
rate_str = str(sample_rate).replace(".", "")

input_modality = dict(
    use_lidar=True,
    use_camera=False,
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
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        point_cloud_range=point_cloud_range,
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=4),
    pts_bbox_head=dict(
        type='PointDGCNN3DHead',
        num_query=300,
        num_classes=10,
        in_channels=256,
        point_init=point_init,
        without_vel=without_vel,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        point_encoder=dict(
            type='PointEncoder2D',
            num_classes=10,
            num_feats=128,
            pos='query_embedding',
            ),
        transformer=dict(
            type='MyDeformableDetrTransformer',
            point_init=point_init,
            use_encoder=False,
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
                type='Deformable3DDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='DGCNNAttn',
                            embed_dims=256,
                            num_heads=8,
                            K=16,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
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
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
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
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),
    
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='MMPointCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','point_coord', 'labels'])
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
    # ------------------------------------
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # LoadPointAnnotations
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),
    # ------------------------------------
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
            dict(type='MMPointCollect3D', keys=['points','point_coord','labels'])
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

    # ------------------------------------
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # LoadPointAnnotations
    dict(type='LoadPointAnnotations', point_noise=point_noise, noise_density=noise_density,
         norm_type='truncated_normal', norm_mean_std=[0.5, norm_std]),
    # ------------------------------------
    dict(type='MMPointCollect3D', keys=['points','point_coord','labels'])
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
    val=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names, modality=input_modality,ann_file=data_root + 'nuscenes_infos_val.pkl',),
    test=dict(type=dataset_type, without_vel=without_vel, pipeline=test_pipeline, classes=class_names, modality=input_modality,ann_file=data_root + 'nuscenes_infos_val.pkl',))

evaluation = dict(interval=20, pipeline=eval_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=20)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'pts_voxel_encoder': dict(lr_mult=0.1),
            'SECOND': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
load_from = f'ckpts/pillar_R50_{rate_str}.pth'   # voxel_R50_01
checkpoint_config = dict(interval=5, max_keep_ckpts=1)
find_unused_parameters = True