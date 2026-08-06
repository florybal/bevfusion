_base_ = [
    '../../../configs/_base_/default_runtime.py'
]

METAINFO = {
        'classes': (
            'obstrucao', 'empilhadeira', 'carga', 'maquina',
            'humano', 'navegavel', 'estrutura', 'portapalete'
        ),
            'palette': [
            [255, 0, 0],       # obstrucao
            [255, 140, 0],     # empilhadeira
            [139, 43, 226],    # carga
            [0, 0, 255],       # maquina
            [0, 170, 255],     # humano
            [0, 255, 0],       # navegavel
            [255, 255, 0],     # estrutura
            [205, 133, 63],    # portapalete
        ]
    }

metainfo = dict(
    classes=METAINFO['classes'],
    palette=METAINFO['palette']
)

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',
        'mmdet3d.evaluation.metrics.plastipak_metric'
    ],
    allow_failed_imports=False
)

# ===== Dataset =====
dataset_type = 'PlastipakDataset'
data_root = '/workspace/official_mmdet3d/data/BEVLOG/finetunning/'
ann_file_train = '/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_train.pkl'
if ann_file_train is None:
    raise ValueError("O arquivo de anotação de treinamento não foi especificado. Por favor, defina 'ann_file' corretamente.")
ann_file_val = '/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_val.pkl'
ann_file_test = '/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_test.pkl'

classes = ('obstrucao', 'empilhadeira', 'carga', 'maquina',
           'humano', 'navegavel', 'estrutura', 'portapalete')

point_cloud_range= [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None


# ===== modelo =====
model = dict(
    type='BEVFusion',
    use_camera=True,
    data_preprocessor=dict(
        pad_size_divisor=32,
        type='Det3DDataPreprocessor',
        bgr_to_rgb=False,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        voxelize_cfg=dict(
            max_num_points=10,
            max_voxels=[120000, 160000],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            voxelize_reduce=True)
    ),
    pts_voxel_encoder=dict(
        num_features=4,
        type='HardSimpleVFE'
    ),
    pts_middle_encoder=dict(
        block_type='basicblock',
        in_channels=4,
        sparse_shape=[1440, 1440, 41],
        encoder_channels=(
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        encoder_paddings=(
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, (1, 1, 0)),
            (0, 0),
        ),
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        order=('conv', 'norm', 'act'),
        type='BEVFusionSparseEncoder'
    ),
    pts_backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)
        ),
        pts_neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True
        ),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
    ),
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256,
    ),
    bbox_head=dict(
            type='TransFusionHead',
            num_proposals=200,           # <-- CORREÇÃO 
            auxiliary=True,               # <-- CORREÇÃO 
            in_channels=512,
            hidden_channel=128,           # <-- ADICIONADO (bate com embed_dims do decoder_layer)
            num_classes=8,
            nms_kernel_size=3,
            bn_momentum=0.1,              # <-- CORREÇÃO
            num_decoder_layers=1,         # <-- CORREÇÃO 
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                code_size=8,   # <-- 8 dimensões: x, y, z, w, l, h, sin(yaw), cos(yaw)
                out_size_factor=8,
                pc_range=[-54.0, -54.0],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                voxel_size=[0.075, 0.075]
            ),
            common_heads=dict(
                center=[2, 2],
                dim=[3, 2],
                height=[1, 2],
                rot=[2, 2],
                # NÃO inclua 'vel'
            ),
            decoder_layer=dict(
                type='TransformerDecoderLayer',
                self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)
                ),
                norm_cfg=dict(type='LN'),
                pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
            ),
            loss_heatmap=dict(
                type='mmdet.GaussianFocalLoss',
                loss_weight=1.0,
                reduction='mean'
            ),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction='mean'
            ),
            loss_bbox=dict(
                type='mmdet.L1Loss',
                loss_weight=0.25,
                reduction='mean'
            ),
        train_cfg=dict(
            dataset='plastikpak',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            point_cloud_range=point_cloud_range,
            min_radius=2,
            gaussian_overlap=0.1,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8 elementos
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    alpha=0.25,
                    gamma=2.0,
                    weight=0.15
                ),
                reg_cost=dict(
                    type='BBoxBEVL1Cost',
                    weight=0.25
                ),
                iou_cost=dict(
                    type='IoU3DCost',
                    weight=0.25
                ),
                iou_calculator=dict(
                    type='BboxOverlaps3D',
                    coordinate='lidar'
                )
            )
        ),
        test_cfg=dict(
            dataset='plastikpak',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            nms_type=None,
        )
    )
)

# ===== Pipelines =====
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],  # Pega apenas X, Y, Z, Intensidade
        backend_args=backend_args),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55],
             bot_pct_lim=[0.0, 0.0], rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1],
         rot_range=[-0.78539816, 0.78539816], translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    #dict(type='ObjectNameFilter', classes=classes), #pode dar erro se não tiver as classes ou o metainfo 
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=6,
         rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                    'lidar_path', 'img_path', 'num_pts_feats']),
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55],
         bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                    'lidar_path', 'img_path', 'num_pts_feats']),
]

# ===== DATALOADERS =====
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='PlastipakDataset',
        data_root=data_root,
        ann_file=ann_file_train,
        metainfo = metainfo,
        pipeline=train_pipeline,
        test_mode=False,
        box_type_3d = 'LiDAR',
        modality=input_modality,
    ),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),

)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='PlastipakDataset',
        data_root=data_root,
        ann_file=ann_file_val,
        metainfo = metainfo,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo = metainfo,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        test_mode=True,
        modality=input_modality,
    ),
)

val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='PlastipakMetric',
    ann_file=ann_file_val,
    metric='bbox',
)
test_evaluator = dict(
    type='PlastipakMetric',
    ann_file=ann_file_test,
    metric='bbox',
)

# ===== OPTMIZER =====
lr = 1e-4

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.01),
    clip_grad=dict(
        max_norm=35,
        norm_type=2)
)

# ===== LEARNING RATE =====
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=lr * 1e-4,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ===== RUNTIME =====
train_cfg = dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5),
)

log_processor = dict(window_size=50)

auto_scale_lr = dict(
    enable=False,
    base_batch_size=8)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')