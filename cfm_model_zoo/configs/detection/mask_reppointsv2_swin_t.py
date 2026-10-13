backend_args = None
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
data = dict(
    train=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            scale=[
                (
                    1333,
                    480,
                ),
                (
                    1333,
                    960,
                ),
            ],
            keep_ratio=True,
            multiscale_mode='range',
            type='Resize'),
        dict(flip_ratio=0.5, type='RandomFlip'),
        dict(
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            to_rgb=True,
            type='Normalize'),
        dict(size_divisor=32, type='Pad'),
        dict(type='LoadRPDV2Annotations'),
        dict(type='RPDV2FormatBundle'),
        dict(
            keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
                'gt_sem_map',
                'gt_sem_weights',
                'gt_masks',
            ],
            type='Collect'),
    ]))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
dist_params = dict(backend='nccl')
evaluation = dict(metric=[
    'bbox',
    'segm',
])
fp16 = None
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        27,
        33,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7,
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    bbox_head=dict(
        coord_pos='center',
        corner_dim=64,
        feat_channels=256,
        first_kernel_size=3,
        gradient_mul=0.1,
        in_channels=256,
        kernel_size=1,
        loss_bbox_init=dict(loss_weight=1.0, type='RPDGIoULoss'),
        loss_bbox_refine=dict(loss_weight=2.0, type='RPDGIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='RPDQualityFocalLoss',
            use_sigmoid=True),
        loss_heatmap=dict(
            alpha=2.0, gamma=4.0, loss_weight=0.25, type='LegacyGaussianFocalLoss'),
        loss_offset=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_sem=dict(
            alpha=0.25, gamma=2.0, loss_weight=0.1, type='SEPFocalLoss'),
        mask_head=dict(
            branch_cfg=dict(
                channels=128,
                in_channels=256,
                in_features=[
                    0,
                    1,
                    2,
                ],
                loss_sem=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    prior_prob=0.01,
                    type='FocalLoss',
                    use_sigmoid=True),
                norm=dict(requires_grad=True, type='BN'),
                num_classes=80,
                num_convs=4,
                out_channels=8,
                out_stride=[
                    8,
                    16,
                    32,
                ],
                semantic_loss_on=False),
            head_cfg=dict(
                aux_loss=True,
                channels=8,
                disable_rel_coords=False,
                in_channels=8,
                mask_loss_weight=[
                    0.0,
                    0.6,
                    1.0,
                ],
                mask_out_stride=4,
                max_proposals=500,
                num_layers=3,
                rel_num=1,
                sizes_of_interest=[
                    64,
                    128,
                    256,
                    512,
                    1024,
                ],
                use_fp16=False),
            type='CondConvMaskHead'),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=80,
        num_points=9,
        point_base_scale=4,
        point_feat_channels=256,
        point_strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        shared_stacked_convs=1,
        stacked_convs=3,
        test_cfg=dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.05),
        train_cfg=dict(
            heatmap=dict(
                allowed_border=-1,
                assigner=dict(
                    gaussian_bump=True,
                    gaussian_iou=0.7,
                    type='PointHMAssigner'),
                debug=False,
                pos_weight=-1),
            init=dict(
                allowed_border=-1,
                assigner=dict(
                    mask_center_sample=True,
                    pos_num=1,
                    scale=4,
                    type='PointAssignerV2',
                    use_center=True),
                debug=False,
                pos_weight=-1),
            refine=dict(
                allowed_border=-1,
                assigner=dict(
                    mask_center_sample=True, topk=9, type='ATSSAssignerV2'),
                debug=False,
                pos_weight=-1)),
        transform_method='exact_minmax',
        type='RepPointsV2Head'),
    neck=dict(
        add_extra_convs=False,
        in_channels=[
            192,
            384,
            768,
        ],
        no_norm_on_lateral=False,
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_outs=5,
        num_repeat=2,
        out_channels=256,
        start_level=0,
        type='BiFPN'),
    init_cfg=None,
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        heatmap=dict(
            allowed_border=-1,
            assigner=dict(
                gaussian_bump=True, gaussian_iou=0.7, type='PointHMAssigner'),
            debug=False,
            pos_weight=-1),
        init=dict(
            allowed_border=-1,
            assigner=dict(
                mask_center_sample=True,
                pos_num=1,
                scale=4,
                type='PointAssignerV2',
                use_center=True),
            debug=False,
            pos_weight=-1),
        refine=dict(
            allowed_border=-1,
            assigner=dict(
                mask_center_sample=True, topk=9, type='ATSSAssignerV2'),
            debug=False,
            pos_weight=-1)),
    type='RepPointsV2MaskDetector')
norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='AdamW',
    weight_decay=0.05)
optimizer_config = dict(
    bucket_size_mb=-1,
    coalesce=True,
    grad_clip=None,
    type='DistOptimizerHook',
    update_interval=1,
    use_fp16=True)
resume_from = None
runner = dict(max_epochs=36, type='EpochBasedRunnerAmp')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='LoadRPDV2Annotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='NewPackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    'data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='NewPackDetInputs'),
]
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='LoadRPDV2Annotations'),
            dict(type='RandomResize',scale= (1024,1024), ratio_range=(0.5, 2.0),keep_ratio=True),
            dict(crop_size=(
                256,
                256,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='NewPackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadRPDV2Annotations'),
    dict(
        scale=[
            (
                1333,
                480,
            ),
            (
                1333,
                960,
            ),
        ],
        keep_ratio=True,
        multiscale_mode='range',
        type='Resize'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='LoadRPDV2Annotations'),
    dict(type='RPDV2FormatBundle'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'gt_sem_map',
            'gt_sem_weights',
            'gt_masks',
        ],
        type='Collect'),
]
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='LoadRPDV2Annotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='NewPackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
workflow = [
    (
        'train',
        1,
    ),
]