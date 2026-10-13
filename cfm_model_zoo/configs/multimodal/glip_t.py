_base_ = './glip_base.py'

model = dict(bbox_head=dict(early_fuse=True, use_checkpoint=True))

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth'  # noqa

optim_wrapper = dict(
    optimizer=dict(lr=0.00001),
    clip_grad=dict(_delete_=True, max_norm=1, norm_type=2))