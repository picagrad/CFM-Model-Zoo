import numpy as np

import torch
from torch import nn
import mmcv
import hashlib
import os
import shutil
import tempfile
from collections import OrderedDict
from mmengine.runner import CheckpointLoader
from mmdet.structures.bbox import bbox2result, bbox_mapping_back, bbox_flip
from mmdet.registry import MODELS
from mmdet.models import SingleStageDetector
import torch.nn.functional as F
from mmdet.models.utils import aligned_bilinear
from mmdet.structures import DetDataSample
# from .reppoints import RepPointsV2Head
from mmengine.structures import InstanceData as Instances
from ..split_utils import SplitSwinFeatures


from mmcv.cnn import ConvModule
from mmengine.model import xavier_init



def swish(x):
    return x * x.sigmoid()


class LayerCombineModule(nn.Module):
    def __init__(self, num_input=2):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(num_input, dtype=torch.float32).view(1, 1, 1, 1, -1),
            requires_grad=True
        )

    def forward(self, inputs):

        weights = self.weights.relu()
        norm_weights = weights / (weights.sum() + 0.0001)

        out = (norm_weights*torch.stack(inputs, dim=-1)).sum(dim=-1)
        return swish(out)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x 

class SingleBiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, no_norm_on_lateral=True, conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super().__init__()

        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.lateral_combine = nn.ModuleList()
        self.lateral_combine_conv = nn.ModuleList()
        self.out_combine = nn.ModuleList()
        self.out_combine_conv = nn.ModuleList()

        for i, in_channel in enumerate(in_channels):
            if in_channel != out_channels:
                self.lateral_convs.append(ConvModule(
                    in_channel,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False))
            else:
                self.lateral_convs.append(Identity())
            if i != len(in_channels)-1:
                self.lateral_combine.append(LayerCombineModule(2))
                self.lateral_combine_conv.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False)
                )
            if i != 0:
                self.out_combine.append(LayerCombineModule(
                    3 if i != len(in_channels)-1 else 2))
                self.out_combine_conv.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False))

    def forward(self, inputs):

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals = laterals + \
            inputs[len(self.lateral_convs):]  # p3,p4,p5,p6,p7

        # top to down
        outs = [laterals[i] for i in range(len(laterals))]

        for i in range(len(laterals)-1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.

            if 'scale_factor' in self.upsample_cfg:
                up_feat = F.interpolate(outs[i],
                                        **self.upsample_cfg)
            else:
                prev_shape = outs[i-1].shape[2:]
                up_feat = F.interpolate(
                    outs[i], size=prev_shape, **self.upsample_cfg)
            # weight combine
            outs[i-1] = self.lateral_combine_conv[i -
                                                  1](self.lateral_combine[i-1]([outs[i-1], up_feat]))

        # down to top
        for i in range(len(outs)-1):
            # print(laterals[i].size())
            down_feat = F.max_pool2d(outs[i], 3, stride=2, padding=1)
            # print(down_feat.size())
            cur_outs = outs[i+1]
            if i != len(laterals)-2:
                cur_inputs = laterals[i+1]
                outs[i +
                     1] = self.out_combine[i]([down_feat, cur_outs, cur_inputs])
            else:
                outs[i+1] = self.out_combine[i]([down_feat, cur_outs])
            outs[i+1] = self.out_combine_conv[i](outs[i+1])

        return outs


@MODELS.register_module()
class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=160,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 num_repeat=6,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_repeat = num_repeat
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.downsample_convs = nn.ModuleList()
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_conv = nn.Sequential(
                    ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False),
                    nn.MaxPool2d(3,2,1)
                    )
                self.downsample_convs.append(extra_conv)

        out_channels = out_channels if self.add_extra_convs else self.in_channels[
            self.backbone_end_level-1]
        self.bi_fpn = nn.ModuleList()
        for i in range(self.num_repeat):
            if i == 0:
                in_channels = self.in_channels[self.start_level:self.backbone_end_level]+[
                    out_channels]*extra_levels
            else:
                in_channels = [self.out_channels]*num_outs
            self.bi_fpn.append(SingleBiFPN(in_channels, self.out_channels, no_norm_on_lateral=no_norm_on_lateral,
                                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_cfg=upsample_cfg))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        outs = list(inputs[self.start_level:self.backbone_end_level])
        used_backbone_levels = len(outs)
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 3, stride=2, padding=1))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(self.num_outs-used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.downsample_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.downsample_convs[i](outs[-1]))

        # p2,p3,p4,p5,p6,p7
        # forward to bifpn
        for i in range(self.num_repeat):
            outs = self.bi_fpn[i](outs)
        return tuple(outs)



@MODELS.register_module()
class RepPointsV2MaskDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 mask_inbox=False):
        self.mask_inbox = mask_inbox
        super(RepPointsV2MaskDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                test_cfg, data_preprocessor, init_cfg)
        # print(isinstance(self.bbox_head, nn.Module))
    def loss(self, *args, **kwargs):
        if self.train:
            return self.forward_train( *args, **kwargs)
        else:
            return self.simple_test(*args, **kwargs)
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_sem_map=None,
                      gt_sem_weights=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | list[BitmapMasks]) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
               
        x = self.extract_feat(img)
        # At a cut there is no image, only Swin features: rebuild the input size to pad the GT masks.
        # hw_shape is the output of stage cut_point-1, i.e. the input / 2**(cut_point+1) (4 at cut 1, 8 at cut 2, ...).
        scale = 2 ** (getattr(self, 'cut_point', 1) + 1)
        if isinstance(img,dict):
            new_dims = [3] + [d * scale for d in img['hw_shape']]
            img = torch.zeros(new_dims).to(gt_bboxes[0].device)
        elif isinstance(img, SplitSwinFeatures):
            new_dims = [3] + [d * scale for d in img.hw_shape]
            img = torch.zeros(new_dims).to(gt_bboxes[0].device)

        masks = []
        # print(len(gt_masks))
        for mask in gt_masks:
            # print(type(mask))
            mask_tensor = img.new_tensor(mask.masks)
            mask_tensor = F.pad(mask_tensor, pad=(0, img.size(-1)-mask_tensor.size(-1), 0, img.size(-2)-mask_tensor.size(-2)))
            masks.append(mask_tensor)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                            gt_labels, gt_bboxes_ignore, gt_sem_map, gt_sem_weights, masks)
        return losses
        
    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip'] if 'flip' in img_info[0] else False
            flip_direction = img_info[0]['flip_direction'] if 'flip' in img_info[0] else None
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
            
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
    
    


    def mask2result(self, x, det_labels, inst_inds, img_meta, det_bboxes, pred_instances=None, rescale=True, return_score=False):
        resized_im_h, resized_im_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]
        if pred_instances is not None:
            pred_instances = pred_instances[inst_inds]
        else:
            pred_instances = self.bbox_head.pred_instances[inst_inds]

        scale_factor = img_meta['scale_factor'] if rescale else [1, 1, 1, 1]
        pred_instances.boxsz = torch.stack((det_bboxes[:, 2] * scale_factor[2] - det_bboxes[:, 0] * scale_factor[0],
            det_bboxes[:, 3] * scale_factor[3] - det_bboxes[:, 1] * scale_factor[1]), axis=-1)
        mask_logits = self.bbox_head.mask_head(x, pred_instances)
        if len(pred_instances) > 0:
            mask_logits = aligned_bilinear(mask_logits, self.bbox_head.mask_head.head.mask_out_stride)
            mask_logits = mask_logits[:, :, :resized_im_h, :resized_im_w]
            mask_logits = F.interpolate(
                mask_logits,
                size=(ori_h, ori_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            mask_pred = (mask_logits > 0.5)
            flip = img_meta['flip'] if 'flip' in img_meta else False
            flip_direction = img_meta['flip_direction'] if 'flip' in img_meta else None
            if flip:
                if flip_direction == 'horizontal':
                    mask_pred = mask_pred.cpu().numpy()[:, :, ::-1]
                elif flip_direction == 'vertical':
                    mask_pred = mask_pred.cpu().numpy()[:, ::-1, :]
                else:
                    raise ValueError
            else:
               mask_pred = mask_pred.cpu().numpy()
        else:
            mask_pred = torch.zeros((self.bbox_head.num_classes, *img_meta['ori_shape'][:2]), dtype=torch.int)
        cls_segms = [[] for _ in range(self.bbox_head.num_classes)]  # BG is not included in num_classes
        cls_scores =[[] for _ in range(self.bbox_head.num_classes)]

        for i, label in enumerate(det_labels):
            score = det_bboxes[i][-1]
            if self.mask_inbox:
                mask_pred_ = torch.zeros_like(mask_pred[i])
                det_bbox_ = det_bboxes[i, :-1].clone()
                det_bbox_[[0, 1]], det_bbox_[[2, 3]] = det_bbox_[[0, 1]].floor(), det_bbox_[[2, 3]].ceil()
                det_bbox_ = det_bbox_.int()
                mask_pred_[det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]] = mask_pred[i][det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]]
                cls_segms[label].append(mask_pred_.cpu().numpy())
            else:
                cls_segms[label].append(mask_pred[i].cpu().numpy() if isinstance(mask_pred[i], torch.Tensor) else mask_pred[i])
            cls_scores[label].append(score.cpu().numpy())
        
        if return_score:
            return cls_segms,cls_scores
        return cls_segms
    
    def predict(self, inputs, data_samples):
        # print(inputs.shape)
        if isinstance(data_samples[0], DetDataSample):
            img_metas = [ds.metainfo for ds in data_samples]
            rescale = 'scale_factor' in img_metas[0]  
        else:
            raise TypeError(f'Expected data_samples to be a list of DetDataSample objects, got {type(data_samples[0])} instead')
            # print(rescale)
        x = self.extract_feat(inputs)
        device = x[0].device
        outs = self.bbox_head(x)
        # print(self.bbox_head)
        # print(self.bbox_head.device)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        mask_list = [self.bbox_head.get_masks([xl[[i]] for xl in x], det_labels, inst_inds, img_metas[i], det_bboxes, rescale = rescale) 
                for i, (det_bboxes, det_labels, inst_inds) in enumerate(bbox_list)]
        # print(self.bbox_head.pred_instances)
        # print(len(result_tuple[0][0]),result_tuple[0][1])
        bboxes = []
        labels = []
        scores = []
        for i,((det_bboxes, det_labels, _), masks,ds) in enumerate(zip(bbox_list, mask_list, data_samples)):
            pred_instance = Instances()
            pred_instance.bboxes = det_bboxes[:,:-1]
            pred_instance.scores = det_bboxes[:,-1]
            pred_instance.labels = det_labels
            pred_instance.masks = torch.tensor(masks, device =device)
            ds.pred_instances = pred_instance


        return data_samples

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            [We all use True]
        Returns:
            np.ndarray: proposals
        """
        # print(len(img['outs']))
        x = self.extract_feat(img)
        # print(len(x))
        outs = self.bbox_head(x)
        # print(len(outs))
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # print(bbox_list)
        if not self.bbox_head.mask_head: # detection only
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
        else:
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels, _ in bbox_list
            ]
            cls_segms = [
                self.mask2result([xl[[i]] for xl in x], det_labels, inst_inds, img_metas[i], det_bboxes, rescale = rescale) 
                for i, (det_bboxes, det_labels, inst_inds) in enumerate(bbox_list)
            ]
            # print(len(bbox_results), len(cls_segms))
            return list(zip(bbox_results, cls_segms))

    def aug_test_simple(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, rescale=False):
        return self.aug_test_simple(imgs, img_metas, rescale)







def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('backbone.layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('backbone.patch_embed'):
            new_v = v
            if 'proj' in k and not 'projection' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def convert_swin_checkpoint_file(src, dst):
    """Convert a RepPointsV2 Swin checkpoint's keys to mmdet 3 names; return the converted file's path.

    The result is saved under dst's basename inside a private temp directory next to dst (torch.save
    embeds the basename in the archive, so this keeps the sha256, and hence the name, identical to the
    old code's), then moved into place with os.replace: atomic and synchronous, so the returned path
    is complete on return and concurrent conversions never share a file. Nothing is written at dst.
    Name: <dst, its basename without .pth>-<first 8 hex of the sha256>.pth (the directory is left as it is, even
    one whose name contains .pth). An existing file with that name is kept.
    """
    checkpoint = CheckpointLoader.load_checkpoint(src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    dst_dir = os.path.dirname(os.path.abspath(dst))
    tmp_dir = tempfile.mkdtemp(prefix='.convert_', dir=dst_dir)
    try:
        tmp = os.path.join(tmp_dir, os.path.basename(dst))
        torch.save(swin_converter(state_dict), tmp)
        sha = _sha256_file(tmp)
        final_file = os.path.join(os.path.dirname(dst),
                                  os.path.basename(dst).replace('.pth', '') + '-{}.pth'.format(sha[:8]))
        if not (os.path.isfile(final_file) and _sha256_file(final_file) == sha):
            os.replace(tmp, final_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'Done!!, save to {final_file}')
    return final_file

