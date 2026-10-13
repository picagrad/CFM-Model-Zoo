import os.path as osp
import torch
import numpy as np
import pycocotools.mask as maskUtils
from mmdet.datasets.transforms import PackDetInputs
from mmengine.structures import PixelData

from mmdet.registry import TRANSFORMS
# from mmcv.parallel import DataContainer as DC

# @TRANSFORMS.register_module()
# class RPDV2FormatBundle():
#     """Default formatting bundle.

#     It simplifies the pipeline of formatting common fields, including "img",
#     "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
#     These fields are formatted as follows.

#     - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
#     - proposals: (1)to tensor, (2)to DataContainer
#     - gt_bboxes: (1)to tensor, (2)to DataContainer
#     - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
#     - gt_labels: (1)to tensor, (2)to DataContainer
#     - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
#     - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
#                        (3)to DataContainer (stack=True)
#     """
    
#     def __init__(self):
#         super(RPDV2FormatBundle, self).__init__()

#     def __call__(self, results):
#         """Call function to transform and format common fields in results.

#         Args:
#             results (dict): Result dict contains the data to convert.

#         Returns:
#             dict: The result dict contains the data that is formatted with
#                 default bundle.
#         """

#         if 'img' in results:
#             img = results['img']
#             # add default meta keys
#             results = self._add_default_meta_keys(results)
#             if len(img.shape) < 3:
#                 img = np.expand_dims(img, -1)
#             img = np.ascontiguousarray(img.transpose(2, 0, 1))
#             results['img'] = DC(to_tensor(img), stack=True)
#         for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
#             if key not in results:
#                 continue
#             results[key] = DC(to_tensor(results[key]))
#         if 'gt_masks' in results:
#             results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
#         if 'gt_semantic_seg' in results:
#             results['gt_semantic_seg'] = DC(
#                 to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
#         if 'gt_sem_map' in results:
#             results['gt_sem_map'] = DC(to_tensor(results['gt_sem_map']), stack=True)
#         if 'gt_sem_weights' in results:
#             results['gt_sem_weights'] = DC(to_tensor(results['gt_sem_weights']), stack=True)
#         if 'gt_contours' in results:
#             results['gt_contours'] = DC(to_tensor(results['gt_contours']))

#         return results

#     def _add_default_meta_keys(self, results):
#         """Add default meta keys.

#         We set default meta keys including `pad_shape`, `scale_factor` and
#         `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
#         `Pad` are implemented during the whole pipeline.

#         Args:
#             results (dict): Result dict contains the data to convert.

#         Returns:
#             results (dict): Updated result dict contains the data to convert.
#         """
#         img = results['img']
#         results.setdefault('pad_shape', img.shape)
#         results.setdefault('scale_factor', 1.0)
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results.setdefault(
#             'img_norm_cfg',
#             dict(
#                 mean=np.zeros(num_channels, dtype=np.float32),
#                 std=np.ones(num_channels, dtype=np.float32),
#                 to_rgb=False))
#         return results

#     def __repr__(self):
#         return self.__class__.__name__

@TRANSFORMS.register_module()
class LoadRPDV2Annotations():
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self, num_classes = 80):
        super(LoadRPDV2Annotations, self).__init__()
        self.num_classes = num_classes

    def _load_semantic_map_from_box(self, results):
        # print(results)
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_bboxes_labels']
        pad_shape = results['img_shape']
        
        gt_areas = gt_bboxes.areas
        gt_bboxes_ten = gt_bboxes.tensor
        gt_sem_map = torch.zeros((self.num_classes, int(pad_shape[0] / 8), int(pad_shape[1] / 8)))
        gt_sem_weights = torch.zeros((self.num_classes, int(pad_shape[0] / 8), int(pad_shape[1] / 8)))
        indexs = np.argsort(gt_areas).numpy()
        for ind in indexs[::-1]:
            box = gt_bboxes_ten[ind]
            box_mask = torch.zeros((int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=torch.int)
            box_mask[int(box[1] / 8):int(box[3] / 8) + 1, int(box[0] / 8):int(box[2] / 8) + 1] = 1
            
            try:
                gt_sem_map[gt_labels[ind]][box_mask > 0] = 1
                gt_sem_weights[gt_labels[ind]][box_mask > 0] = 1 / gt_areas[ind]
            except IndexError as ie:
                print(f'box: {box}')
                box_mask = torch.zeros((int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=torch.int)
                print(f'original box_mask.shape: {box_mask.shape}')
                box_mask[int(box[1] / 8):int(box[3] / 8) + 1, int(box[0] / 8):int(box[2] / 8) + 1] = 1
                print(f'final box_mask.shape {box_mask.shape}')
                print(f'gt_sem_map.shape: {gt_sem_map.shape}')
                print(f'gt_sem_weights.shape: {gt_sem_weights.shape}')
                print(f'gt_labels[ind]: {gt_labels[ind]}')
                print(f'gt_sem_map[gt_labels[ind]].shape: {gt_sem_map[gt_labels[ind]].shape}')
                raise ie


        results['gt_sem'] = {'sem_map':gt_sem_map, 'sem_weights': gt_sem_weights}

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = self._load_semantic_map_from_box(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox_semantic_map={True})'
        return repr_str
    

@TRANSFORMS.register_module()
class NewPackDetInputs(PackDetInputs):
    def __init__(self,
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction')):
        super().__init__(meta_keys = meta_keys)

    def transform(self, results: dict) -> dict:
        out_results = super().transform(results)
        # print(out_results)
        if 'gt_sem' in results:
            out_results['data_samples'].gt_sem = PixelData(**results['gt_sem'])
        return out_results
        
