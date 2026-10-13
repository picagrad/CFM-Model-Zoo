'''The registry tables of the registered models: data only.

The logic that resolves and builds these models is in ``cfm_model_zoo.directory``, which re-exports every
name defined here and reads them through its own namespace; import them from there. Its docstring says what
each table holds.
'''
import os
from typing import Optional

from .split_models import *   # the Split classes and the optional-dependency flags (its __all__)

_CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def _config_path(relative: str) -> str:
    '''Absolute path of a config shipped inside the package (cfm_model_zoo/configs/<relative>).'''
    return os.path.join(_CONFIG_ROOT, relative)


def _shared(filename: str, url: Optional[str] = None, size: Optional[int] = None,
            sha256: Optional[str] = None) -> dict:
    '''A checkpoint downloaded from its publisher's URL (not through mim), verified by size and sha256.
    url None: no public download location (the file must be supplied).'''
    return {'filename': filename, 'url': url, 'size': size, 'sha256': sha256}


# The registered names are the automatic names of their configs (cfm_model_zoo.naming.registered_name,
# <Pkg>Split<Method><Backbone>); a library check recomputes every one from its config. They are written out
# here so that they exist without mmdet/mmseg installed and without parsing every config at import.
#
# LEGACY_NAMES: the short names used before these (e.g. by the paper's code) stay mapped to the same entries,
# in every registry dict below.
LEGACY_NAMES = {
    'MaskRCNN': 'MMDetSplitMaskRCNNSwinT',
    'RetinaNet': 'MMDetSplitRetinaNetR50',
    'CascadeMaskRCNN': 'MMDetSplitCascadeRCNNR50',
    'Mask2FormerSWIN': 'MMDetSplitMask2FormerSwinT',
    'Mask2FormerResNet': 'MMDetSplitMask2FormerR50',
    'GroundingDINO': 'MMDetSplitGroundingDINOSwinT',
    'GLIP': 'MMDetSplitGLIPSwinT',
    'RepPointsV2': 'MMDetSplitRepPointsV2MaskDetectorSwinT',
    'ATSSResNet': 'MMDetSplitATSSR50',
    'ATSSSWIN-L': 'MMDetSplitATSSSwinL',
    'FCOS': 'MMDetSplitFCOSR50',
    'FasterRCNN': 'MMDetSplitFasterRCNNR50',
    'DDQDETRResNet': 'MMDetSplitDDQDETRR50',
    'DDQDETRSWIN-L': 'MMDetSplitDDQDETRSwinL',
    'DINOResNet': 'MMDetSplitDINOR50',
    'DINOSWIN-L': 'MMDetSplitDINOSwinL',
    'DeformableDETR': 'MMDetSplitDeformableDETRR50',
    'SegANNResNet': 'MMSegSplitANNR50',
    'SegAPCNetResNet': 'MMSegSplitAPCNetR50',
    'SegCCNetResNet': 'MMSegSplitCCNetR50',
    'SegDANetResNet': 'MMSegSplitDANetR50',
    'SegDeepLabV3ResNet': 'MMSegSplitDeepLabV3R50',
    'SegDeepLabV3PlusResNet': 'MMSegSplitDeepLabV3PlusR50',
    'SegDMNetResNet': 'MMSegSplitDMNetR50',
    'SegDNLNetResNet': 'MMSegSplitDNLNetR50',
    'SegENCNetResNet': 'MMSegSplitEncNetR50',
    'SegFastFCNASPPResNet': 'MMSegSplitFastFCNASPPR50',
    'SegFastFCNENCResNet': 'MMSegSplitFastFCNEncR50',
    'SegFastFCNPSPResNet': 'MMSegSplitFastFCNPSPR50',
    'SegFCNResNet': 'MMSegSplitFCNR50',
    'SegFPNResNet': 'MMSegSplitFPNR50',
    'SegGCNetResNet': 'MMSegSplitGCNetR50',
    'SegISANetResNet': 'MMSegSplitISANetR50',
    'SegNonLocalNetResNet': 'MMSegSplitNonLocalNetR50',
    'SegPSANetResNet': 'MMSegSplitPSANetR50',
    'SegPSPNetResNet': 'MMSegSplitPSPNetR50',
    'SegMaskFormerResNet': 'MMSegSplitMaskFormerR50',
    'SegMaskFormerSWIN': 'MMSegSplitMaskFormerSwinT',
    'SegMask2FormerResNet': 'MMSegSplitMask2FormerR50',
    'SegMask2FormerSWIN': 'MMSegSplitMask2FormerSwinT',
    'SegKNetDeepLabResNet': 'MMSegSplitKNetASPPR50',
    'SegKNetFCNResNet': 'MMSegSplitKNetFCNR50',
    'SegKNetPSPResNet': 'MMSegSplitKNetPSPR50',
    'SegKNetUPerResNet': 'MMSegSplitKNetUPerR50',
    'SegKNetSWIN': 'MMSegSplitKNetUPerSwinT',
    'SegUperNetResNet': 'MMSegSplitUPerNetR50',
    'SegUperNetSWIN': 'MMSegSplitUPerNetSwinT',
}


def _with_legacy_names(registry: dict) -> dict:
    '''The registry plus each old short name as a second key for the same value.'''
    return {**registry, **{old: registry[new] for old, new in LEGACY_NAMES.items() if new in registry}}


if MMDET_AVAILABLE:
    detectors = {'MMDetSplitMaskRCNNSwinT': MMDetSplitMaskRCNN,
                 'MMDetSplitRetinaNetR50': MMDetSplitRetinaNet,
                 'MMDetSplitCascadeRCNNR50': MMDetSplitCascadeRCNN,
                 'MMDetSplitMask2FormerSwinT': MMDetSplitMask2Former,
                 'MMDetSplitMask2FormerR50': MMDetSplitMask2Former,
                 'MMDetSplitGroundingDINOSwinT': MMDetSplitGroundingDINO,
                 'MMDetSplitGLIPSwinT': MMDetSplitGLIP,
                 'MMDetSplitRepPointsV2MaskDetectorSwinT': globals().get('MMDetSplitRepPointsV2MaskDetector'),
                 'MMDetSplitATSSR50': MMDetSplitATSS,
                 'MMDetSplitATSSSwinL': MMDetSplitATSS,
                 'MMDetSplitFCOSR50': MMDetSplitFCOS,
                 'MMDetSplitFasterRCNNR50': MMDetSplitFasterRCNN,
                 'MMDetSplitDDQDETRR50': MMDetSplitDDQDETR,
                 'MMDetSplitDDQDETRSwinL': MMDetSplitDDQDETR,
                 'MMDetSplitDINOR50': MMDetSplitDINO,
                 'MMDetSplitDINOSwinL': MMDetSplitDINO,
                 'MMDetSplitDeformableDETRR50': MMDetSplitDeformableDETR,
                 }
    detectors = {k: v for k, v in detectors.items() if v is not None}   # RepPointsV2 needs legacy/
    DETECTION_MODELS = {
        'MMDetSplitMaskRCNNSwinT': {
            'mim_name': 'mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco', 'backbone': 'SWIN',
            'local_config': _config_path('detection/mask_rcnn_swin_t.py'),
            'local_checkpoint': 'mask_rcnn_swin_t.pth',
        },
        'MMDetSplitRetinaNetR50': {
            'mim_name': 'retinanet_r50_fpn_ms-640-800-3x_coco', 'backbone': 'ConvNet',
            'local_config': _config_path('detection/retinanet_r50.py'),
            'local_checkpoint': 'retinanet_r50.pth',
        },
        'MMDetSplitCascadeRCNNR50': {
            'mim_name': 'cascade-mask-rcnn_r50_fpn_20e_coco', 'backbone': 'ConvNet',
            'local_config': _config_path('detection/cascade_mask_rcnn_r50.py'),
            'local_checkpoint': 'cascade_rcnn_r50.pth',
        },
        'MMDetSplitMask2FormerSwinT': {
            'mim_name': 'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco', 'backbone': 'SWIN',
            'local_config': _config_path('detection/mask2former_swin_t.py'),
            'local_checkpoint': 'mask2former_swin_instance.pth',
        },
        'MMDetSplitMask2FormerR50': {
            'mim_name': 'mask2former_r50_8xb2-lsj-50e_coco', 'backbone': 'ConvNet',
            'local_config': _config_path('detection/mask2former_r50.py'),
            'local_checkpoint': 'mask2former_r50.pth',
        },
        'MMDetSplitGroundingDINOSwinT': {
            'mim_name': 'grounding_dino_swin-t_finetune_16xb2_1x_coco', 'backbone': 'SWIN',
        },
        'MMDetSplitGLIPSwinT': {
            'mim_name': 'glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco', 'backbone': 'SWIN',
            'local_config': _config_path('multimodal/glip_t.py'),
            'local_checkpoint': 'glip.pth',
        },
        'MMDetSplitRepPointsV2MaskDetectorSwinT': {
            # No mim entry. The weights are Microsoft's release of Mask RepPoints V2 Swin-T 3x
            # (Swin-Transformer-Object-Detection); prepare_checkpoint converts their Swin keys for the port
            # in legacy/ once, into the cache directory.
            'mim_name': None, 'backbone': 'SWIN',
            'local_config': _config_path('detection/mask_reppointsv2_swin_t.py'),
            'shared_checkpoint': _shared(
                'mask_reppointsv2_swin_tiny_patch4_window7_3x.pth',
                url='https://github.com/SwinTransformer/storage/releases/download/v1.0.9/'
                    'mask_reppointsv2_swin_tiny_patch4_window7_3x.pth',
                size=562590621,
                sha256='9b13cc01f40fecdfa3e7ebfe6163f23a2f7b47569d5edcfb2bbbc81d30c20456'),
        },
        'MMDetSplitATSSR50': {'mim_name': 'atss_r50_fpn_dyhead_1x_coco', 'backbone': 'ConvNet'},
        'MMDetSplitATSSSwinL': {'mim_name': 'atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco', 'backbone': 'SWIN'},
        'MMDetSplitFCOSR50': {
            'mim_name': 'fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco',
            'backbone': 'ConvNet'},
        'MMDetSplitFasterRCNNR50': {'mim_name': 'faster-rcnn_r50_fpn_mstrain_3x_coco', 'backbone': 'ConvNet'},
        'MMDetSplitDDQDETRR50': {
            # mmdet 3.3.0's mim index lists this name, but its metafile points it at a config that does not
            # exist, so mim cannot resolve it. The config ships with this package; the checkpoint is
            # OpenMMLab's, downloaded from the URL in mmdet's configs/ddq/metafile.yml.
            'mim_name': 'ddq-detr-5scale_r50_8xb2-12e_coco', 'backbone': 'ConvNet',
            'local_config': _config_path('detection/ddq_r50.py'),
            'shared_checkpoint': _shared(
                'ddq_detr_5scale_coco_1x.pth',
                url='https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_5scale_coco_1x.pth',
                size=199538331,
                sha256='e379939e11b1b70fca27e3a5c39711cde5325f9d6354eb1cb9c1902ff1da2e6b'),
        },
        'MMDetSplitDDQDETRSwinL': {
            'mim_name': 'ddq-detr-4scale_swinl_8xb2-30e_coco', 'backbone': 'SWIN',
            'local_config': _config_path('detection/ddq_swin_l.py'),
            'shared_checkpoint': _shared(
                'ddq_detr_swinl_30e.pth',
                url='https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth',
                size=883034647,
                sha256='11bf2157266c95f69de3f8d525fd4d670f75882747fd2bf1f01fe72cfe3f70ac'),
        },
        'MMDetSplitDINOR50': {'mim_name': 'dino-4scale_r50_improved_8xb2-12e_coco', 'backbone': 'ConvNet'},
        'MMDetSplitDINOSwinL': {'mim_name': 'dino-5scale_swin-l_8xb2-36e_coco', 'backbone': 'SWIN'},
        'MMDetSplitDeformableDETRR50': {'mim_name': 'deformable-detr-refine-twostage_r50_16xb2-50e_coco',
                                        'backbone': 'ConvNet'},
    }
    if not LEGACY_AVAILABLE:
        del DETECTION_MODELS['MMDetSplitRepPointsV2MaskDetectorSwinT']
    detectors, DETECTION_MODELS = _with_legacy_names(detectors), _with_legacy_names(DETECTION_MODELS)
else:
    detectors = {}
    DETECTION_MODELS = {}


# Registered names, kept whether or not the optional packages are installed, so a missing
# extra is reported as a missing extra rather than as an unknown model name.
_SEG_BACKBONE = {
    'MMSegSplitANNR50': 'ConvNet', 'MMSegSplitAPCNetR50': 'ConvNet',
    'MMSegSplitCCNetR50': 'ConvNet', 'MMSegSplitDANetR50': 'ConvNet',
    'MMSegSplitDeepLabV3R50': 'ConvNet', 'MMSegSplitDeepLabV3PlusR50': 'ConvNet',
    'MMSegSplitDMNetR50': 'ConvNet', 'MMSegSplitDNLNetR50': 'ConvNet',
    'MMSegSplitEncNetR50': 'ConvNet', 'MMSegSplitFastFCNASPPR50': 'ConvNet',
    'MMSegSplitFastFCNEncR50': 'ConvNet', 'MMSegSplitFastFCNPSPR50': 'ConvNet',
    'MMSegSplitFCNR50': 'ConvNet', 'MMSegSplitFPNR50': 'ConvNet',
    'MMSegSplitGCNetR50': 'ConvNet', 'MMSegSplitISANetR50': 'ConvNet',
    'MMSegSplitNonLocalNetR50': 'ConvNet', 'MMSegSplitPSANetR50': 'ConvNet',
    'MMSegSplitPSPNetR50': 'ConvNet', 'MMSegSplitMaskFormerR50': 'ConvNet',
    'MMSegSplitMaskFormerSwinT': 'SWIN', 'MMSegSplitMask2FormerR50': 'ConvNet',
    'MMSegSplitMask2FormerSwinT': 'SWIN', 'MMSegSplitKNetASPPR50': 'ConvNet',
    'MMSegSplitKNetFCNR50': 'ConvNet', 'MMSegSplitKNetPSPR50': 'ConvNet',
    'MMSegSplitKNetUPerR50': 'ConvNet', 'MMSegSplitKNetUPerSwinT': 'SWIN',
    'MMSegSplitUPerNetR50': 'ConvNet', 'MMSegSplitUPerNetSwinT': 'SWIN',
}
_SEG_MIM_NAME = {
    'MMSegSplitANNR50': 'ann_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitAPCNetR50': 'apcnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitCCNetR50': 'ccnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitDANetR50': 'danet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitDeepLabV3R50': 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitDeepLabV3PlusR50': 'deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitDMNetR50': 'dmnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitDNLNetR50': 'dnl_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitEncNetR50': 'encnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitFastFCNASPPR50': 'fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512',
    'MMSegSplitFastFCNEncR50': 'fastfcn_r50-d32_jpu_enc_4xb4-160k_ade20k-512x512',
    'MMSegSplitFastFCNPSPR50': 'fastfcn_r50-d32_jpu_psp_4xb4-160k_ade20k-512x512',
    'MMSegSplitFCNR50': 'fcn_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitFPNR50': 'fpn_r50_4xb4-160k_ade20k-512x512',
    'MMSegSplitGCNetR50': 'gcnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitISANetR50': 'isanet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitNonLocalNetR50': 'nonlocal_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitPSANetR50': 'psanet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitPSPNetR50': 'pspnet_r50-d8_4xb4-160k_ade20k-512x512',
    'MMSegSplitMaskFormerR50': 'maskformer_r50-d32_8xb2-160k_ade20k-512x512',
    'MMSegSplitMaskFormerSwinT': 'maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512',
    'MMSegSplitMask2FormerR50': 'mask2former_r50_8xb2-160k_ade20k-512x512',
    'MMSegSplitMask2FormerSwinT': 'mask2former_swin-t_8xb2-160k_ade20k-512x512',
    'MMSegSplitKNetASPPR50': 'knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512',
    'MMSegSplitKNetFCNR50': 'knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512',
    'MMSegSplitKNetPSPR50': 'knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512',
    'MMSegSplitKNetUPerR50': 'knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512',
    'MMSegSplitKNetUPerSwinT': 'knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512',
    'MMSegSplitUPerNetR50': 'upernet_r50_4xb4-160k_ade20k-512x512',
    'MMSegSplitUPerNetSwinT': 'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512',
}
_SEGMENTATION_NAMES = tuple(_SEG_MIM_NAME)
_DETECTION_NAMES = ('MMDetSplitMaskRCNNSwinT', 'MMDetSplitRetinaNetR50', 'MMDetSplitCascadeRCNNR50',
                    'MMDetSplitMask2FormerSwinT', 'MMDetSplitMask2FormerR50', 'MMDetSplitGroundingDINOSwinT',
                    'MMDetSplitGLIPSwinT', 'MMDetSplitRepPointsV2MaskDetectorSwinT', 'MMDetSplitATSSR50',
                    'MMDetSplitATSSSwinL', 'MMDetSplitFCOSR50', 'MMDetSplitFasterRCNNR50',
                    'MMDetSplitDDQDETRR50', 'MMDetSplitDDQDETRSwinL', 'MMDetSplitDINOR50',
                    'MMDetSplitDINOSwinL', 'MMDetSplitDeformableDETRR50')
REGISTERED_NAMES = _DETECTION_NAMES + _SEGMENTATION_NAMES          # the registered names, in registry order
_DETECTION_NAMES += tuple(old for old, new in LEGACY_NAMES.items() if new in _DETECTION_NAMES)
_SEGMENTATION_NAMES += tuple(old for old, new in LEGACY_NAMES.items() if new in _SEGMENTATION_NAMES)

if MMSEG_AVAILABLE:
    segmentors = {
        'MMSegSplitANNR50': MMSegSplitEncoderDecoder,
        'MMSegSplitAPCNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitCCNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitDANetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitDeepLabV3R50': MMSegSplitEncoderDecoder,
        'MMSegSplitDeepLabV3PlusR50': MMSegSplitEncoderDecoder,
        'MMSegSplitDMNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitDNLNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitEncNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitFastFCNASPPR50': MMSegSplitEncoderDecoder,
        'MMSegSplitFastFCNEncR50': MMSegSplitEncoderDecoder,
        'MMSegSplitFastFCNPSPR50': MMSegSplitEncoderDecoder,
        'MMSegSplitFCNR50': MMSegSplitEncoderDecoder,
        'MMSegSplitFPNR50': MMSegSplitEncoderDecoder,
        'MMSegSplitGCNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitISANetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitNonLocalNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitPSANetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitPSPNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitMaskFormerR50': MMSegSplitEncoderDecoder,
        'MMSegSplitMaskFormerSwinT': MMSegSplitEncoderDecoder,
        'MMSegSplitMask2FormerR50': MMSegSplitEncoderDecoder,
        'MMSegSplitMask2FormerSwinT': MMSegSplitEncoderDecoder,
        'MMSegSplitKNetASPPR50': MMSegSplitEncoderDecoder,
        'MMSegSplitKNetFCNR50': MMSegSplitEncoderDecoder,
        'MMSegSplitKNetPSPR50': MMSegSplitEncoderDecoder,
        'MMSegSplitKNetUPerR50': MMSegSplitEncoderDecoder,
        'MMSegSplitKNetUPerSwinT': MMSegSplitEncoderDecoder,
        'MMSegSplitUPerNetR50': MMSegSplitEncoderDecoder,
        'MMSegSplitUPerNetSwinT': MMSegSplitEncoderDecoder,
    }
    # Every segmentor resolves through mim.
    SEGMENTATION_MODELS = {
        name: {'mim_name': _SEG_MIM_NAME[name], 'backbone': _SEG_BACKBONE[name]}
        for name in segmentors
    }
    segmentors, SEGMENTATION_MODELS = _with_legacy_names(segmentors), _with_legacy_names(SEGMENTATION_MODELS)
else:
    segmentors = {}
    SEGMENTATION_MODELS = {}

backbone_style = {k: v['backbone'] for k, v in {**DETECTION_MODELS, **SEGMENTATION_MODELS}.items()}
