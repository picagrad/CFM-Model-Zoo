import os
import torch
from contextlib import contextmanager
from copy import deepcopy

from torch import nn
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.instancenorm import _InstanceNorm
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass

from mmengine.config import Config

try:
    from mmdet.structures import DetDataSample
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

try:
    from mmseg.structures import SegDataSample
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False

from .naming import split_class_name
from .split_backbones import ResNetSplitMixin, SwinSplitMixin, split_backbone_in_place


@dataclass
class SplitSwinFeatures:
    outs: List[torch.Tensor]
    hw_shape: torch.Tensor
    device: torch.device
    x: Optional[torch.tensor] = None
    
    def __len__(self):
        return self.outs[0].shape[0] if self.outs else 0
    
    def __iter__(self):
        for i in range(len(self)):
            outs = [out[i:i+1] for out in self.outs]
            yield SplitSwinFeatures(outs = outs, device=self.device, hw_shape= self.hw_shape, x=self.x[i:i+1] if self.x is not None else None)

@dataclass
class SplitResNetFeatures:
    outs: List[torch.Tensor]
    device: torch.device
    x: Optional[torch.tensor] = None
    
    def __len__(self):
        return self.outs[0].shape[0] if self.outs else 0

    def __iter__(self):
        for i in range(len(self)):
            outs = [out[i:i+1] for out in self.outs]
            yield SplitResNetFeatures(outs = outs, device=self.device, x=self.x[i:i+1] if self.x is not None else None)
    
def _feature_frontend(model: nn.Module, data : Union[torch.Tensor, dict], preprocessed: bool = False) ->dict:
    '''
    Return the split features extracted by the frontend of the split model
    if input 'data' is a dictionary, assume preprocessing has not been performed and preprocess it ourselves
    Otherwise assume that 'data' is the tensor part of the preprocessed input
    

    Args:
        model (nn.Module): The model with a backbone from which to extract features
        data (Union[torch.Tensor, dict]): Image tensor with shape (N, C, H ,W) or dictionary with 'inputs' key containing the image tensor and 
                                        'data_samples' key containing the data samples (see mmdet.structures.DetDataSample)

    Returns:
        out (dict): Dictionary with 'inputs' key that contains the split features extracted by the frontend
                    if 'data_samples' key is present in the input dictionary, it is also returned in the output dictionary


    '''
    # print(model)
    if isinstance(data, dict):
        if not preprocessed:
            data = model.frontend_preprocessor(data, model.training)
        x = data['inputs']
        out = {'inputs': x, 'data_samples': data['data_samples']} if 'data_samples' in data else {'inputs': x}
    else:
        x = data
        out = {'inputs': x}

    if model.cut_point>0:
        feats =  model.backbone.split_forward(x, output_layer = model.cut_point-1)
        # data['inputs'] = {"hw_shape": out[0],
        #                   "outs": out[1]}
        if isinstance(model.backbone, SwinSplitMixin):
            out['inputs'] = SplitSwinFeatures(outs=feats[1], hw_shape=feats[0], device=x.device)
        elif isinstance(model.backbone, ResNetSplitMixin):
            out['inputs'] = SplitResNetFeatures(outs=list(feats), device=x.device)

    return out       

def _extract_feat_from_split(model: nn.Module, batch_inputs: Union[torch.Tensor, SplitSwinFeatures, SplitResNetFeatures], use_neck:bool = True) -> Tuple[torch.Tensor]:
    """Resume the backbone at model.cut_point from the frontend's split features and return its full
    multi-level features (through the neck when ``use_neck`` and the model has one).

    Args:
        model (nn.Module): The split model whose backbone resumes at its cut point
        batch_inputs: SplitSwinFeatures / SplitResNetFeatures from feature_frontend, or the
            preprocessed image tensor (N, C, H, W) at cut_point=0.

    Returns:
        tuple[Tensor]: Multi-level features that may have
        different resolutions.
    """
    # x, hw_shape, outputs = self.backbone.split_forward(batch_inputs, outs=[], output_layer=1)
    # print("splitting at layer 1")
    # print(batch_inputs)
    # print(type(batch_inputs))
    if isinstance(batch_inputs, SplitSwinFeatures):
        hw_shape = batch_inputs.hw_shape
        outs = batch_inputs.outs
        x = batch_inputs.x
    elif isinstance(batch_inputs, SplitResNetFeatures):
        outs = batch_inputs.outs
        x = batch_inputs.x
    # elif isinstance(batch_inputs, dict):
    #     x = batch_inputs["x"] if "x" in batch_inputs else None
    #     hw_shape = batch_inputs["hw_shape"]
    #     outs = batch_inputs["outs"]
    else:
        x = batch_inputs
        hw_shape = 0
        outs = []
    if isinstance(model.backbone, SwinSplitMixin):
        x = model.backbone.split_forward(x, hw_shape, outs=outs, input_layer=model.cut_point)
    elif isinstance(model.backbone, ResNetSplitMixin):
        x = model.backbone.split_forward(x, outs=outs, input_layer=model.cut_point)
    else:
        raise TypeError(f'Got {type(model.backbone)} backbone, but only backbones split by '
                        f'SwinSplitMixin / ResNetSplitMixin are supported')
    
    if model.with_neck and use_neck:
        x = model.neck(x)
    return x


# Model families whose raw (mmengine mode='tensor') output is broken in stock mmdet/mmseg: the unsplit model
# raises the same error. Matched by class name along the MRO, so no model library has to be imported here.
_RAW_BROKEN_DETECTORS = {
    'GroundingDINO': "its forward_transformer needs the text features that its loss/predict build first",
    'GLIP': "its head needs the language features that its loss/predict build first",
}
_RAW_BROKEN_HEADS = ('MaskFormerHead', 'Mask2FormerHead')


def _raw_unsupported_reason(model: nn.Module) -> Optional[str]:
    """Why backend(mode='raw') cannot work for ``model`` (None when it can)."""
    for cls in type(model).__mro__:
        if cls.__name__ in _RAW_BROKEN_DETECTORS:
            return _RAW_BROKEN_DETECTORS[cls.__name__]
    head = getattr(model, 'decode_head', None)
    if head is not None and any(c.__name__ in _RAW_BROKEN_HEADS for c in type(head).__mro__):
        return "the MaskFormer/Mask2Former decode head needs the data samples"
    return None


def _make_split_train(new_cls):
    """Build the train() override for a split model class.

    Same shape as mmdet's ResNet.train(): call up, then put the norms back in eval when the
    flag says to. Reading ``self.freeze_norms`` at CALL time -- rather than baking it in at
    construction -- is what makes the policy fully reversible: flip it, call .train(), and
    stock behaviour is back. Freezing only writes .training flags; it never touches weights
    or running statistics, so nothing is destroyed either way.
    """
    def train(self, mode: bool = True):
        super(new_cls, self).train(mode)     # incl. mmdet's own ResNet.train()/norm_eval
        if getattr(self, 'freeze_norms', False):
            freeze_norm_layers(self)
        return self
    return train


_TASK_PACKAGE = {'detection': 'mmdet', 'segmentation': 'mmseg'}


def NetworkSplitter(task: str = 'detection', freeze_norms_default: bool = False,
                    package: Optional[str] = None):
    """Turn an mm model class into its Split class (a generated subclass ``<Pkg>Split<Class>``, e.g.
    ``MMDetSplitMaskRCNN``, or the decorated class itself when its name already contains 'Split').

    A Split model's __init__ makes it complete: base __init__ -> backbone re-classed in place onto
    its split class (TypeError if unsupported) -> cut_point (a validated property, also settable
    later) -> frontend preprocessor. ``package`` is the mm library whose scope builds it; ``task`` keeps the old
    call form NetworkSplitter(task='segmentation') working.
    """
    package = package or _TASK_PACKAGE[task]

    def decorator(cls):
        new_cls = (type(split_class_name(cls.__name__, package), (cls,), {}) if 'Split' not in cls.__name__
                   else cls)
        original_init = cls.__bases__[0].__init__ if cls.__name__ == new_cls.__name__ else cls.__init__

        def __init__(self, cut_point: int = 0, **kwargs):
            kwargs.pop('type', None)
            original_init(self, **kwargs)
            backbone = getattr(self, 'backbone', None)
            if backbone is None:
                raise TypeError(f'{type(self).__name__} has no backbone to split')
            split_backbone_in_place(backbone)
            self.cut_point = cut_point      # checked by the property below
            self.prepare_preprocessing()

        def _get_cut_point(self):
            state = self.__dict__
            if '_cut_point' in state:
                return state['_cut_point']
            if 'cut_point' in state:        # a model pickled before cut_point became a property
                return state['cut_point']
            raise AttributeError(f'{type(self).__name__} has no cut_point yet')

        def _set_cut_point(self, cut_point):
            n_stages = self.backbone.num_split_stages
            if isinstance(cut_point, bool) or not isinstance(cut_point, int) or not 0 <= cut_point <= n_stages:
                raise ValueError(f'cut_point must be an int in 0..{n_stages} for {type(self).__name__} '
                                 f'(0 splits after preprocessing, k after backbone stage k of {n_stages}), '
                                 f'got {cut_point!r}')
            self.__dict__['_cut_point'] = cut_point

        cut_point = property(_get_cut_point, _set_cut_point, doc=(
            'Where the model is split: 0 after preprocessing, k after backbone stage k. Setting it moves the split '
            'at run time (feature_frontend and backend read it on every call); a value outside '
            '0..backbone.num_split_stages, or not an int, raises ValueError and leaves it unchanged.'))

        @classmethod
        def create_from_cfg_and_checkpoint(cls_, cfg, checkpoint_path, device='cpu', cut_point=1, *,
                                           trust_checkpoint=False):
            """Build from an explicit config (path or Config) and checkpoint, through _build_once.
            trust_checkpoint: as in build_split_model."""
            from .directory import _build_once   # runtime import: directory imports this module
            if isinstance(cfg, (str, os.PathLike)):
                cfg = Config.fromfile(cfg)
            model = _build_once(cfg, checkpoint_path, cut_point, device, cls_._cfm_package,
                                trust_checkpoint=trust_checkpoint)
            if not isinstance(model, cls_):
                raise TypeError(f'{cls_.__name__}.create_from_cfg_and_checkpoint: the config builds '
                                f'{type(model).__name__}, not {cls_.__name__}')
            return model

        @classmethod
        def prepare_checkpoint(cls_, checkpoint_path, cache_dir=None):
            """Hook _build_once runs before loading. Identity unless a Split class overrides it;
            ``cache_dir`` is where an override may write a converted copy."""
            return checkpoint_path

        def prepare_preprocessing(self):
            self.frontend_preprocessor = deepcopy(self.data_preprocessor)
            self.data_preprocessor = PassThroughPreprocessor()

        def extract_feat(self, batch_inputs):
            """Extract features (the backbone resumes at cut_point)."""
            return _extract_feat_from_split(self, batch_inputs)

        def feature_frontend(self, data, preprocessed: bool = False):
            return _feature_frontend(self, data, preprocessed=preprocessed)

        def feature_backbone(self, data, use_neck: bool = False):
            '''Takes the output of feature_frontend and extracts the full set of backbone features.'''
            return _extract_feat_from_split(self, data['inputs'], use_neck=use_neck)

        # The public surface is backend(data, *, mode, **kwargs); the three modes live in the private
        # _backend_* methods below (a Split class may override them). Replaces the former public
        # backend_inference / backend_raw / backend_loss.
        def backend(self, data, *, mode=None, **kwargs):
            """Run the model from the cut onwards on ``feature_frontend``'s output.

            ``mode`` (required keyword): 'raw' (the backend without the test-time wrapper),
            'loss' (parsed task loss; ``individual_losses=True`` returns one loss per image) or
            'inference' (``test_step``; returns the predictions). Other keywords go to the mode's method.

            The frontend runs once on the batch and the backend per image, so with unfrozen norms the
            frontend's normalisation statistics come from the whole batch. Users who need exact per-image
            behaviour including norms should run ``feature_frontend`` per image too.
            """
            if mode == 'raw':
                return self._backend_raw(data, **kwargs)
            if mode == 'loss':
                return self._backend_loss(data, **kwargs)
            if mode == 'inference':
                return self._backend_inference(data, **kwargs)
            raise ValueError(f"backend: mode must be one of 'raw', 'loss', 'inference'; got {mode!r}")

        def _backend_inference(self, data):
            return self.test_step(data)

        def _backend_raw(self, data):
            reason = _raw_unsupported_reason(self)
            if reason:
                raise NotImplementedError(
                    f"backend(mode='raw') is not available for {type(self).__name__}: raw (tensor-mode) output "
                    f"does not work because OpenMMLab's own forward(mode='tensor') is broken for this model "
                    f"family ({reason}). Use backend(mode='loss') or backend(mode='inference') instead.")
            return self(data['inputs'], data['data_samples'])

        def get_single_input_loss(self, x, s):
            return self.parse_losses(self.loss(x, [s]))[0]

        def _backend_loss(self, data, individual_losses: bool = False):
            if individual_losses:
                # Batch-1 items either way: the Split*Features containers yield out[i:i+1]; the bare
                # preprocessed batch the frontend returns at cut_point=0 is split the same way.
                inputs = data['inputs']
                items = inputs.split(1) if isinstance(inputs, torch.Tensor) else inputs
                individual_losses = [self.get_single_input_loss(x, s)
                                     for x, s in zip(items, data['data_samples'])]
                return torch.stack(individual_losses)
            return self.parse_losses(self.loss(data['inputs'], data['data_samples']))

        new_cls.__init__ = __init__
        new_cls.cut_point = cut_point
        new_cls.extract_feat = extract_feat
        # attach only what the class does not define itself (RepPointsV2 defines prepare_checkpoint)
        for name, member in (('create_from_cfg_and_checkpoint', create_from_cfg_and_checkpoint),
                             ('prepare_checkpoint', prepare_checkpoint),
                             ('prepare_preprocessing', prepare_preprocessing),
                             ('feature_frontend', feature_frontend),
                             ('feature_backbone', feature_backbone),
                             ('backend', backend),
                             ('_backend_inference', _backend_inference),
                             ('_backend_raw', _backend_raw),
                             ('_backend_loss', _backend_loss),
                             ('get_single_input_loss', get_single_input_loss)):
            if name not in dir(new_cls):
                setattr(new_cls, name, member)
        new_cls._cfm_package = package
        # Assigned UNCONDITIONALLY: a `not in dir(new_cls)` guard is always False for train(),
        # which every nn.Module inherits. new_cls is the generated <Pkg>Split<Name> subclass (the
        # decorated class itself only when its name already contains 'Split'), so
        # super(new_cls, self) resolves to the mm base class and its own train() runs first.
        new_cls.freeze_norms = freeze_norms_default
        new_cls.train = _make_split_train(new_cls)
        return new_cls

    return decorator


class PassThroughPreprocessor(nn.Module):
    """The backend's data_preprocessor: returns ``data`` unchanged (the frontend has already preprocessed it)."""

    def forward(self, data, training=False):
        return data

# --------------------------------------------------------------------------------------
# Frozen-task-model norm handling.
#
# MECHANISM lives on the split model (NetworkSplitter attaches a train() override);
# POLICY lives with whoever is training. The splitter defaults to freeze_norms=False, so it
# stays purely structural -- anyone using it to train a split model end to end gets stock
# semantics and never needs to know the flag exists. Code that trains something else around a
# frozen task model (e.g. a feature codec at the cut) opts in by setting ``freeze_norms = True``.
#
# Mirrors mmdet's own idiom (ResNet.train() -> `if mode and self.norm_eval: m.eval()`),
# generalised from the backbone to the whole model and from _BatchNorm to every
# stat-tracking norm.
# --------------------------------------------------------------------------------------
def freeze_norm_layers(task_model):
    """Put every stat-tracking norm layer into eval mode.

    Uses ``.eval()`` only, never ``track_running_stats = False``: that attribute is a
    construction-time selector ("always use batch statistics"), not a freeze switch, and changing
    it on a built InstanceNorm makes it normalise from the input and overwrite its running stats.
    You rarely call this directly: set ``freeze_norms = True`` on the split model and its own
    ``train()`` applies it on every mode change.

    Matches ``_NormBase`` (BatchNorm1d/2d/3d and SyncBatchNorm -- a *sibling* subclass of
    _BatchNorm, so ``isinstance(m, nn.BatchNorm2d)`` misses it) plus ``_InstanceNorm``.
    """
    for mod in task_model.modules():
        if isinstance(mod, (_NormBase, _InstanceNorm)):
            mod.eval()


@contextmanager
def task_model_mode(task_model, *, training: bool):
    """Temporarily put the task model in loss (``training=True``) or inference (``False``) mode.

    The previous mode is restored on exit, also when the body raises (the exception propagates
    unchanged). Both directions go through the split model's own ``train()``, which re-applies the
    norm freeze whenever ``freeze_norms`` is set.

    Usage::

        with task_model_mode(model, training=False):
            preds = model.backend(features, mode='inference')
    """
    was = task_model.training
    task_model.train(training)
    try:
        yield
    finally:
        task_model.train(was)


def to_device(data_sample:Union[dict,torch.Tensor], device):
    # Build the set of types that support .to(device) — only check against available packages
    _movable_types = [torch.Tensor]
    if MMDET_AVAILABLE:
        _movable_types.append(DetDataSample)
    if MMSEG_AVAILABLE:
        _movable_types.append(SegDataSample)
    _movable_types = tuple(_movable_types)

    if isinstance(data_sample, dict):
        for k,v in data_sample.items():
            if isinstance(v, _movable_types):
                data_sample[k] = v.to(device)
            elif isinstance(v, list):
                new_v = [to_device(v2, device) for v2 in v]
                data_sample[k] = new_v
            else:
                print(f'{k} could not be sent to {device}')
    elif isinstance(data_sample, _movable_types):
        data_sample = data_sample.to(device)
    elif isinstance(data_sample, list):
        data_sample = [to_device(v, device) for v in data_sample]
    else:
        print(f'input could not be sent to {device}')
    return data_sample
