"""Split backbones.

A backbone is split by re-classing it IN PLACE onto a generated class ``<Pkg>Split<ExactClass>``
(``MMDetSplitResNet``, ``MMSegSplitSwinTransformer``) whose MRO is ``(<Family>SplitMixin, <ExactClass>)``.
The mixins hold only ``split_forward`` and the ``num_split_stages`` property: no ``__init__`` and no
attributes, so swapping ``__class__`` on an already-built backbone skips nothing. The class is generated
from the EXACT backbone class (never a base class, which would drop a subclass's overrides), cached, and
published as an attribute of this module under its own name so pickling resolves it; ``__getattr__``
regenerates it in a fresh process. The package prefix is needed because mmdet and mmseg both define
``SwinTransformer``, ``ResNet``, ``ResNetV1d`` and ``ResNeXt``, and pickle finds a class by ``module.name``.
The names used before, ``Split<ExactClass>_<pkg>``, still resolve to the same classes (old pickles load).

``split_forward`` is OUR re-implementation of a backbone's ``forward()``, cut in two. Only classes it
has been verified against may be split (VERIFIED_BACKBONES), plus subclasses that inherit a verified
``forward()`` unchanged; anything else raises ``TypeError`` at construction rather than silently
computing something other than the model's own forward. A subclass of one of the generated Split
classes must override ``split_forward`` or guarantee it stays correct: that is on the subclasser.

Pickling across processes: ``__getattr__`` regenerates only ``MMDetSplit<Name>`` / ``MMSegSplit<Name>``
(the class is looked up in that package's MODELS registry). An accepted backbone class from any other
package (e.g. a library user's own ResNet subclass in ``my_models``, named ``MyModelsSplit<Name>``) is
split and pickles within the process that generated it, but a fresh process cannot resolve it by name.
Such classes are not rejected.
"""
import importlib
import re
import sys
from typing import List, Optional, Tuple, Union

import torch

from .naming import split_class_name

try:
    from mmdet.models.backbones import ResNet as _MMDetResNet, SwinTransformer as _MMDetSwin
    MMDET_AVAILABLE = True
except ImportError:
    _MMDetResNet = _MMDetSwin = None
    MMDET_AVAILABLE = False

try:
    from mmseg.models.backbones import (ResNet as _MMSegResNet, ResNetV1c as _MMSegResNetV1c,
                                        SwinTransformer as _MMSegSwin)
    MMSEG_AVAILABLE = True
except ImportError:
    _MMSegResNet = _MMSegResNetV1c = _MMSegSwin = None
    MMSEG_AVAILABLE = False


class SwinSplitMixin:
    """``split_forward`` for mmdet's and mmseg's ``SwinTransformer``. Stateless by contract (no
    ``__init__``, no attributes). No ``__slots__``: it would break the in-place ``__class__`` swap."""

    @property
    def num_split_stages(self) -> int:
        """The number of stages a cut can follow (cut_point 1..num_split_stages)."""
        return len(self.stages)

    def split_forward(self,
                      x: torch.Tensor,
                      hw_shape: Tuple = (0, 0),
                      outs: List[torch.Tensor] = [],
                      input_layer: int = 0,
                      output_layer: Optional[int] = None) -> Union[Tuple[Tuple, List[torch.Tensor]], List[torch.Tensor]]:
        if input_layer == 0:
            x, hw_shape = self.patch_embed(x)

            if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)

        for i, stage in enumerate(self.stages):
            if i < input_layer - 1:
                continue
            if i == input_layer - 1:
                out = outs[-1]
                outs = outs[:-1]
                out_hw_shape = hw_shape
                if stage.downsample != None:  # noqa: E711 -- kept verbatim from SplitSwinTransformer
                    x, hw_shape = stage.downsample(out, out_hw_shape)
                else:
                    x, hw_shape = out, out_hw_shape
            else:
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i == output_layer:
                return out_hw_shape, outs + [out]
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs = outs + [out]
        return outs


class ResNetSplitMixin:
    """``split_forward`` for mmdet's and mmseg's ``ResNet`` and the subclasses that inherit its
    ``forward()`` (V1c, V1d, ResNeXt).
    Stateless by contract. Uses mmdet's non-mutating slice: mmseg's version did ``outs.pop(-1)``,
    which mutated the caller's ``SplitResNetFeatures.outs`` list."""

    @property
    def num_split_stages(self) -> int:
        """The number of stages a cut can follow (cut_point 1..num_split_stages)."""
        return len(self.res_layers)

    def split_forward(self,
                      x: Optional[torch.Tensor],
                      outs: List[torch.Tensor] = [],
                      input_layer: int = 0,
                      output_layer: Optional[int] = None) -> Union[Tuple[Tuple, List[torch.Tensor]], List[torch.Tensor]]:
        """Forward function."""

        if input_layer == 0:
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            if i < input_layer - 1:
                continue
            if i == input_layer - 1 and x is None:
                if i not in self.out_indices:
                    x = outs[-1]
                    outs = outs[:-1]
                else:
                    x = outs[-1]
                continue
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == output_layer:
                return tuple(outs + [x])

            if i in self.out_indices:
                outs = outs + [x]

        return tuple(outs)


SPLIT_MIXINS = (SwinSplitMixin, ResNetSplitMixin)

# Verified backbone classes -> the mixin whose split_forward was verified against them. Verified means:
# frontend + backend (split_forward up to the cut, then resumed) is bit-identical to the class's OWN
# forward() at every cut point 0..max and at an odd input size. Add a class only after repeating that
# comparison for it.
VERIFIED_BACKBONES = {cls: mixin for cls, mixin in (
    (_MMDetResNet, ResNetSplitMixin),
    (_MMSegResNet, ResNetSplitMixin),
    (_MMSegResNetV1c, ResNetSplitMixin),
    (_MMDetSwin, SwinSplitMixin),
    (_MMSegSwin, SwinSplitMixin),
) if cls is not None}
_SPLIT_BACKBONE_CLASSES = {}   # exact backbone class -> generated split class


def _qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def _forward_owner(cls: type) -> type:
    """The nearest class in ``cls``'s MRO that defines ``forward``: the forward() ``cls`` actually runs."""
    return next(c for c in cls.__mro__ if 'forward' in vars(c))


def split_backbone_class_name(base: type) -> str:
    """``<Pkg>Split<Class>``: ``MMDetSplitResNet`` for mmdet's ResNet."""
    return split_class_name(base.__name__, base.__module__.split('.')[0])


def _old_split_backbone_class_name(base: type) -> str:
    """The name used before ``<Pkg>Split<Class>``, kept as an alias: ``SplitResNet_mmdet``."""
    return f'Split{base.__name__}_{base.__module__.split(".")[0]}'


def get_split_backbone_class(base: type) -> type:
    """The generated split class for the EXACT backbone class ``base`` (cached).

    Accepted iff ``base`` itself is in VERIFIED_BACKBONES, or the nearest class in its MRO that
    defines forward() is (a verified ancestor whose forward() ``base`` inherits unchanged: ResNeXt,
    ResNetV1d). The exact class's own entry wins when both are verified. A class
    that overrides forward() (RegNet, DetectoRS_ResNet) or belongs to no verified family (MobileNetV2)
    raises TypeError: split_forward follows the verified class's forward and could silently differ.
    """
    if issubclass(base, SPLIT_MIXINS):
        return base   # already split (a subclass of a Split class keeps split_forward correct itself)
    if base in _SPLIT_BACKBONE_CLASSES:
        return _SPLIT_BACKBONE_CLASSES[base]
    owner = _forward_owner(base)
    mixin = VERIFIED_BACKBONES.get(base) or VERIFIED_BACKBONES.get(owner)
    if mixin is None:
        via = '' if owner is base else f' (it runs the forward() of {_qualname(owner)})'
        raise TypeError(
            f'Backbone {_qualname(base)} cannot be split: split_forward has not been verified against its '
            f'forward(){via}. Verify that the split composition is bit-identical to forward() at every cut '
            f'point, then add the class to VERIFIED_BACKBONES in cfm_model_zoo/split_backbones.py. '
            f'Verified: {", ".join(sorted(_qualname(c) for c in VERIFIED_BACKBONES))}.')
    name = split_backbone_class_name(base)
    cls = type(name, (mixin, base), {'__module__': __name__, '__qualname__': name})
    setattr(sys.modules[__name__], name, cls)
    setattr(sys.modules[__name__], _old_split_backbone_class_name(base), cls)   # the legacy name, same class
    _SPLIT_BACKBONE_CLASSES[base] = cls
    return cls


def split_backbone_in_place(backbone: torch.nn.Module) -> int:
    """Re-class an already-built backbone onto its split class (nothing is rebuilt or copied) and return
    its number of stages, ``num_split_stages``: the valid cut points are 0..that number."""
    backbone.__class__ = get_split_backbone_class(type(backbone))
    return backbone.num_split_stages


_GENERATED_NAME = re.compile(r'^(?P<prefix>MMDet|MMSeg)Split(?P<base>\w+)$')
_OLD_GENERATED_NAME = re.compile(r'^Split(?P<base>\w+)_(?P<pkg>mmdet|mmseg)$')
_PREFIX_PACKAGE = {'MMDet': 'mmdet', 'MMSeg': 'mmseg'}


def __getattr__(name):
    """Regenerate ``<Pkg>Split<Name>`` (or its legacy name ``Split<Name>_<pkg>``) on first access in a fresh
    process (unpickling)."""
    m = _GENERATED_NAME.match(name)
    pkg = _PREFIX_PACKAGE[m['prefix']] if m is not None else None
    if m is None:
        m = _OLD_GENERATED_NAME.match(name)
        pkg = m['pkg'] if m is not None else None
    if m is not None:
        base = importlib.import_module(f'{pkg}.registry').MODELS.get(m['base'])
        if isinstance(base, type) and base.__module__.split('.')[0] == pkg:
            try:
                return get_split_backbone_class(base)
            except TypeError:   # not splittable: no such class, and hasattr() must stay False
                pass
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

