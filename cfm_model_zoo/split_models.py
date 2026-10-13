import copy
import hashlib
import importlib
import json
import os
import re
import sys
import uuid
import warnings
from typing import Optional, Union

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.registry import MODELS as _ROOT_MODELS

from .naming import PACKAGE_PREFIX, split_class_name
from .split_utils import NetworkSplitter

# =============================================================================
# Transformers compatibility patches
#
# mmdet's vlfuse_helper.py (used by GLIP) imports from transformers at module
# load time. Some of these imports broke in transformers 5.x. The patches below
# fix those breakages without modifying mmdet or transformers source files.
#
# To bypass a patch in the future (e.g. if a transformers upgrade fixes the
# upstream issue), simply remove or guard the relevant block with a version
# check: `if transformers.__version__ < "X.Y"`.
#
# NOTE: PATCH 1 (apply_chunking_to_forward) lives in cfm_model_zoo/__init__.py
# so it runs before any mmdet submodule is imported.
# =============================================================================

# Lazy imports: mmdet and mmseg are only required when their respective model classes are used. The
# ImportError is kept (as LEGACY_IMPORT_ERROR is below) so a missing extra is reported with its real cause.
MMDET_IMPORT_ERROR: Optional[ImportError] = None
MMSEG_IMPORT_ERROR: Optional[ImportError] = None
try:
    from mmdet.models.detectors import (ATSS, CascadeRCNN, DDQDETR, DeformableDETR, DINO, FasterRCNN, FCOS,
                                        GLIP, GroundingDINO, Mask2Former, MaskRCNN, RetinaNet)
    from mmdet.models.language_models.bert import BertEncoder as _MMDetBertEncoder

    # PATCH 2: batch_encode_plus removed in transformers 5.x
    #
    # mmdet's BertModel.forward (language_models/bert.py) calls
    # tokenizer.batch_encode_plus(...) which was removed in transformers 5.x.
    # PreTrainedTokenizerBase.__call__ is functionally identical — it accepts
    # the same arguments and returns the same output.
    #
    # To bypass: remove this block once mmdet updates its tokenizer call.
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, 'batch_encode_plus'):
        PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

    # PATCH 3: 2D/3D attention mask → 4D for transformers 5.x SDPA compatibility
    #
    # mmdet's BertEncoder.forward passes a 2D mask [bs, seq] (standard BERT) or
    # a 3D mask [bs, seq, seq] (GroundingDINO's sub-sentence representation) to
    # HuggingFace BertModel. In transformers 5.x, these trigger a broken code
    # path in sdpa_mask that produces a 5D tensor, causing a runtime expand error.
    #
    # Root cause: transformers 5.x's _preprocess_mask_arguments has an early-exit
    # for 4D masks that bypasses sdpa_mask entirely. Expanding to 4D here triggers
    # that early-exit and avoids the bug without changing model outputs.
    #
    # The .bool() cast is required because PyTorch SDPA expects bool or float
    # masks (not int64). mmdet convention: 1=attend, 0=ignore, matching SDPA's
    # True=attend convention.
    #
    # To bypass: remove this block once transformers fixes the 5D expand bug in
    # sdpa_mask, or if mmdet updates its BertEncoder to pass 4D masks natively.
    if not getattr(_MMDetBertEncoder, '_cfm_patched', False):
        _orig_bert_encoder_fwd = _MMDetBertEncoder.forward

        def _patched_bert_encoder_fwd(self, x):
            mask = x.get('attention_mask')
            if mask is not None:
                if mask.ndim == 2:
                    seq_len = mask.shape[1]
                    x = {**x, 'attention_mask': mask[:, None, None, :].expand(-1, 1, seq_len, -1).bool()}
                    result = _orig_bert_encoder_fwd(self, x)
                    result['embedded'] = result['embedded'] * mask.unsqueeze(-1).float()
                    result['masks'] = mask
                    return result
                elif mask.ndim == 3:
                    x = {**x, 'attention_mask': mask[:, None, :, :].bool()}
                    result = _orig_bert_encoder_fwd(self, x)
                    result['masks'] = mask  # restore 3D mask for downstream GroundingDINO layers
                    return result
            return _orig_bert_encoder_fwd(self, x)

        _MMDetBertEncoder.forward = _patched_bert_encoder_fwd
        _MMDetBertEncoder._cfm_patched = True

    MMDET_AVAILABLE = True
except ImportError as _mmdet_error:
    MMDET_AVAILABLE = False
    MMDET_IMPORT_ERROR = _mmdet_error

try:
    from mmseg.models.segmentors import EncoderDecoder
    MMSEG_AVAILABLE = True
except ImportError as _mmseg_error:
    MMSEG_AVAILABLE = False
    MMSEG_IMPORT_ERROR = _mmseg_error

# legacy/ (the RepPointsV2 port) gets its own try: an ImportError inside it (e.g. an mmcv build
# without CornerPool) must cost RepPointsV2 alone. Inside the mmdet try it set MMDET_AVAILABLE=False
# and silently emptied every detector.
LEGACY_IMPORT_ERROR: Optional[ImportError] = None
if MMDET_AVAILABLE:
    try:
        from .legacy import RepPointsV2MaskDetector, convert_swin_checkpoint_file
    except ImportError as _legacy_error:
        LEGACY_IMPORT_ERROR = _legacy_error
        warnings.warn(f'cfm_model_zoo.legacy failed to import ({type(_legacy_error).__name__}: '
                      f'{_legacy_error}); RepPointsV2 is unavailable. Every other model is unaffected.',
                      UserWarning, stacklevel=2)
LEGACY_AVAILABLE = MMDET_AVAILABLE and LEGACY_IMPORT_ERROR is None
# RepPointsV2's checkpoint conversion (prepare_checkpoint below). The names its output takes:
_ALREADY_CONVERTED = re.compile(r'_converted(-[0-9a-f]{8})?\.pth$')
# Bump whenever legacy's convert_swin_checkpoint_file / swin_converter changes what it writes: a cached
# conversion recorded under another version is not reused.
_REPPOINTS_CONVERTER_VERSION = 1

# =========================================================================
# The one hand-written Split class: RepPointsV2 is a legacy port (mmdet-2 style loss/inference
# signatures, Swin checkpoint conversion). Everything else is generated from the exact class.
# =========================================================================
if LEGACY_AVAILABLE:
    @NetworkSplitter()
    class MMDetSplitRepPointsV2MaskDetector(RepPointsV2MaskDetector):
        '''
        MMDetSplitRepPointsV2MaskDetector is a class that allows spliting a RepPointsV2MaskDetector model at the input cut_point of the backbone
        Requires special modifications due to RepPointsV2MaskDetector utilizng older version of mmdet and lack of backward compatibility see legacy folder for more details
        Created using the NetworkSplitter automatic network splitter
        '''
        @classmethod
        def prepare_checkpoint(cls, checkpoint_path, cache_dir=None):
            """The legacy port's checkpoint needs its Swin keys converted once. An already converted file, e.g.
            a previous output of this hook (a name ending in ``_converted.pth`` or ``_converted-<8 hex>.pth``),
            is used as it is. Otherwise the converted copy is written into the cache directory (``cache_dir``,
            default resolve_download_path()), never next to the user's file, and reused while it was made from
            the same source bytes by the same converter version (a read-only cache serves it too)."""
            name = os.path.basename(checkpoint_path)
            if _ALREADY_CONVERTED.search(name):
                return checkpoint_path
            from .directory import _require_writable_cache, resolve_download_path   # directory imports this module
            cache_dir = resolve_download_path(cache_dir)
            stem = name[:-len('.pth')] if name.endswith('.pth') else name
            dst = os.path.join(cache_dir, f'{stem}_converted.pth')
            # The converted file is named after ITS hash, so the cache is keyed on the SOURCE's: a record
            # next to it says which source bytes it came from, and a matching, intact conversion is reused.
            record = dst + '.source.json'
            digest = hashlib.sha256()
            with open(checkpoint_path, 'rb') as f:
                for block in iter(lambda: f.read(1 << 24), b''):
                    digest.update(block)
            source_sha256 = digest.hexdigest()
            try:
                with open(record) as f:
                    previous = json.load(f)
            except (OSError, ValueError):
                previous = None
            if isinstance(previous, dict):          # anything malformed counts as no record
                done = previous.get('converted')
                if (isinstance(done, str) and previous.get('source_sha256') == source_sha256
                        and previous.get('converter_version') == _REPPOINTS_CONVERTER_VERSION
                        and os.path.isfile(done) and os.path.getsize(done) == previous.get('converted_size')):
                    return done
            _require_writable_cache(cache_dir)      # only now: a read-only cache can still serve a conversion
            converted = convert_swin_checkpoint_file(checkpoint_path, dst)
            tmp = f'{record}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp'   # unique across hosts sharing the cache
            try:
                with open(tmp, 'w') as f:
                    json.dump({'source_sha256': source_sha256, 'converter_version': _REPPOINTS_CONVERTER_VERSION,
                               'converted': converted, 'converted_size': os.path.getsize(converted)}, f)
                os.replace(tmp, record)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return converted

        # mmdet-3 signatures over the port's mmdet-2 ones, so the generic _backend_loss /
        # _backend_inference / per-image loop that every other Split model uses run this one too.
        @staticmethod
        def _legacy_scale_factor(scale_factor):
            """mmdet-3 stores (w, h); the port's head expects (w, h, w, h). Idempotent."""
            return tuple(scale_factor) * 2 if len(scale_factor) == 2 else tuple(scale_factor)

        def loss(self, batch_inputs, batch_data_samples):
            """mmdet-3 loss -> the port's forward_train. Reads copies of the metainfo; the samples are not changed."""
            return self.forward_train(
                batch_inputs,
                [{**ds.metainfo, 'scale_factor': self._legacy_scale_factor(ds.scale_factor)}
                 for ds in batch_data_samples],
                [ds.gt_instances.bboxes for ds in batch_data_samples],
                [ds.gt_instances.labels for ds in batch_data_samples],
                gt_bboxes_ignore=[ds.ignored_instances.bboxes for ds in batch_data_samples],
                gt_masks=[ds.gt_instances.masks for ds in batch_data_samples],
                gt_sem_map=[ds.gt_sem.sem_map for ds in batch_data_samples],
                gt_sem_weights=[ds.gt_sem.sem_weights for ds in batch_data_samples])

        def predict(self, batch_inputs, batch_data_samples):
            """The port reads scale_factor from the samples' metainfo (it also writes pred_instances there)."""
            for ds in batch_data_samples:
                if 'scale_factor' in ds:
                    ds.set_metainfo({'scale_factor': self._legacy_scale_factor(ds.scale_factor)})
            return super().predict(batch_inputs, batch_data_samples)

SUPPORTED_MODELS = tuple(
    (([MaskRCNN, RetinaNet, CascadeRCNN, Mask2Former, GroundingDINO, GLIP]
      + ([RepPointsV2MaskDetector] if LEGACY_AVAILABLE else [])
      + [ATSS, FCOS, FasterRCNN, DDQDETR, DINO, DeformableDETR]) if MMDET_AVAILABLE else [])
    + ([EncoderDecoder] if MMSEG_AVAILABLE else []))

_HAND_WRITTEN = {RepPointsV2MaskDetector: MMDetSplitRepPointsV2MaskDetector} if LEGACY_AVAILABLE else {}
_SPLIT_CLASSES = {}                         # exact model class -> its Split class
_SPLIT_IMPORT = 'cfm_model_zoo.split_models'
# A Split class name: <Pkg>Split<Class> (MMDetSplitMaskRCNN), or the name used before it, Split<Class>. The old
# names stay mapped: a module attribute and a root-registry name for every Split class (saved configs, pickles).
_SPLIT_NAME = re.compile(r'^(?P<prefix>' + '|'.join(PACKAGE_PREFIX.values()) + r')?Split(?P<base>[A-Z]\w*)$')


def _package_registries() -> dict:
    """{package: <package>.registry.MODELS} for every installed mm package we split."""
    registries = {}
    for package in ('mmdet', 'mmseg'):
        try:
            registries[package] = importlib.import_module(f'{package}.registry').MODELS
        except ImportError:
            pass
    return registries


def _package_of(cls: type) -> str:
    registries = _package_registries()
    top = cls.__module__.split('.')[0]
    if top in registries:
        return top
    for package, registry in registries.items():
        registry.import_from_location()
        if cls in registry.module_dict.values():
            return package
    raise TypeError(f'{cls.__module__}.{cls.__qualname__} is not a model class of mmdet or mmseg (not in '
                    f'their MODELS registries); pass package= explicitly.')


def _register_split_class(split_cls: type, base: type, package: str) -> type:
    name = split_cls.__name__
    existing = _ROOT_MODELS.module_dict.get(name)
    if existing is not None and getattr(existing, '_cfm_base', None) is not base:
        raise ValueError(f"Cannot register {name} for {base.__module__}.{base.__qualname__}: mmengine's root "
                         f"MODELS registry already maps {name} to {existing.__module__}.{existing.__qualname__}.")
    split_cls.__module__ = __name__
    split_cls.__qualname__ = name
    split_cls._cfm_base = base
    split_cls._cfm_package = package
    setattr(sys.modules[__name__], name, split_cls)
    _ROOT_MODELS.register_module(name=name, module=split_cls, force=True)
    # The legacy name Split<Class>, for code and saved configs written before the package prefix. Unlike the
    # new name it carries no package, so where two packages share a class name the first one keeps it.
    old = f'Split{base.__name__}'
    taken = _ROOT_MODELS.module_dict.get(old)
    if taken is None or getattr(taken, '_cfm_base', None) is base:
        setattr(sys.modules[__name__], old, split_cls)
        _ROOT_MODELS.register_module(name=old, module=split_cls, force=True)
    _SPLIT_CLASSES[base] = split_cls
    return split_cls


def register_split_model(cls: type, package: Optional[str] = None) -> type:
    """Generate (or return the cached) Split class for the EXACT model class ``cls``.

    Named ``<Pkg>Split<Name>`` (``MMDetSplitMaskRCNN``). It is registered in mmengine's root MODELS registry,
    so ``type='<Pkg>Split<Name>'`` (or the old ``'Split<Name>'``) builds from any package scope, and published
    as an attribute of this module, so pickling resolves it. Public:
    call it to add a model class outside SUPPORTED_MODELS without the 'untested' warning.
    """
    base = getattr(cls, '_cfm_base', cls)
    if base in _SPLIT_CLASSES:
        return _SPLIT_CLASSES[base]
    package = package or _package_of(base)
    split_cls = _HAND_WRITTEN.get(base) or NetworkSplitter(package=package)(base)
    return _register_split_class(split_cls, base, package)


def get_split_class(cls: type, package: Optional[str] = None) -> type:
    """The Split class for ``cls``; generated on demand, with a warning, if ``cls`` is untested."""
    base = getattr(cls, '_cfm_base', cls)
    if base not in _SPLIT_CLASSES and base not in SUPPORTED_MODELS:
        warnings.warn(f'{base.__module__}.{base.__qualname__} is not in SUPPORTED_MODELS: generating '
                      f'{split_class_name(base.__name__, package or _package_of(base))} on demand. This model '
                      f'type is untested; check its outputs before relying on them.', UserWarning, stacklevel=2)
    return register_split_model(base, package)


def _resolve_model_class(model_type, package: str) -> type:
    """cfg.model.type (name, scoped name, Split name or class) -> the exact mm model class."""
    if isinstance(model_type, type):
        return getattr(model_type, '_cfm_base', model_type)
    if not isinstance(model_type, str):
        raise TypeError(f'cfg.model.type must be a str or a class, got {type(model_type).__name__}')
    short = model_type.split('.')[-1]
    known = _ROOT_MODELS.module_dict.get(short)
    if known is not None and hasattr(known, '_cfm_base'):
        return known._cfm_base
    cls = importlib.import_module(f'{package}.registry').MODELS.get(model_type)
    if cls is None and _SPLIT_NAME.match(short):
        cls = getattr(sys.modules[__name__], short, None)    # generated in another process
    if cls is None:
        raise KeyError(f'{model_type!r} is not a model type registered in {package}.registry.MODELS')
    return getattr(cls, '_cfm_base', cls)


def to_split_config(cfg: Union[Config, str, os.PathLike], cut_point: int, package: str) -> Config:
    """The one config rewrite, and the only one.

    Deep-copies, sets model.type to the Split class of the EXACT configured class, sets
    model.cut_point, and appends this module to custom_imports (keeping any existing imports) so a
    saved config rebuilds on its own. Nothing else changes. Never mutates its input; idempotent.
    """
    cfg = Config.fromfile(cfg) if isinstance(cfg, (str, os.PathLike)) else copy.deepcopy(cfg)
    split_cls = get_split_class(_resolve_model_class(cfg.model.type, package), package)
    cfg.model.type = split_cls.__name__
    cfg.model.cut_point = cut_point
    existing = cfg.get('custom_imports') or {}
    imports = existing.get('imports', [])
    imports = [imports] if isinstance(imports, str) else list(imports)
    if _SPLIT_IMPORT not in imports:
        imports.append(_SPLIT_IMPORT)
    cfg.custom_imports = dict(imports=imports, allow_failed_imports=existing.get('allow_failed_imports', False))
    return cfg


def __getattr__(name):
    """Regenerate an on-demand Split class in a fresh process (unpickling, saved configs), by its name
    ``<Pkg>Split<Class>`` (searched in that package only) or its legacy name ``Split<Class>`` (every package).

    Only for MODEL classes (subclasses of mmengine's BaseModel): the package MODELS registries also
    hold backbones, necks and heads, and a probe such as ``MMDetSplitResNet`` must raise AttributeError,
    not register a Split class built from a backbone."""
    m = _SPLIT_NAME.match(name)
    if m is not None:
        wanted = {p for p, prefix in PACKAGE_PREFIX.items() if prefix == m['prefix']} if m['prefix'] else None
        for package, registry in _package_registries().items():
            if wanted is not None and package not in wanted:
                continue
            base = registry.get(m['base'])
            if (isinstance(base, type) and issubclass(base, BaseModel)
                    and not hasattr(base, '_cfm_base')):
                return get_split_class(base, package)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


for _base in SUPPORTED_MODELS:
    register_split_model(_base)
globals().pop('_base', None)   # never bound when SUPPORTED_MODELS is empty (neither mmdet nor mmseg imports)

# What `from cfm_model_zoo.split_models import *` provides: the Split classes registered above (generated, so
# listed here rather than written out), the optional-dependency flags and the public functions. Split classes
# generated on demand later (get_split_class, unpickling) are reached by attribute, not by the star import.
__all__ = sorted({c.__name__ for c in _SPLIT_CLASSES.values()} | {c.__name__ for c in _HAND_WRITTEN.values()}) + [
    'MMDET_AVAILABLE', 'MMSEG_AVAILABLE', 'LEGACY_AVAILABLE',
    'MMDET_IMPORT_ERROR', 'MMSEG_IMPORT_ERROR', 'LEGACY_IMPORT_ERROR',
    'SUPPORTED_MODELS', 'register_split_model', 'get_split_class', 'to_split_config',
]
