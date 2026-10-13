"""Automatic names, ``<Pkg>Split<Name>[<Backbone>]``.

- **Classes** (no backbone): a Split model class is ``<Pkg>Split<Class>`` (``MMDetSplitMaskRCNN``,
  ``MMSegSplitEncoderDecoder``) and a split backbone class ``<Pkg>Split<Backbone>`` (``MMDetSplitResNet``):
  ``split_class_name``.
- **Registered names** are a function of the config alone (``registered_name``): the class name's package
  prefix + ``Split`` + METHOD + BACKBONE.

  * METHOD is the config's ``model.type``, except for mmseg's generic containers (``EncoderDecoder``,
    ``CascadeEncoderDecoder``), whose method lives in the decode head: then it is the config's collection in
    mmseg's own model-zoo metafile (``In Collection``, sanitised: ``DeepLabV3+`` -> ``DeepLabV3Plus``), plus the
    decode head's type minus ``Head`` when that collection spans more than one decode-head type in the metafile
    (KNet: the head inside its ``kernel_generate_head``). Whether a collection has several is read from the
    metafile, never from this library's registry, so adding a model never renames another. A config with no
    metafile entry falls back to its class.
  * BACKBONE comes from the backbone config: ``ResNet*`` -> ``R<depth>``, ``ResNeXt`` -> ``X<depth>``,
    ``SwinTransformer`` -> ``SwinT``/``SwinS``/``SwinB``/``SwinL`` (embed_dims and depths), anything else its type.

The metafile is indexed by each model's ``Config:`` field as well as by its ``Name``: the two can differ (mmdet
3.3 names the model of ``faster-rcnn_r50_fpn_ms-3x_coco.py`` ``faster-rcnn_r50_fpn_mstrain_3x_coco``).
"""
import functools
import importlib
import os
import re
import warnings
from typing import Dict, Optional, Tuple

PACKAGE_PREFIX = {'mmdet': 'MMDet', 'mmseg': 'MMSeg'}
_GENERIC_CONTAINERS = {'mmseg': ('EncoderDecoder', 'CascadeEncoderDecoder')}
_SWIN_SIZES = {(96, (2, 2, 6, 2)): 'T', (96, (2, 2, 18, 2)): 'S', (128, (2, 2, 18, 2)): 'B',
               (192, (2, 2, 18, 2)): 'L'}
_SWIN_DEFAULTS = (96, (2, 2, 6, 2))       # mmdet's and mmseg's SwinTransformer defaults
_SPLIT_PREFIX = re.compile(r'^(?:MMDet|MMSeg)?Split(?=[A-Z])')   # a rewritten config's Split class


def package_prefix(package: str) -> str:
    """'mmdet' -> 'MMDet', 'mmseg' -> 'MMSeg'; any other package name in CamelCase ('my_models' -> 'MyModels')."""
    return PACKAGE_PREFIX.get(package) or ''.join(p[:1].upper() + p[1:] for p in re.split(r'[^0-9A-Za-z]+', package))


def split_class_name(name: str, package: str) -> str:
    """The Split class name of the class ``name`` from ``package``: ``MMDetSplitMaskRCNN``, ``MMDetSplitResNet``."""
    return f'{package_prefix(package)}Split{name}'


def sanitise(text: str) -> str:
    """A metafile name as part of a class name: '+' -> 'Plus', everything else that is not a letter or digit dropped."""
    return re.sub(r'[^0-9A-Za-z]', '', text.replace('+', 'Plus'))


def _type_name(kind) -> str:
    name = kind.__name__ if isinstance(kind, type) else str(kind).split('.')[-1]
    return _SPLIT_PREFIX.sub('', name)


def backbone_tag(backbone) -> str:
    """The BACKBONE part of a registered name, from the backbone config."""
    kind = _type_name(backbone.get('type'))
    if kind.startswith('ResNeXt'):
        return f"X{backbone.get('depth')}"
    if kind.startswith('ResNet'):
        return f"R{backbone.get('depth')}"
    if kind == 'SwinTransformer':
        size = _SWIN_SIZES.get((backbone.get('embed_dims', _SWIN_DEFAULTS[0]),
                                tuple(backbone.get('depths', _SWIN_DEFAULTS[1]))))
        if size:
            return f'Swin{size}'
    return sanitise(kind)


def _head_tag(model_cfg) -> str:
    head = model_cfg.get('decode_head')
    if isinstance(head, (list, tuple)):
        head = head[-1]
    if head is None:
        return ''
    head = head.get('kernel_generate_head') or head          # KNet: the head its kernels are generated from
    kind = _type_name(head.get('type'))
    return sanitise(kind[:-len('Head')] if kind.endswith('Head') else kind)


@functools.lru_cache(maxsize=None)
def _metafile(package: str) -> Tuple[dict, ...]:
    """Every model of ``package``'s model-zoo metafiles (its installed ``.mim/model-index.yml``)."""
    import yaml
    root = os.path.join(os.path.dirname(importlib.import_module(package).__file__), '.mim')
    with open(os.path.join(root, 'model-index.yml')) as f:
        index = yaml.safe_load(f)
    models = []
    for rel in index.get('Import', []):
        with open(os.path.join(root, rel)) as f:
            for m in (yaml.safe_load(f) or {}).get('Models', []) or []:
                if m.get('Config'):
                    models.append({**m, 'Config': os.path.join(root, m['Config'])})
    return tuple(models)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def metafile_entry(package: str, mim_name: Optional[str] = None, config_file: Optional[str] = None) -> Optional[dict]:
    """The metafile model of a mim name (its ``Name``, else its ``Config`` file name) or of a config file."""
    models = _metafile(package)
    for key, field in ((mim_name, 'Name'), (mim_name, 'Config'), (config_file and _stem(config_file), 'Config')):
        if key:
            hit = next((m for m in models if (m[field] if field == 'Name' else _stem(m[field])) == key), None)
            if hit is not None:
                return hit
    return None


def mim_config_path(package: str, mim_name: str) -> str:
    """The installed config file of a mim config name (looked up by metafile Name, then by Config file name)."""
    entry = metafile_entry(package, mim_name)
    if entry is None:
        raise KeyError(f'{mim_name!r} is not in the {package} model-zoo metafiles')
    return entry['Config']


@functools.lru_cache(maxsize=None)
def collection_head_types(package: str, collection: str) -> Tuple[str, ...]:
    """The decode-head types (minus 'Head') of every model of ``collection`` in ``package``'s metafile."""
    from mmengine import Config
    heads = set()
    for m in _metafile(package):
        if m.get('In Collection') == collection and os.path.isfile(m['Config']):
            heads.add(_head_tag(Config.fromfile(m['Config'], import_custom_modules=False).model))
    return tuple(sorted(heads))


def method_name(cfg, package: str, mim_name: Optional[str] = None) -> str:
    """The METHOD part of a registered name (see the module docstring)."""
    kind = _type_name(cfg.model.type)
    if kind not in _GENERIC_CONTAINERS.get(package, ()):
        return kind
    entry = metafile_entry(package, mim_name, getattr(cfg, 'filename', None))
    if entry is None or not entry.get('In Collection'):
        return kind + _head_tag(cfg.model)             # no metafile entry: the class, and its head
    collection = entry['In Collection']
    multi = len(collection_head_types(package, collection)) > 1
    return sanitise(collection) + (_head_tag(cfg.model) if multi else '')


def registered_name(cfg, package: str, *, mim_name: Optional[str] = None) -> str:
    """The automatic registered name of a config: ``<Pkg>Split<METHOD><BACKBONE>``."""
    backbone = cfg.model.get('backbone')
    return split_class_name(method_name(cfg, package, mim_name) + (backbone_tag(backbone) if backbone else ''),
                            package)


def registered_names(configs: Dict[str, tuple]) -> Dict[str, str]:
    """{label: registered_name} for {label: (cfg, package, mim_name)}. A name that comes out for more than one
    config is kept for each, with a warning naming the configs: pass the mim config name for those instead."""
    names = {label: registered_name(cfg, package, mim_name=mim_name)
             for label, (cfg, package, mim_name) in configs.items()}
    by_name = {}
    for label, name in names.items():
        by_name.setdefault(name, []).append(label)
    for name, labels in by_name.items():
        if len(labels) > 1:
            warnings.warn(f'Automatic class naming not unique: {name} for {", ".join(labels)}; pass the mim '
                          f'config name instead, e.g. build_split_model({labels[0]!r}, package=...).',
                          UserWarning, stacklevel=2)
    return names
