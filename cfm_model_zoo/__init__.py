import sys
import warnings

# =============================================================================
# PATCH: apply_chunking_to_forward moved from modeling_utils → pytorch_utils
#
# transformers < 5.0: transformers.modeling_utils.apply_chunking_to_forward
# transformers >= 5.0: transformers.pytorch_utils.apply_chunking_to_forward
#
# mmdet's vlfuse_helper.py does:
#   from transformers.modeling_utils import apply_chunking_to_forward
# at import time (inside a try/except ImportError). When this fails silently,
# BertPreTrainedModel falls back to `object`, breaking BertEncoderLayer.__init__.
#
# This patch must run before any mmdet submodule is imported. Since all mmdet
# imports in this package go through split_utils or split_models (which are
# submodules of cfm_model_zoo), this __init__.py always runs first.
# =============================================================================
try:
    import transformers.modeling_utils as _tm
    if not hasattr(_tm, 'apply_chunking_to_forward'):
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, 'apply_chunking_to_forward'):
            import transformers as _tr
            raise RuntimeError(
                f"transformers.apply_chunking_to_forward not found in modeling_utils "
                f"or pytorch_utils (transformers=={_tr.__version__}). "
                f"This compat shim may need updating."
            )
        _tm.apply_chunking_to_forward = _pu.apply_chunking_to_forward
        del _pu
        if 'mmdet' in sys.modules:
            warnings.warn(
                "cfm_model_zoo: mmdet was imported before cfm_model_zoo.__init__ ran. "
                "The transformers 5.x compat patch for apply_chunking_to_forward may not "
                "have taken effect. If you see errors related to BertPreTrainedModel or "
                "BertEncoderLayer, ensure cfm_model_zoo is imported before mmdet.",
                stacklevel=2,
            )
    del _tm
except ImportError:
    pass  # transformers not installed; mmdet will handle this gracefully

# =============================================================================
# torch.load safe globals for OpenMMLab training checkpoints (PyTorch 2.6+)
#
# PyTorch 2.6 changed the default of `weights_only` in torch.load from False to True. mmengine's
# CheckpointLoader calls torch.load without specifying weights_only, so it is True, and some OpenMMLab
# checkpoints hold mmengine's HistoryBuffer and NumPy objects (the training state saved with the weights),
# which weights_only rejects.
#
# TORCH_LOAD_SAFE_GLOBALS lists what the registered checkpoints need. The library allowlists it ONLY while
# it loads a checkpoint itself (directory._build_once, with torch.serialization.safe_globals); importing
# cfm_model_zoo leaves torch's process-wide list as it was. Measured on all 47 registered checkpoints
# (torch 2.6.0, NumPy 2.4.2, mmengine 0.10.7): 39 need nothing; DeformableDETR, MaskFormer and Mask2Former
# (ADE20K) need every entry but getattr; DINO (R-50), GLIP and GroundingDINO need every entry (getattr
# rebuilds the HistoryBuffer's statistics methods).
#
# NumPy: these checkpoints were saved with NumPy 1.x, so their pickles name
# "numpy.core.multiarray._reconstruct" and "...scalar". In NumPy 2.x those functions live in numpy._core,
# and the unpickler matches by that string, so they are registered under the old name. Under NumPy 2.x,
# rebuilding a dtype yields an instance of a numpy.dtypes class, which must be allowed too.
# =============================================================================
TORCH_LOAD_SAFE_GLOBALS = ()
try:
    import numpy as _np
    from mmengine.logging.history_buffer import HistoryBuffer as _HB

    _multiarray = _np._core.multiarray if hasattr(_np, '_core') else _np.core.multiarray
    TORCH_LOAD_SAFE_GLOBALS = (
        _HB,
        _np.ndarray,
        _np.dtype,
        getattr,
        (_multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
        (_multiarray.scalar, 'numpy.core.multiarray.scalar'),
    ) + tuple(c for c in (getattr(getattr(_np, 'dtypes', None), n, None) for n in ('Float64DType', 'Int64DType'))
              if c is not None)
    del _np, _HB, _multiarray
except ImportError:
    pass  # mmengine or numpy not installed
