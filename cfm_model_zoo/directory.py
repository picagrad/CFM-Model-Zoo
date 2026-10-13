'''Model registry, file resolution and build_split_model().

The registry tables named below are defined in ``cfm_model_zoo.registry`` (data only) and re-exported here,
so every one stays importable from this module, which holds the logic and reads them through its own namespace.

- ``detectors`` / ``segmentors`` map the registered names to Split classes. ``DETECTION_MODELS`` /
  ``SEGMENTATION_MODELS`` hold each entry's mim config name, its backbone family and, where needed,
  a config shipped inside this package (``local_config``, under ``cfm_model_zoo/configs``).
- A registered name is the automatic name of its config, ``<Pkg>Split<Method><Backbone>``
  (``MMDetSplitFasterRCNNR50``, ``MMSegSplitUPerNetSwinT``; see ``cfm_model_zoo.naming``), listed in
  ``REGISTERED_NAMES``. The short names used before (``FasterRCNN``, ``SegUperNetSWIN``) stay mapped to the
  same entries in every dict above; ``LEGACY_NAMES`` maps each to its registered name.
- Weights come from two places:
  * OpenMMLab, through mim (``source='mim'``): downloaded once into the cache directory and checked
    against the server's size before use;
  * a checkpoint downloaded from its publisher's URL (``shared_checkpoint``), for the models mim cannot
    resolve: DDQ-DETR (OpenMMLab's download server) and RepPointsV2 (the release of Microsoft's
    Swin-Transformer-Object-Detection). Downloaded once into the cache directory and checked against
    its size and sha256; if that fails, SharedCheckpointUnavailable says how to supply the file yourself.
- ``source='local'`` never downloads, for a registered name and a raw mim config name alike. It
  uses the packaged config and the registry's checkpoint file name in the cache directory when an entry
  has both, else the cached mim files. ``checkpoint=`` replaces the checkpoint. (GLIP and GroundingDINO
  build their BERT language model through transformers, which fetches it from the Hugging Face Hub on
  first use under either source.)
- Any other mmdet/mmseg config name on mim builds too: pass ``package=``.
'''
from .split_models import *   # the Split classes, the optional-dependency flags, to_split_config (its __all__)
from .registry import (   # the registry tables (data only), re-exported: the logic reads them from this module
    detectors, segmentors, DETECTION_MODELS, SEGMENTATION_MODELS, backbone_style, LEGACY_NAMES, REGISTERED_NAMES,
    _DETECTION_NAMES, _SEGMENTATION_NAMES, _SEG_BACKBONE, _SEG_MIM_NAME, _CONFIG_ROOT, _config_path, _shared,
    _with_legacy_names)
from . import TORCH_LOAD_SAFE_GLOBALS
from typing import NamedTuple, Optional, Union
from mim.commands.search import get_model_info
from mim import download
from mim.utils import DEFAULT_CACHE_DIR, module_full_name, get_content_from_url
from mmengine import Config
from mmengine.hub import get_config
import hashlib
import importlib
import inspect
import logging
import torch
from mmengine.registry import MODELS as _ROOT_MODELS, DefaultScope, init_default_scope
from mmengine.runner import load_checkpoint
import requests
from requests.exceptions import RequestException, SSLError
import ssl
from urllib.parse import urlparse
import glob
import shutil
import os
import re
import contextlib
import difflib
import functools
import pickle
import threading
import time
import uuid
import warnings

logger = logging.getLogger(__name__)
SOURCES = ('mim', 'local')
CACHE_ENV_VAR = 'CFM_MODEL_ZOO_CACHE_DIR'
_LEGACY_MODELS = ('MMDetSplitRepPointsV2MaskDetectorSwinT', 'RepPointsV2')   # registered names that need legacy/


def _fallback_sequence(preferred: str):
    '''The sources a raw mim config name is tried with: 'mim' falls back to the files already on disk;
    'local' is local only (it never downloads).'''
    return ('mim', 'local') if preferred == 'mim' else ('local',)


##########################################################################################
# Source resolution. A checkpoint URL is only ever used to fetch the file or to derive its file
# name in the cache directory.
##########################################################################################

def _mim_weights_url(package: str, mim_name: str) -> str:
    '''Local metadata-only lookup (no network) — resolves a mim config name to its checkpoint
    URL via the package's own installed model index. Raises ValueError if package isn't
    installed locally or mim_name isn't in its index.'''
    if not mim_name:
        raise ValueError('no mim config name available for this model')
    info = get_model_info(module_full_name(package), configs=[mim_name],
                          shown_fields=['weight'], to_dict=True)
    return info[mim_name]['weight']


# We do NOT use mim's own checkpoint download. mim.download() -> _download_configs() ->
# download_from_file() has two defects that silently produce truncated checkpoint files, which
# then blow up much later inside torch.load ("PytorchStreamReader ... failed finding central
# directory" / "unexpected EOF ... file might be corrupted"):
#   1. download_from_file() reads Content-Length into `size` but never compares it against the
#      bytes actually written; a mid-stream socket close just ends the iter_content loop and the
#      function returns "successfully". Its only integrity check is an optional sha256 hash_prefix
#      that the openmmlab/GitHub model-index entries don't carry.
#   2. _download_configs() decides a checkpoint is cached by os.path.exists() alone (no size/hash
#      -- mim's own "# TODO: check checkpoint hash"), so a partial file from an earlier attempt in
#      the same job is accepted as complete on the next call.
# get_content_from_url() also hardcodes timeout=15 with no way to override it through the public
# call chain. So instead: we resolve the weight URL from the package's (local, network-free)
# model-index, fetch it ourselves with size verification via _fetch_checkpoint_with_timeout(),
# place it exactly where mim expects it, then call mim.download() only for its (local) config
# dump -- it sees the file present and skips the download.
_FALLBACK_DOWNLOAD_TIMEOUT = 120  # seconds; escalated read-timeout for our own fetch
_HEAD_TIMEOUT = 30                # seconds; for the Content-Length probe on a cached file
_MIN_CHECKPOINT_BYTES = 1024     # anything smaller is a 0-byte stub or an HTML error page
_STALE_PARTIAL_SECONDS = 3600    # a .partial file untouched this long belongs to a killed download


def _purge(path: str) -> None:
    '''Remove a checkpoint file and the leftover .partial temp files of KILLED downloads of it (the
    per-process-unique .partial.<pid>.<token> that _fetch_checkpoint_with_timeout writes, and the older
    bare .partial), so nothing lingers in a persistent, shared cache dir. A .partial file modified
    within the last _STALE_PARTIAL_SECONDS may be another process's download in flight and is left
    alone: removing it would make that process's os.replace fail.'''
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    now = time.time()
    for p in glob.glob(f'{glob.escape(path)}.partial*'):
        try:
            if now - os.path.getmtime(p) >= _STALE_PARTIAL_SECONDS:
                os.remove(p)
        except FileNotFoundError:
            pass


def resolve_download_path(explicit: Optional[str] = None) -> str:
    '''The checkpoint/config cache directory: ``explicit`` if given, else $CFM_MODEL_ZOO_CACHE_DIR,
    else mim's own default cache directory.'''
    if explicit:
        return os.path.expanduser(explicit)
    return os.path.expanduser(os.environ.get(CACHE_ENV_VAR) or DEFAULT_CACHE_DIR)


def _remote_checkpoint_size(url: str, check_certificate: bool) -> Optional[int]:
    '''Best-effort Content-Length via HEAD (following redirects -- GitHub release assets 302 to
    release-assets.githubusercontent.com). Returns None if the server doesn't report it or the
    probe fails; callers must treat None as "cannot verify", not "size 0".'''
    try:
        resp = requests.head(url, allow_redirects=True, timeout=_HEAD_TIMEOUT,
                             verify=check_certificate)
        cl = resp.headers.get('content-length')
        return int(cl) if cl is not None else None
    except SSLError as e:
        if _is_certificate_failure(e):
            print(f'⚠️  could not verify the TLS certificate of {urlparse(url).hostname or url} for the size check; '
                  + '; '.join(_CERTIFICATE_FIXES))
        return None
    except (RequestException, ValueError):
        return None


def _checkpoint_is_complete(path: str, url: str, check_certificate: bool) -> bool:
    '''True iff `path` is a plausible fully-downloaded checkpoint: it exists, is above a trivial
    size floor, and -- when the remote Content-Length is obtainable -- matches it exactly. If the
    remote size can't be probed, an existing non-trivial file is accepted (our own fetches are
    already verified against their GET response's Content-Length at download time; this weaker
    branch only matters for a pre-existing file we didn't write, e.g. a persistent cache hit).'''
    if not os.path.isfile(path):
        return False
    local_size = os.path.getsize(path)
    if local_size < _MIN_CHECKPOINT_BYTES:
        return False
    remote_size = _remote_checkpoint_size(url, check_certificate)
    if remote_size is None:
        print(f'⚠️  could not probe remote size for {os.path.basename(path)}; '
              f'accepting existing {local_size}-byte file without an exact-size check')
        return True
    return local_size == remote_size


class CertificateVerificationFailed(OSError):
    '''A download's TLS certificate could not be verified. Retrying cannot fix it, so it is raised at once.'''


_CERTIFICATE_FIXES = (
    'an outdated CA bundle is the usual cause: pip install -U certifi',
    'behind a proxy that re-signs TLS (common on clusters and corporate networks), point requests at its CA: '
    'export REQUESTS_CA_BUNDLE=/path/to/proxy-ca.pem',
    'to skip verification at your own risk, pass check_certificate=False')


def _is_certificate_failure(error: BaseException) -> bool:
    '''True only for a failed certificate verification: an ssl.SSLCertVerificationError anywhere in the chain
    (requests' SSLError -> urllib3's MaxRetryError.reason -> urllib3's SSLError -> the ssl error; mim's
    string-raise TypeError -> its __context__), or OpenSSL's text for it. requests raises SSLError for EVERY
    TLS failure; the others (a protocol mismatch, an EOF from a CDN or proxy, a record error mid-transfer)
    are ordinary network errors and are retried.'''
    seen, todo = set(), [error]
    while todo:
        e = todo.pop()
        if not isinstance(e, BaseException) or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, ssl.SSLCertVerificationError):
            return True
        text = str(e)
        if 'CERTIFICATE_VERIFY_FAILED' in text or 'certificate verify failed' in text.lower():
            return True
        todo += [e.__cause__, e.__context__, getattr(e, 'reason', None), *e.args]
    return False


def _certificate_error(url: str, cause: Exception) -> CertificateVerificationFailed:
    host = urlparse(url).hostname or url
    return CertificateVerificationFailed(f'could not verify the TLS certificate of {host} while fetching {url}: '
                                         f'{cause}' + ''.join(f'\n  - {fix}' for fix in _CERTIFICATE_FIXES))


def _fetch_checkpoint_with_timeout(url: str, dest_path: str, check_certificate: bool,
                                   timeout: int = _FALLBACK_DOWNLOAD_TIMEOUT,
                                   retries: int = 2) -> None:
    '''Fetch a checkpoint at an escalated read-timeout, verifying the full byte count landed.
    Writes to a per-process-unique .partial.<pid>.<token> temp path and only os.replace()s it
    onto dest_path once the transfer is confirmed complete -- a stall must never leave a
    truncated file at dest_path, since a later call (or mim) would treat its mere existence as a
    finished download. The unique temp name matters once download_path is a persistent dir
    shared across jobs (see resolve_download_path): two jobs racing to fetch the same
    not-yet-cached checkpoint must not write to the same temp file and corrupt each other's
    transfer -- each still downloads its own full copy (that redundancy was judged not worth
    engineering around), but neither can clobber the other's bytes, and _purge sweeps only
    .partial files untouched for _STALE_PARTIAL_SECONDS: a download stalled that long is treated as
    dead (if it was merely that slow, its final os.replace fails and the fetch is retried).'''
    tmp_path = f'{dest_path}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}'
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            try:
                response = get_content_from_url(url, timeout=timeout, stream=True,
                                                check_certificate=check_certificate)
            except TypeError as te:
                # mim's get_content_from_url raises a str (highlighted_error), which Python turns
                # into this TypeError; the real network error is its __context__.
                cause = te.__context__
                if cause is None:
                    raise
                if _is_certificate_failure(cause):
                    raise _certificate_error(url, cause) from cause
                raise IOError(f'could not fetch {url}: {type(cause).__name__}: {cause}') from cause
            expected_size = int(response.headers.get('content-length', 0) or 0)
            if not expected_size:
                # some responses (chunked, no header) omit it -- fall back to a HEAD probe so we
                # still have something to verify against rather than accepting any length.
                expected_size = _remote_checkpoint_size(url, check_certificate) or 0
            written = 0
            with open(tmp_path, 'wb') as fw:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fw.write(chunk)
                        written += len(chunk)
            if expected_size and written != expected_size:
                raise IOError(f'incomplete download: got {written} of {expected_size} bytes')
            if not expected_size:
                print(f'⚠️  {os.path.basename(dest_path)}: server reported no Content-Length; '
                      f'wrote {written} bytes, cannot verify completeness')
            if written < _MIN_CHECKPOINT_BYTES:
                raise IOError(f'suspiciously small download: {written} bytes')
            os.replace(tmp_path, dest_path)
            return
        except CertificateVerificationFailed:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        except (RequestException, IOError) as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if _is_certificate_failure(e):
                raise _certificate_error(url, e) from e
            last_err = e
            if isinstance(e, SSLError):         # any other TLS error is a network error like the rest
                last_err = IOError(f'could not fetch {url}: {type(e).__name__}: {e}')
                last_err.__cause__ = e
            if attempt < retries:
                print(f'⚠️  checkpoint fetch failed ({e}); retrying '
                      f'{attempt}/{retries - 1} at timeout={timeout}s...')
    raise last_err


def _resolve_from_mim(package: str, mim_name: str, download_path: str, check_certificate: bool):
    '''Resolve a mim config name to (cfg_path, checkpoint_path), downloading as needed.

    We fetch the checkpoint ourselves with byte-count verification (see the block comment above
    _fetch_checkpoint_with_timeout for why mim's own download can't be trusted), placing it where
    mim expects, then call mim.download() only to produce the flattened config .py -- that step
    is fully local, and mim skips the checkpoint download because the file is already present.'''
    if not mim_name:
        raise ValueError('no mim config name available for this model')

    os.makedirs(download_path, exist_ok=True)
    weights_url = _mim_weights_url(package, mim_name)  # local model-index lookup, no network
    for one_url in weights_url.split(','):
        filename = one_url.split('/')[-1]
        checkpoint_path = os.path.join(download_path, filename)
        if _checkpoint_is_complete(checkpoint_path, one_url, check_certificate):
            continue
        if os.path.exists(checkpoint_path):
            print(f'⚠️  {filename} is present but incomplete/corrupt -- re-fetching')
        _purge(checkpoint_path)
        print(f'Fetching {filename} (timeout {_FALLBACK_DOWNLOAD_TIMEOUT}s)...')
        _fetch_checkpoint_with_timeout(one_url, checkpoint_path, check_certificate)

    # Checkpoint(s) present and verified -> mim.download() only does its (network-free) config
    # dump, and still runs on EVERY call: it resolves the fully unrolled config (all _base_ files
    # merged) and the checkpoint name for us, which we don't want to re-derive for unfamiliar
    # models. But its dump is `Config.dump(<dest>/<name>.py)`, an unconditional, non-atomic
    # truncate-and-write. With dest = the shared cache, two jobs building the same model at the
    # same moment race: one reads the file while the other is rewriting it, and gets
    # "'ConfigDict' object has no attribute 'model'". So mim dumps into
    # a private temp dir instead; the shared config is left alone if it is already valid, and is
    # otherwise published with an atomic os.replace (same filesystem), so a reader only ever sees
    # no file or a complete one.
    cfg_path = os.path.join(download_path, f'{mim_name}.py')
    tmp_dir = os.path.join(download_path, f'.mimtmp.{os.getpid()}.{uuid.uuid4().hex}')
    os.makedirs(tmp_dir)
    try:
        for one_url in weights_url.split(','):   # make mim see the verified checkpoint(s) as present
            filename = one_url.split('/')[-1]
            os.symlink(os.path.join(download_path, filename), os.path.join(tmp_dir, filename))
        checkpoint_name = download(package, [mim_name], dest_root=tmp_dir,
                                   check_certificate=check_certificate)[0]
        tmp_cfg = os.path.join(tmp_dir, f'{mim_name}.py')
        # mim only prints (does not raise) when it cannot find the config, so tmp_cfg may be
        # missing -- then behave as before and let the caller fail on the absent config.
        if os.path.exists(tmp_cfg) and not _config_is_valid(cfg_path):
            os.replace(tmp_cfg, cfg_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    checkpoint_path = os.path.join(download_path, checkpoint_name)
    return cfg_path, checkpoint_path


def _config_is_valid(path: str) -> bool:
    '''True iff `path` is a complete, loadable mmengine config with a `model` section -- the bar a
    cached config must clear to be reused untouched. An empty or half-written file fails it.'''
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        return 'model' in Config.fromfile(path)
    except Exception:  # noqa: BLE001 -- any parse failure means "not usable"
        return False


_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _short(error: Exception, head_lines: int = 10, tail_chars: int = 1000, max_chars: int = 2500) -> str:
    '''An error's text without terminal colour codes, capped: the first lines and the tail (mim lists
    its whole index on a typo and ends with the actionable part).'''
    text = _ANSI.sub('', str(error))
    if len(text) <= max_chars and len(text.splitlines()) <= 2 * head_lines:
        return text
    head = '\n'.join(text.splitlines()[:head_lines])[:max_chars - tail_chars]
    tail = text[len(head):][-tail_chars:]                   # only what the head did not already show
    cut = len(text) - len(head) - len(tail)
    if cut <= 0:
        return text
    return f'{head}\n[... {cut} chars truncated ...]\n{tail.lstrip(chr(10))}'


def _resolve_from_local(package: str, mim_name: str, download_path: str, local_config: Optional[str] = None,
                        local_checkpoint: Optional[str] = None, checkpoint: Optional[str] = None):
    '''Files already on disk; never the network. The packaged config with the registry's checkpoint
    file name in the cache when the entry has both, else the cached mim config and checkpoint.
    ``checkpoint`` (an explicit path) replaces the checkpoint either way.'''
    if local_config and (local_checkpoint or checkpoint):
        cfg_path = local_config
        checkpoint_path = checkpoint or os.path.join(download_path, local_checkpoint)
    else:
        weights_url = _mim_weights_url(package, mim_name)
        cfg_path = os.path.join(download_path, f'{mim_name}.py')
        checkpoint_path = checkpoint or os.path.join(download_path, weights_url.split('/')[-1])
    missing = [p for p in (cfg_path, checkpoint_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'local files not found: {", ".join(missing)}')
    return cfg_path, checkpoint_path


class CacheUnavailable(OSError):
    '''The checkpoint cache directory cannot be created or written.'''


def _require_writable_cache(download_path: str) -> None:
    '''Create the cache directory if needed; raise an error naming the knobs if it cannot be written.'''
    try:
        os.makedirs(download_path, exist_ok=True)
        writable = os.access(download_path, os.W_OK | os.X_OK)
        problem = None if writable else 'it is not writable'
    except OSError as e:
        problem = str(e)
    if problem:
        raise CacheUnavailable(f"cannot use {download_path!r} as the checkpoint cache ({problem}); set {CACHE_ENV_VAR} "
                      f'or pass download_path= to use another directory')


def _config_for_checkpoint(package: str, mim_name: str, download_path: str) -> str:
    '''The mim config of ``mim_name`` as a flattened file ``<cache>/<mim_name>.py``, for use with an
    explicit checkpoint. A valid cached copy is used as is (nothing is written). Otherwise it is
    dumped through mmengine.hub.get_config, which reads the installed package's own files (no
    network while the package is installed) and never sees a checkpoint.'''
    cfg_path = os.path.join(download_path, f'{mim_name}.py')
    if _config_is_valid(cfg_path):
        return cfg_path
    if not mim_name:
        raise ValueError('no mim config name available for this model')
    info = get_model_info(module_full_name(package), configs=[mim_name], shown_fields=['config'], to_dict=True)
    rel = info[mim_name]['config']        # 'configs/<dir>/<name>.py'; get_config wants it relative to configs/
    cfg = get_config(f"{package}::{rel[len('configs/'):] if rel.startswith('configs/') else rel}")
    _require_writable_cache(download_path)
    tmp_cfg = os.path.join(download_path, f'.{mim_name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.py')
    try:
        cfg.dump(tmp_cfg)
        os.replace(tmp_cfg, cfg_path)
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
    return cfg_path


def _resolve_source(source: str, package: str, mim_name: str, download_path: str, check_certificate: bool,
                    local_config=None, local_checkpoint=None, checkpoint=None):
    if checkpoint and (source == 'mim' or not local_config):
        # An explicit checkpoint is never downloaded, linked or copied; only the config is needed.
        return _config_for_checkpoint(package, mim_name, download_path), checkpoint
    if source == 'mim':
        return _resolve_from_mim(package, mim_name, download_path, check_certificate)
    return _resolve_from_local(package, mim_name, download_path, local_config, local_checkpoint, checkpoint)


def _source_available(source: str, package: str, mim_name: str, download_path: str,
                      local_config=None, local_checkpoint=None) -> bool:
    '''Cheap best-effort probe (no download), only for suggestion messages.'''
    try:
        if source == 'mim':
            _mim_weights_url(package, mim_name)
            return True
        _resolve_from_local(package, mim_name, download_path, local_config, local_checkpoint)
        return True
    except Exception as e:   # broad on purpose: a failed probe only means no suggestion; the real error is raised
        logger.debug('source probe %s for %s failed: %r', source, mim_name, e)
        return False


def _resolve_curated(model_type: str, package: str, entry: dict, source: str, download_path: str,
                     check_certificate: bool, checkpoint: Optional[str] = None):
    '''Registered names: single source, no silent fallback (a packaged curated config is not
    guaranteed to match mim's). On failure, suggests the other source if it looks available.'''
    mim_name = entry.get('mim_name')
    try:
        return _resolve_source(source, package, mim_name, download_path, check_certificate,
                               entry.get('local_config'), entry.get('local_checkpoint'), checkpoint)
    except (CacheUnavailable, CertificateVerificationFailed):
        raise
    except Exception as e:
        if checkpoint and (source == 'mim' or not entry.get('local_config')):
            # _config_for_checkpoint served this call under either source: no other source takes another path.
            raise RuntimeError(
                f"Could not build the config for {model_type} to use with checkpoint={checkpoint!r} ({_short(e)}). "
                f"With an explicit checkpoint the config comes from the installed {package} package's own "
                f"config files, whichever source is chosen, so another source would fail the same way. "
                f"Check that {package} is installed correctly (pip install -U {package}), that the cache "
                f"directory {download_path!r} is writable.") from e
        alternatives = [alt for alt in SOURCES if alt != source and _source_available(
            alt, package, mim_name, download_path, entry.get('local_config'), entry.get('local_checkpoint'))]
        msg = f"source={source!r} failed for {model_type} ({_short(e)})."
        if alternatives:
            msg += (f" {model_type} looks available via: {', '.join(alternatives)}."
                    f" Consider source={alternatives[0]!r}.")
        else:
            msg += f" No other source looks available for {model_type} either."
        raise RuntimeError(msg) from e


def _resolve_generic(package: str, mim_name: str, source: str, download_path: str, check_certificate: bool,
                     checkpoint: Optional[str] = None):
    '''Any other mim config name: one canonical checkpoint per name, so fall back across sources.
    With an explicit checkpoint the config comes from mim's config files and nothing is downloaded.'''
    errors = {}
    for src in (('local',) if checkpoint else _fallback_sequence(source)):
        try:
            return _resolve_source(src, package, mim_name, download_path, check_certificate, checkpoint=checkpoint)
        except (CacheUnavailable, CertificateVerificationFailed):
            raise
        except Exception as e:
            errors[src] = e
    texts = {k: _short(v) for k, v in errors.items()}
    detail = (next(iter(texts.values())) if len(set(texts.values())) == 1
              else '; '.join(f'{k}: {v}' for k, v in texts.items()))
    hint = ('' if checkpoint or source == 'mim' else
            f" source='local' never downloads: pass source='mim' to download {mim_name} into {download_path!r}.")
    raise RuntimeError(f"Could not resolve {mim_name} ({package}) via any source ({', '.join(errors)}): {detail}.{hint}")


class SharedCheckpointUnavailable(RuntimeError):
    '''A checkpoint that mim cannot resolve (shared_checkpoint) could not be found or downloaded.'''


def _shared_file_problem(path: str, spec: dict) -> Optional[str]:
    '''None if ``path`` is a usable copy of the shared checkpoint, else why it is not.'''
    if not os.path.isfile(path):
        return 'not found'
    size = os.path.getsize(path)
    if size < _MIN_CHECKPOINT_BYTES:
        return f'only {size} bytes'
    if spec.get('size') is not None and size != spec['size']:
        return f'size {size} != expected {spec["size"]}'
    if spec.get('sha256'):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != spec['sha256']:
            return 'sha256 mismatch'
    return None


def _resolve_shared(model_type: str, entry: dict, download_path: str, check_certificate: bool,
                    checkpoint: Optional[str] = None, allow_download: bool = True):
    '''A shared entry: the packaged config, and the checkpoint from ``checkpoint=``, the cache, or
    (only for source='mim', and only if the entry has a URL) a verified download into the cache.'''
    cfg_path = entry['local_config']
    if checkpoint is not None:
        return cfg_path, checkpoint
    spec = entry['shared_checkpoint']
    cache_path = os.path.join(download_path, spec['filename'])
    problem = _shared_file_problem(cache_path, spec)
    if problem is None:
        return cfg_path, cache_path
    tried, cause = [f'cache: {cache_path} ({problem})'], None
    if not spec.get('url'):
        tried.append('download: no public download location for this checkpoint')
    elif not allow_download:
        tried.append("download: skipped (source='local' never downloads; pass source='mim' to download it)")
    else:
        try:
            os.makedirs(download_path, exist_ok=True)
            _purge(cache_path)          # also sweeps .partial.* files left by a killed download
            _fetch_checkpoint_with_timeout(spec['url'], cache_path, check_certificate)
            problem = _shared_file_problem(cache_path, spec)
            if problem is None:
                return cfg_path, cache_path
            _purge(cache_path)
            tried.append(f'download: {spec["url"]} (downloaded, but {problem})')
        except CertificateVerificationFailed:
            raise                       # its own type: it names the host and the fixes
        except Exception as e:  # noqa: BLE001 -- reported in full below and chained
            cause = e
            tried.append(f'download: {spec["url"]} ({type(e).__name__}: {e})')
    raise SharedCheckpointUnavailable(
        f'{model_type}: no checkpoint available.\n'
        f'This checkpoint cannot be resolved through OpenMMLab (mim); it is downloaded from its publisher.\n'
        f'Sources tried:\n' + ''.join(f'  - {t}\n' for t in tried) +
        f'Expected cache path: {cache_path}\n'
        f'To use a file you downloaded yourself, copy it to the expected cache path, or pass it explicitly:\n'
        f"  build_split_model({model_type!r}, cut_point=1, source='local', "
        f"checkpoint='/path/to/{spec['filename']}')") from cause


class ModelFiles(NamedTuple):
    package: str      # 'mmdet' or 'mmseg'
    config: str       # config file path
    checkpoint: str   # checkpoint file path


def resolve_model_files(model_type: str, *, source: str = 'local', download_path: Optional[str] = None,
                        package: Optional[str] = None, check_certificate: bool = True,
                        checkpoint: Optional[str] = None) -> ModelFiles:
    '''The one resolver: the package, config file and checkpoint file of ``model_type``.

    model_type: a registered name (``detectors`` / ``segmentors``; an old short name works too) or any
        mim config name (then ``package`` is required).
    source: 'mim' (download into the cache if needed) or 'local' (files already on disk only).
    checkpoint: an explicit checkpoint file, used instead of any download. A model with no packaged
        config gets its config from mim's (local) config dump, so this needs mim's index but no network.
    '''
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    if checkpoint is not None:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'checkpoint= {checkpoint} does not exist')
    if model_type in _LEGACY_MODELS and MMDET_AVAILABLE and not LEGACY_AVAILABLE:
        raise ImportError(f'{model_type} needs cfm_model_zoo.legacy, which failed to import: '
                          f'{LEGACY_IMPORT_ERROR!r}') from LEGACY_IMPORT_ERROR
    download_path = resolve_download_path(download_path)
    for registered, names in (('mmdet', _DETECTION_NAMES), ('mmseg', _SEGMENTATION_NAMES)):
        if model_type in names and package not in (None, registered):
            raise ValueError(f'{model_type} is a registered {registered} model, got package={package!r}')
    for registered, names, available, error, extra in (
            ('mmdet', _DETECTION_NAMES, MMDET_AVAILABLE, MMDET_IMPORT_ERROR, 'detection'),
            ('mmseg', _SEGMENTATION_NAMES, MMSEG_AVAILABLE, MMSEG_IMPORT_ERROR, 'segmentation')):
        if (model_type in names or package == registered) and not available:
            cause = f' (importing it failed: {type(error).__name__}: {error})' if error is not None else ''
            raise ImportError(f"{model_type} needs {registered}{cause}. If it is not installed: "
                              f"pip install 'cfm-model-zoo[{extra}]'") from error
    if source == 'mim':
        _require_writable_cache(download_path)
    for registered, models in (('mmdet', DETECTION_MODELS), ('mmseg', SEGMENTATION_MODELS)):
        if model_type in models:
            entry = models[model_type]
            if 'shared_checkpoint' in entry:
                cfg_path, ckpt = _resolve_shared(model_type, entry, download_path, check_certificate,
                                                 checkpoint, allow_download=(source == 'mim'))
            else:
                cfg_path, ckpt = _resolve_curated(model_type, registered, entry, source, download_path,
                                                  check_certificate, checkpoint)
            return ModelFiles(registered, cfg_path, ckpt)
    if package not in ('mmdet', 'mmseg'):
        by_lower = {n.lower(): n for n in _DETECTION_NAMES + _SEGMENTATION_NAMES}
        close = [by_lower[m] for m in difflib.get_close_matches(model_type.lower(), by_lower, n=3, cutoff=0.5)]
        raise ValueError(f"{model_type} is not a registered name"
                         + (f' (did you mean: {", ".join(close)}?)' if close else '')
                         + ". For a raw mim config name, pass package='mmdet' or 'mmseg'.")
    cfg_path, ckpt = _resolve_generic(package, model_type, source, download_path, check_certificate, checkpoint)
    return ModelFiles(package, cfg_path, ckpt)


def _build_once(cfg: Config, checkpoint_path: Optional[str], cut_point: int,
                device: Union[str, 'torch.device'], package: str, download_path: Optional[str] = None, *,
                trust_checkpoint: bool = False):
    '''Build a split model ONCE from a Config and a checkpoint -- one path for every mm library
    (follows mmengine.hub.get_model). No init_detector / init_model.

    checkpoint_path=None leaves the constructor's initialisation (tests only). download_path: the
    cache directory a checkpoint conversion (prepare_checkpoint) writes into, if one is needed.
    trust_checkpoint: read the checkpoint with torch.load(weights_only=False) instead of under
    TORCH_LOAD_SAFE_GLOBALS (see build_split_model).'''
    if trust_checkpoint and checkpoint_path is None:
        raise ValueError('trust_checkpoint=True needs a checkpoint to trust: pass checkpoint= (or a model whose '
                         'checkpoint resolves)')
    cfg = to_split_config(cfg, cut_point, package)
    with DefaultScope.overwrite_default_scope(package):
        importlib.import_module(f'{package}.utils').register_all_modules(init_default_scope=False)
        model = _ROOT_MODELS.build(cfg.model)
    # init_detector/init_model leave the package's scope as the process-wide default, and code that
    # builds dataloaders and evaluators from unscoped names ('CocoDataset', 'CocoMetric') relies on
    # it. Keep that side effect (set once the build has succeeded).
    # `or`: a config may set default_scope=None explicitly, which init_default_scope rejects.
    init_default_scope(cfg.get('default_scope') or package)
    if checkpoint_path is not None:
        if trust_checkpoint:
            warnings.warn(f'trust_checkpoint=True: loading {checkpoint_path} with torch.load(weights_only=False). '
                          'This runs any code the file contains; use it only for checkpoints you fully trust.',
                          UserWarning, stacklevel=3)
        # Both read the file (a conversion such as RepPointsV2's loads it), so both are in the scope.
        with _trusted_torch_load() if trust_checkpoint else _checkpoint_safe_globals():
            reading = checkpoint_path
            try:
                path = reading = _prepare_checkpoint(type(model), checkpoint_path, download_path)
                # non-strict: mm prints its missing/unexpected-key warning, as before
                load_checkpoint(model, path, map_location='cpu')
            except pickle.UnpicklingError as error:
                if trust_checkpoint or 'Weights only load failed' not in str(error):
                    raise
                # inside the scope, so torch counts the allowlist (and the caller's own entries) as allowed
                raise _globals_not_allowed(reading, error) from error
        model._is_init = True      # mmengine's own guard against a later init_weights()
    model.cfg = cfg
    return model.to(device).eval()


@contextlib.contextmanager
def _checkpoint_safe_globals():
    '''Allow TORCH_LOAD_SAFE_GLOBALS (see cfm_model_zoo/__init__.py) in torch.load(weights_only=True) inside
    the block only. torch keeps one process-wide list, so the entries are visible to every thread while the
    block runs. Entries the caller had registered already are left out: torch's context manager removes its
    entries on exit, and must not take back the caller's own.'''
    present = set(torch.serialization.get_safe_globals())
    with torch.serialization.safe_globals([g for g in TORCH_LOAD_SAFE_GLOBALS if g not in present]):
        yield


# One trusted load at a time: each replaces torch.load and puts back what it found, so two at once would
# put back each other's replacement. Reentrant, for a trusted build started from inside one.
_TRUSTED_LOAD_LOCK = threading.RLock()


@contextlib.contextmanager
def _trusted_torch_load():
    '''torch.load with weights_only=False for the calls THIS thread makes inside the block. mmengine's
    CheckpointLoader (load_checkpoint, and RepPointsV2's conversion) calls torch.load without weights_only and
    has no way to pass it, so torch.load is replaced while the block runs and put back on exit, error or not.
    A torch.load call from any other thread meanwhile goes to the original, unchanged.'''
    with _TRUSTED_LOAD_LOCK:
        original = torch.load
        owner = threading.get_ident()

        @functools.wraps(original)
        def load(*args, **kwargs):
            if threading.get_ident() == owner:
                kwargs['weights_only'] = False
            return original(*args, **kwargs)

        torch.load = load
        try:
            yield
        finally:
            torch.load = original


class CheckpointGlobalsNotAllowed(pickle.UnpicklingError):
    '''The checkpoint needs globals that torch.load(weights_only=True) does not allow, even with
    TORCH_LOAD_SAFE_GLOBALS. ``path``: the file; ``unsafe_globals``: the names torch found in it that are not
    allowed (sorted), or None when torch cannot list them (a file in torch's legacy format).
    trust_checkpoint=True loads it with torch.load(weights_only=False).'''

    def __init__(self, message: str, path: Optional[str] = None, unsafe_globals: Optional[list] = None):
        super().__init__(message)
        self.path = path
        self.unsafe_globals = unsafe_globals


def _globals_not_allowed(path: str, error: pickle.UnpicklingError) -> CheckpointGlobalsNotAllowed:
    try:
        unsafe = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    except Exception as e:     # torch lists them only for its zip format
        unsafe, why = None, f'torch cannot list them for this file ({e}; it reads only its zip format)'
    reason = next((line.strip() for line in str(error).splitlines()
                   if 'WeightsUnpickler error' in line), '').split(' Please use')[0].rstrip('.')
    if unsafe:
        needs = 'it needs these globals, which are not on that list: ' + ', '.join(unsafe)
    else:
        needs = ('it needs a global that is not on that list; '
                 + (why if unsafe is None else 'torch found none by reading the file statically')
                 + (f'. torch: {reason}' if reason else ''))
    return CheckpointGlobalsNotAllowed(
        f'cannot load checkpoint {path} under the library\'s allowlist (torch.load(weights_only=True) with '
        f'cfm_model_zoo.TORCH_LOAD_SAFE_GLOBALS): {needs}. If you trust this file, pass trust_checkpoint=True: '
        'that loads it with torch.load(weights_only=False), which runs any code the file contains, so use it '
        'only for a file you trust.', path, unsafe)


def _prepare_checkpoint(model_cls, checkpoint_path: str, cache_dir: Optional[str]) -> str:
    '''Run the class's prepare_checkpoint hook, passing cache_dir only when the hook accepts it, so an
    override written with the one-argument signature keeps working.'''
    prepare = model_cls.prepare_checkpoint
    params = inspect.signature(prepare).parameters
    if 'cache_dir' in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return prepare(checkpoint_path, cache_dir=cache_dir)
    return prepare(checkpoint_path)


def build_split_model(model_type: str, cut_point: int = 0, *, use_mim: bool = False, source: Optional[str] = None,
                      download_path: Optional[str] = None, device: str = 'cpu', check_certificate: bool = True,
                      package: Optional[str] = None, checkpoint: Optional[str] = None,
                      trust_checkpoint: bool = False) -> torch.nn.Module:
    '''Build ``model_type`` split at ``cut_point``, in eval mode on ``device``.

    cut_point: 0 splits after preprocessing; k > 0 after the k-th backbone stage.
    source: 'mim' or 'local' (see resolve_model_files); the legacy ``use_mim=True`` means 'mim'.
        Default 'local'.
    checkpoint: an explicit checkpoint file, used instead of any download under either source (see
        resolve_model_files).
    trust_checkpoint: False (default) reads the checkpoint with torch.load(weights_only=True), allowing
        TORCH_LOAD_SAFE_GLOBALS on top of torch's own list; a file that needs anything else raises
        CheckpointGlobalsNotAllowed, naming what it needs. True reads it with torch.load(weights_only=False),
        with a UserWarning: that runs any code the file contains, so only for a file you trust.
    Everything after ``cut_point`` is keyword-only.
    '''
    print(f'Building model of type {model_type} and splitting at cut_point {cut_point}')
    if source is not None and use_mim:
        raise ValueError("Don't mix source= with the legacy use_mim=")
    files = resolve_model_files(model_type, source=source or ('mim' if use_mim else 'local'),
                                download_path=download_path, package=package,
                                check_certificate=check_certificate, checkpoint=checkpoint)
    return _build_once(Config.fromfile(files.config), files.checkpoint, cut_point, device, files.package,
                       resolve_download_path(download_path), trust_checkpoint=trust_checkpoint)
