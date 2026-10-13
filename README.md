# CFM-Model-Zoo

A library of split vision models for compression for machines (CFM), also known as split computing, feature coding or
task-aware compression. Each model is split at a backbone **cut point** into a **frontend** (preprocessing and the
backbone up to the cut) and a **backend** (the rest of the backbone, the neck and the task head), so the features at
the cut can be compressed, transmitted, and turned into the task's output on the other side. The cut point is
flexible: it can be changed at run time, without rebuilding the model.

CFM-Model-Zoo is a growing corpus of split models across a range of computer vision tasks, built on the OpenMMLab
family of toolboxes and possibly extended beyond it. This release covers MMDetection (object detection, instance and
panoptic segmentation) and MMSegmentation (semantic segmentation); more tasks and toolboxes will follow. Every split
model is checked against the unsplit OpenMMLab model it comes from: [docs/verified_configs.md](docs/verified_configs.md)
lists all 1,504 zoo configs and what was verified for each.

Authors: Alon Harell. Code: <https://github.com/picagrad/CFM-Model-Zoo>.

## Install

**Before you start (not installed by uv):**

- an NVIDIA driver and a CUDA toolkit (`nvcc`) whose major version matches the CUDA build of PyTorch you install,
  to build MMCV's CUDA ops;
- a C++ compiler, `git` and `curl`;
- OpenCV's requirements;
- [uv](https://docs.astral.sh/uv/).

On Ubuntu: `apt install build-essential git curl libgl1 libglib2.0-0`, plus the CUDA toolkit from NVIDIA.

**Install.** The following installs the exact versions used in our verification. Other versions are likely to work
and will be verified in the near future.

- Python 3.12, PyTorch 2.6.0 (its CUDA 12.4 build), MMCV 2.1.0, MMEngine 0.10.7
- MMDetection 3.3.0, MMSegmentation 1.2.2, transformers 5.1.0, and the rest of [`constraints.txt`](constraints.txt)

MMCV is built from source with its CUDA ops (about 12 minutes), so it comes before this package:

```bash
git clone https://github.com/picagrad/CFM-Model-Zoo.git CFM-Model-Zoo && cd CFM-Model-Zoo
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install setuptools==69.5.1 wheel    # builds MMCV under Python 3.12 (needs pkg_resources and bdist_wheel)
CUDA_HOME=/usr/local/cuda bash scripts/build_mm_libs.sh   # MMCV 2.1.0 (CUDA ops), MMDetection 3.3.0, MMSegmentation 1.2.2
uv pip install --no-build-isolation -e ".[detection,segmentation]" -c constraints.txt
```

`scripts/build_mm_libs.sh` clones MMCV into `/tmp`, builds it for `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"` (edit it
for other GPUs), and installs MMDetection and MMSegmentation with `mim`.

For another CUDA build of PyTorch, change the `--index-url` (PyTorch 2.6.0 also ships `cu118` and `cu126`) and use a
matching toolkit. Leaving out `-c constraints.txt` installs the newest versions the extras allow.

These install steps, and the quickstart examples below, were tested end to end in a clean Ubuntu 22.04 + CUDA 12.4
container, on CPU and on an NVIDIA RTX 3080 Ti.

## Quickstart

### Detection

<!-- check: run -->
```python
from mmengine.dataset import Compose, pseudo_collate
from mmdet.utils import get_test_pipeline_cfg
from cfm_model_zoo.directory import build_split_model

# Downloads the OpenMMLab config and weights into the mim cache directory on first use.
model = build_split_model('MMDetSplitFasterRCNNR50', cut_point=1, source='mim')

pipeline = Compose(get_test_pipeline_cfg(model.cfg))
data = pseudo_collate([pipeline(dict(img_path='demo.jpg', img_id=0))])   # demo.jpg: any image you supply

front = model.feature_frontend(data)        # preprocessing + the backbone up to the cut
print([tuple(t.shape) for t in front['inputs'].outs])   # the features a codec would compress

preds = model.backend(front, mode='inference')      # resumes at the cut; returns DetDataSamples
print(preds[0].pred_instances.bboxes.shape)
```

### Segmentation

<!-- check: run -->
```python
from mmengine.dataset import Compose, pseudo_collate
from cfm_model_zoo.directory import build_split_model

model = build_split_model('MMSegSplitUPerNetR50', cut_point=1, source='mim')

pipeline = Compose([t for t in model.cfg.test_pipeline if t['type'] != 'LoadAnnotations'])
data = pseudo_collate([pipeline(dict(img_path='demo.jpg'))])

preds = model.backend(model.feature_frontend(data), mode='inference')
print(preds[0].pred_sem_seg.data.shape)
```

### Plain MMDetection / MMSegmentation

The Split classes are registered with MMEngine, so a rewritten config works with the usual mm APIs:

<!-- check: run -->
```python
from mmengine import Config
from mmdet.apis import init_detector
from cfm_model_zoo.directory import resolve_model_files
from cfm_model_zoo.split_models import to_split_config

files = resolve_model_files('MMDetSplitFasterRCNNR50', source='mim')   # or any mmdet config and checkpoint
cfg = to_split_config(Config.fromfile(files.config), cut_point=2, package='mmdet')
model = init_detector(cfg, files.checkpoint, device='cpu')
print(type(model).__name__, model.cut_point)               # MMDetSplitFasterRCNN 2
```

Use `mmseg.apis.init_model` the same way, with `package='mmseg'`. A saved `model.cfg` rebuilds on its own: its
`custom_imports` loads the Split classes.

### The split-model API

- `feature_frontend(data)`: preprocessing and the backbone up to the cut; returns the features at the cut.
- `backend(front, *, mode)`: everything after the cut. `mode` is required: `'inference'` (predictions), `'loss'` (the
  task loss; `individual_losses=True` gives one loss per image) or `'raw'` (the head's raw outputs). Any other mode
  raises `ValueError`.
- `feature_backbone(front)`: the full set of backbone features from a frontend output.
- `model.cut_point`: where the split is; set it to move the split at run time (0 to the number of backbone stages).
- `'raw'` is not available for GroundingDINO, GLIP and the MaskFormer / Mask2Former segmentors, because OpenMMLab's own
  `forward(mode='tensor')` is broken for them; `'loss'` and `'inference'` work.

## Models

Every registered model is named `<Package>Split<Method><Backbone>`, for example `MMDetSplitFasterRCNNR50` or
`MMSegSplitUPerNetSwinT`:

- **Package:** currently `MMDet` or `MMSeg`.
- **Method:** the detector's class name in MMDetection. MMSegmentation's segmentors all share the class
  `EncoderDecoder`, so for them it is the method's name in MMSegmentation's model zoo, followed by the decode head
  when that method comes with several (`MMSegSplitKNetUPerR50`).
- **Backbone:** `R50`, `R101`, `X101`, `SwinT`, `SwinS`, `SwinB` or `SwinL`.

The full list, with the papers to cite for each model, is in
[`cfm_model_zoo/configs/README.md`](cfm_model_zoo/configs/README.md); in code, `cfm_model_zoo.directory.detectors` and
`segmentors` hold the same names. The split classes themselves are named `<Package>Split<Class>`
(`MMDetSplitFasterRCNN`, `MMSegSplitEncoderDecoder`) and the split backbones `<Package>Split<Backbone>`
(`MMDetSplitResNet`).

- **Any other config:** every MMDetection or MMSegmentation config on mim also builds, with
  `build_split_model('<config name>', package='mmdet', source='mim')`. A model type we have not tested prints a
  warning.
- **Cut points:** `cut_point=0` splits after preprocessing; `cut_point=k` splits after the k-th backbone stage.
- **Backbones:** MMDetection's `ResNet` and `SwinTransformer`, MMSegmentation's `ResNet`, `ResNetV1c` and
  `SwinTransformer`, and subclasses that inherit one of those forward passes unchanged (for example `ResNetV1d`,
  `ResNeXt`). For the classes named here, the split output was compared with the backbone's own forward at every cut.
  Any other backbone raises `TypeError` when the model is built.
- **Training:** supported with limitations. For every model, one forward and backward pass through the split model
  gives the same loss, buffers and gradients as the unsplit model; full training schedules run by MMEngine's Runner
  are not yet tested.
- **Verification:** every MMDetection and MMSegmentation zoo config, with the split-versus-unsplit check done for it,
  is listed in [docs/verified_configs.md](docs/verified_configs.md).

## Loading checkpoints, and what importing the library changes

- **MMDetection's BERT wrapper** (used by GLIP and GroundingDINO) is patched on import so it runs under
  transformers 5.x. Nothing else outside this library is changed.
- **torch.load:** some OpenMMLab checkpoints hold training state (MMEngine's `HistoryBuffer`, NumPy arrays and
  scalars) that `torch.load(weights_only=True)`, PyTorch 2.6's default, rejects. The library allows exactly those types
  (`cfm_model_zoo.TORCH_LOAD_SAFE_GLOBALS`) only while it loads a checkpoint itself; afterwards torch is as it was. To
  read such a checkpoint yourself, wrap `torch.load` in
  `torch.serialization.safe_globals(cfm_model_zoo.TORCH_LOAD_SAFE_GLOBALS)`.
- **A checkpoint that needs anything else** (for example one saved with its training state under NumPy 2) raises
  `cfm_model_zoo.directory.CheckpointGlobalsNotAllowed`, naming the file and what it needs. Checkpoints that fail with
  `CheckpointGlobalsNotAllowed` may be loaded with `build_split_model(..., trust_checkpoint=True)` (also on
  `create_from_cfg_and_checkpoint`). However, this allows that checkpoint to run arbitrary code: only use it for
  trusted checkpoints!

## Where the weights come from

- **OpenMMLab (mim), `source='mim'`:** downloaded once into the cache directory and checked against the server's file
  size. The cache is mim's own (`.cache/mim` in your home directory), or `$CFM_MODEL_ZOO_CACHE_DIR`.
- **DDQ-DETR (ResNet-50, Swin-L), `source='mim'`:** OpenMMLab's weights, downloaded from OpenMMLab's server into the
  cache directory and checked against their size and sha256. Their configs ship with this package, because
  MMDetection 3.3.0's mim index points these two models at configs that do not exist.
- **RepPointsV2, `source='mim'`:** the Mask RepPoints V2 Swin-T 3x weights released by Microsoft's
  [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) (there is
  no mim entry), downloaded into the cache directory and checked against their size and sha256. Their Swin keys are
  converted for the port in `legacy/` when the model is built; the converted copy is written into the cache directory
  and reused while the source file is unchanged. An already converted file, e.g. a previous output of this conversion
  (its name ends in `_converted.pth` or `_converted-<8 hex digits>.pth`), can be passed as it is:
  `build_split_model('MMDetSplitRepPointsV2MaskDetectorSwinT', cut_point=1, source='local',
  checkpoint='/path/to/file_converted.pth')`.
- **`source='local'`:** files already on disk only. This library never downloads under it, for a registered name and a
  raw mim config name alike; a missing file is an error that says to pass `source='mim'`. One exception, under either
  source: GLIP and GroundingDINO build their BERT language model (`bert-base-uncased`) through transformers, which
  downloads it from the Hugging Face Hub on first use and caches it there.

Every download verifies the server's TLS certificate. If verification fails (an outdated CA bundle, or a proxy that
re-signs TLS), `cfm_model_zoo.directory.CertificateVerificationFailed` is raised at once, naming the host and the fix:
`pip install -U certifi`, or `REQUESTS_CA_BUNDLE` pointing at your proxy's CA. `check_certificate=False` skips
verification at your own risk. Any other TLS error is treated as a network error and retried.

This library distributes no weights; downloaded weights remain under their publishers' terms.

## Models and citations

[`cfm_model_zoo/configs/README.md`](cfm_model_zoo/configs/README.md) lists every registered model, the papers to cite
for each, the credits for the code adapted in `cfm_model_zoo/legacy/`, and the BibTeX entries. If you use a model
through this library, please cite the papers listed for it there, and the OpenMMLab toolboxes. The third-party notices
are in [`NOTICE`](NOTICE), which lists what comes from where, and the licence texts are in
[`third_party_licenses/`](third_party_licenses/).

## AI assistance

Parts of this codebase, including its tests and documentation, were written with the assistance of an AI coding tool
(Anthropic's Claude, via Claude Code). The research design, the experiments, the release decisions, and the
verification of every reported result were carried out by the author, who takes full responsibility for the code and its
correctness.

## License

The whole package (its code and configuration files) is for academic use only: see [`LICENSE`](LICENSE). It distributes
no pretrained weights; they are downloaded from their publishers (OpenMMLab; Microsoft's
Swin-Transformer-Object-Detection) and remain under their terms. Third-party components keep their own licences; their
texts are in `third_party_licenses/`, and [`NOTICE`](NOTICE) lists what comes from where. Commercial use is excluded
because `cfm_model_zoo/legacy/` contains code that Swin-Transformer-Object-Detection modified from AdelaiDet and ATSS,
which are licensed for non-commercial use only.

## About this repository

This repository is a public snapshot of a private research repository, updated whenever that research is ready for
publication. It contains only the parts released here, and it starts without that repository's commit history.
