#!/bin/bash
# Docker script to build MMCV, MMDetection, and MMSegmentation with CUDA support
# This runs inside the Docker container during build

set -e  # Exit on error

echo "========================================="
echo "Building MM Libraries for Docker"
echo "========================================="

# Verify environment
echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | grep release || echo 'nvcc not found')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Set environment variables for CUDA build
# CUDA_HOME should be /usr/local/cuda from nvidia/cuda base image
if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

export PATH=${CUDA_HOME}/bin:${PATH}
# Only set LD_LIBRARY_PATH for CUDA libs during build (not conda libs)
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=${MAX_JOBS:-8}
# Set CUDA architectures to compile for (no GPU available at build time)
# 7.5=Turing, 8.0=Ampere(A100), 8.6=Ampere(RTX3090), 9.0=Hopper(H100)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

echo "DEBUG: CUDA_HOME is set to: $CUDA_HOME"
echo "DEBUG: Checking if CUDA_HOME exists: $(ls -ld $CUDA_HOME 2>&1 | head -1)"
echo "DEBUG: nvcc location: $(which nvcc)"
echo "DEBUG: Python location: $(which python)"

# Install build dependencies
echo "Installing build dependencies..."
pip install --no-cache-dir ninja cython

# ===================================================
# 1. Build MMCV from source
# ===================================================
echo ""
echo "[1/3] Building MMCV 2.1.0 with CUDA ops..."

cd /tmp
# Clone should work fine now without library conflicts
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

# Checkout version 2.1.0 (compatible with mmdet 3.3.0)
git checkout v2.1.0

# Install build requirements
pip install --no-cache-dir -r requirements/build.txt

# Build and install with CUDA ops
# Use pip with --no-build-isolation to keep setuptools 69.5.1 (has pkg_resources)
echo "Building MMCV with CUDA ops..."
echo "DEBUG: Checking environment before build..."
python -c "import os; print(f'Python sees CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"NOT SET\")}')"
python -c "import os; print(f'TORCH_CUDA_ARCH_LIST: {os.environ.get(\"TORCH_CUDA_ARCH_LIST\", \"NOT SET\")}')"

# Install (not editable) with --no-build-isolation to preserve setuptools version
pip install . --no-build-isolation

# Verify MMCV installation (change dir first to import from site-packages, not local source)
cd /
python -c "
import mmcv
from mmcv.ops import get_compiling_cuda_version
print(f'✓ MMCV {mmcv.__version__} installed')
print(f'✓ MMCV CUDA: {get_compiling_cuda_version()}')
"

# ===================================================
# 2. Install MMDetection via MIM
# ===================================================
echo ""
echo "[2/3] Installing MMDetection..."

# Install OpenMIM
pip install --no-cache-dir openmim

# Fix: openxlab dependency downgrades setuptools to 60.2.0, which breaks with Python 3.12
# Reinstall setuptools 69.5.1 to fix the ImpImporter error
echo "Reinstalling setuptools 69.5.1 after openmim dependencies..."
pip install --no-cache-dir 'setuptools==69.5.1'

# Install MMDetection 3.3.0 and MMSegmentation 1.2.2
mim install mmdet==3.3.0
mim install mmsegmentation==1.2.2

# Verify MM libraries installation
python -c "
import mmdet
import mmseg
import mmcv
print(f'✓ MMDetection {mmdet.__version__} installed')
print(f'✓ MMSegmentation {mmseg.__version__} installed')
print(f'✓ Compatible with MMCV {mmcv.__version__}')
"

# ===================================================
# 3. Test CUDA operations
# ===================================================
echo ""
echo "Testing CUDA operations..."

python -c "
import torch
import mmcv
from mmcv.ops import box_iou_rotated, nms

if torch.cuda.is_available():
    # Test rotated box IoU
    boxes1 = torch.randn(10, 5).cuda()
    boxes2 = torch.randn(10, 5).cuda()
    ious = box_iou_rotated(boxes1, boxes2)
    print(f'✓ box_iou_rotated test passed: {ious.shape}')

    # Test NMS
    boxes = torch.randn(100, 4).cuda()
    scores = torch.randn(100).cuda()
    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f'✓ nms test passed: {len(keep)} boxes kept')

    print('✓ All CUDA ops working correctly!')
else:
    print('⚠ CUDA not available - ops will run on CPU')
"

# Cleanup
echo ""
echo "Cleaning up build artifacts..."
cd /
rm -rf /tmp/mmcv
pip cache purge

echo ""
echo "========================================="
echo "✓ MM Libraries build complete!"
echo "========================================="
echo "Installed:"
echo "  - MMCV 2.1.0 (with CUDA ops)"
echo "  - MMDetection 3.3.0"
echo "  - MMSegmentation 1.2.2"
echo "========================================="
