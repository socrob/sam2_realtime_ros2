# Installation procedure on Dolores Desktop

# 0) Make sure CUDA is discoverable
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# 1) Use a known-good host compiler (you already installed gcc-12/g++-12)
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

# 2) Make the build single-threaded & disable ninja (ninja sometimes hides the real failure)
export MAX_JOBS=1
export NVCC_THREADS=1
export USE_NINJA=0

# 3) Set your GPU arch explicitly (pick ONE that matches your GPU)
#   - Turing (RTX 20xx)  : 7.5
#   - Ampere (RTX 30xx)  : 8.6
#   - Ada (RTX 40xx)     : 8.9
# If unsure but you know it's 30xx, use 8.6, for 40xx use 8.9, etc.
export TORCH_CUDA_ARCH_LIST="7.5"

# 4) Slightly tame compiler optimizations (avoids rare nvcc segfaults)
export CXXFLAGS="-O2"
export NVCC_FLAGS="--expt-relaxed-constexpr"

# 5) Make sure Python build tooling is present (you did most already)
python -m pip install -U setuptools wheel  # keep ninja out of the path with USE_NINJA=0

# 6) Retry with verbose logs
pip install -v --no-build-isolation --no-cache-dir -e ".[notebooks]"
