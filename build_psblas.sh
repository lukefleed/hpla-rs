#!/usr/bin/env bash
# Build PSBLAS 3 from source into local/psblas3/.
#
# PSBLAS is not available via spack, so we build it locally with the same
# compiler flags used for the other backends (Clang + LTO + fPIC for Rust
# static linking).
#
# Prerequisites:
#   source ~/.cargo/env
#   source ~/spack/share/spack/setup-env.sh
#   spack load intel-oneapi-mkl openmpi
#
# References:
#   https://github.com/sfilippone/psblas3
#   https://psctoolkit.github.io/products/psblas/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${REPO_ROOT}/resources/psblas3"
BUILD_DIR="${SRC_DIR}/build"
INSTALL_DIR="${REPO_ROOT}/local/psblas3"

# Clone if not present
if [ ! -d "${SRC_DIR}" ]; then
    echo "==> Cloning PSBLAS 3"
    git clone --depth 1 https://github.com/sfilippone/psblas3.git "${SRC_DIR}"
fi

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

echo "==> Configuring PSBLAS (install to ${INSTALL_DIR})"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DOCS=OFF \
    -DENABLE_SERIAL=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_FLAGS="-O3 -march=native -ffast-math -flto -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math -flto -fPIC" \
    -DCMAKE_Fortran_FLAGS="-O3 -march=native -ffree-line-length-none -frecursive -fPIC"

echo "==> Building PSBLAS"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "==> Installing"
cmake --install "${BUILD_DIR}"

echo "==> Done. Libraries: $(ls "${INSTALL_DIR}/lib/"*.a 2>/dev/null | wc -l) static archives"
