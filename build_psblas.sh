#!/bin/bash
set -e

# Load environments
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh

# We must ensure MPI is loaded before running cmake if Spack provides it
# For PSBLAS 3, you also need BLAS/LAPACK. We'll use the ones provided by Intel MKL
spack load intel-oneapi-mkl
spack load openmpi || echo "Warning: Ensure MPI is available"

# If doesn't exist, create a directory called resources/psblas3 and clone psblas3 into it
if [ ! -d "resources/psblas3" ]; then
    mkdir -p resources
    cd resources
    git clone https://github.com/sfilippone/psblas3.git
    cd ..
fi

# Directory definitions
SRC_DIR="$(pwd)/resources/psblas3"
BUILD_DIR="$SRC_DIR/build"
INSTALL_DIR="$(pwd)/local/psblas3"

# Clean up any existing build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

cd "$BUILD_DIR"

echo # Configure with optimized flags matching Spack defaults for other libraries + fPIC for Rust static linkage
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_DOCS=OFF \
    -DENABLE_SERIAL=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_FLAGS="-g -O3 -march=native -mtune=native -flto -fPIC" \
    -DCMAKE_CXX_FLAGS="-g -O3 -march=native -mtune=native -flto -fPIC" \
    -DCMAKE_Fortran_FLAGS="-g -O3 -march=native -mtune=native -ffree-line-length-none -frecursive -fPIC"

echo "Compiling PSBLAS..."
make -j$(nproc)

echo "Installing PSBLAS to $INSTALL_DIR..."
make install

echo "PSBLAS compilation complete. Installation located at $INSTALL_DIR"
