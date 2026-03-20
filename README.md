# hpla-rs — Single-Threaded SpMV Benchmark Suite

A benchmarking framework for comparing Sparse Matrix-Vector multiplication (`y += A*x`) across Rust and C/C++/Fortran libraries under strictly identical, single-threaded conditions. Designed for publication in an international supercomputing journal.

## Backends

| Configuration | Library | Format | Language |
|---------------|---------|--------|----------|
| `faer/csc` | [Faer](https://github.com/sarah-quinones/faer-rs) | CSC | Rust |
| `petsc/csr_inodes` | [PETSc](https://petsc.org/) | CSR + Inode optimization | C |
| `petsc/csr_raw` | [PETSc](https://petsc.org/) | CSR scalar | C |
| `eigen/csc_map` | [Eigen](https://eigen.tuxfamily.org/) | CSC via `Eigen::Map` | C++ |
| `eigen/csr_map` | [Eigen](https://eigen.tuxfamily.org/) | CSR via `Eigen::Map` (cross-format control) | C++ |
| `mkl/csr_ie` | [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | CSR Inspection-Execution | C |
| `mkl/csc_ie` | [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | CSC Inspection-Execution (cross-format control) | C |
| `psblas/csr` | [PSBLAS](https://github.com/sfilippone/psblas3) | CSR via Fortran C bindings | C++/Fortran |

## Prerequisites

All external dependencies are managed through [Spack](https://spack.io/).

```bash
# PETSc (serial, no optional solvers, LTO-enabled)
spack install petsc ~mpi ~hdf5 ~superlu-dist ~mumps ~suite-sparse ~scalapack \
    ~strumpack ~ptscotch ~hwloc ~X \
    cflags='-O3 -march=native -mtune=native -flto' \
    cxxflags='-O3 -march=native -mtune=native -flto' \
    fflags='-O3 -march=native -mtune=native -flto -ffree-line-length-none'

# Eigen (header-only)
spack install eigen

# Intel MKL
spack install intel-oneapi-mkl

# OpenMPI (required by PSBLAS)
spack install openmpi
```

Download test matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/):

```bash
mkdir -p matrices && cd matrices
# Example: atmosmodd (1.27M rows, 8.8M nnz) and thermal2 (1.23M rows, 8.6M nnz)
for m in Bourchtein/atmosmodd Schmid/thermal2; do
    wget "https://suitesparse-collection-website.herokuapp.com/MM/${m}.tar.gz"
done
for f in *.tar.gz; do tar xf "$f" --strip-components=1; done
rm -f *.tar.gz
cd ..
```

## Building PSBLAS

[PSBLAS](https://github.com/sfilippone/psblas3) is not available via spack and must be built locally. A script is provided:

```bash
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh
spack load intel-oneapi-mkl openmpi
bash build_psblas.sh
```

This clones the repo into `resources/psblas3/`, builds with cmake (Clang + LTO + fPIC), and installs static libraries into `local/psblas3/`. See [PSBLAS documentation](https://psctoolkit.github.io/products/psblas/) for details.

## Build and Run

```bash
# Load environment
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh
spack load intel-oneapi-mkl openmpi

# Verify compilation
cargo check --all-targets

# Run benchmarks (single-threaded, pinned to core 0)
taskset -c 0 cargo bench
```

Results are written to `target/criterion/`. Generate plots with:

```bash
cd python && python3 plot.py
```

### Roofline Analysis

To overlay a bandwidth ceiling on the plots, first measure single-core STREAM Triad bandwidth:

```bash
bash stream_bench.sh
```

This produces `python/hw_config.json` with the measured bandwidth. Then re-run `plot.py` — it will generate:
- `python/gemv/roofline.png` — classic roofline (log-log) with all backends and matrices
- Per-matrix bar charts with a dashed red bandwidth ceiling line

The roofline uses a cold-cache compulsory traffic model:
`AI = 2*nnz / ((nrows+1)*4 + nnz*12 + ncols*8 + nrows*16)` FLOP/byte.

## Architecture

### Zero-Copy FFI

All backends operate on the same raw memory buffers (`row_ptr`, `col_idx`, `values`) allocated once by Rust. PETSc uses `MatCreateSeqAIJWithArrays`, Eigen uses `Eigen::Map`, and MKL uses `mkl_sparse_d_create_csr` — all zero-copy projections of the same data.

### Cross-Language LTO

Both `rustc` and `clang` emit LLVM IR. By compiling C/C++ wrappers with `-flto` and configuring Cargo with `lto = "fat"`, the LLVM linker treats the Rust caller and C/C++ kernels as a single compilation unit. This eliminates ABI call overhead by inlining FFI wrappers directly into the benchmark loop.

### Benchmark Methodology

The Criterion harness runs 100 samples per backend with 3 s warm-up and 8 s measurement, using `Throughput::Elements(2 * nnz)` to report GFLOP/s (one multiply + one add per nonzero).

All backends accumulate `y += A*x` without resetting `y` between iterations. This measures the kernel in steady-state without injecting a `memcpy` into the hot path. `criterion::black_box` is applied only to the Faer backend (pure Rust in the same LTO unit); FFI backends are inherently opaque to LLVM's dead code elimination.

### Compiler Profile

| Setting | Value |
|---------|-------|
| C/C++ flags | `-O3 -march=native -ffast-math -flto` |
| `opt-level` | `3` |
| `codegen-units` | `1` |
| `lto` | `"fat"` |
| `panic` | `"abort"` |
| CPU affinity | `taskset -c 0` |
| Threading | `OMP_NUM_THREADS=1` |

## License

MIT
