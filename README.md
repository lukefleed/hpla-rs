# Single-Threaded SpMV Benchmark

Compares Sparse Matrix-Vector multiplication (`y += A*x`) performance across Rust and C/C++/Fortran numerical libraries on a single core.

## Backends

| Configuration | Library | Format | Language |
|---------------|---------|--------|----------|
| `faer/csc` | [faer](https://github.com/sarah-quinones/faer-rs) 0.24 | CSC | Rust |
| `faer/csr` | [faer](https://github.com/sarah-quinones/faer-rs) 0.24 | CSR (cross-format control) | Rust |
| `petsc/csr_inodes` | [PETSc](https://petsc.org/) 3.24 | CSR + Inode optimization | C |
| `petsc/csr_raw` | [PETSc](https://petsc.org/) 3.24 | CSR scalar | C |
| `eigen/csc_map` | [Eigen](https://eigen.tuxfamily.org/) | CSC via `Eigen::Map` | C++ |
| `eigen/csr_map` | [Eigen](https://eigen.tuxfamily.org/) | CSR via `Eigen::Map` (cross-format control) | C++ |
| `mkl/csr_ie` | [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | CSR Inspection-Execution | C |
| `mkl/csc_ie` | [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | CSC Inspection-Execution (cross-format control) | C |
| `psblas/csr` | [PSBLAS](https://github.com/sfilippone/psblas3) | CSR via Fortran C bindings | C++/Fortran |
| `psblas/csc` | [PSBLAS](https://github.com/sfilippone/psblas3) | CSC via Fortran C bindings (cross-format control) | C++/Fortran |

Faer, Eigen, MKL, and PSBLAS each provide both CSR and CSC variants to isolate the effect of storage format from kernel quality.

## Target Hardware

All benchmarks run on a dual-socket Intel Xeon Gold 5318Y (Ice Lake-SP), pinned to a single core via `taskset -c 0`.

| | |
|---|---|
| CPU | Intel Xeon Gold 5318Y @ 2.10 GHz |
| Microarchitecture | Ice Lake-SP (family 6, model 106) |
| Sockets / Cores / Threads | 2 / 24 per socket / 2 per core |
| L1d / L2 / L3 | 48 KB / 1.25 MB / 36 MB per socket |
| ISA extensions | AVX-512F, AVX-512BW, AVX-512VL, AVX-512VNNI |
| STREAM Triad (1 core) | 13.53 GB/s |

## Prerequisites

[`spack.yaml`](spack.yaml) pins all external dependencies.

```bash
source ~/spack/share/spack/setup-env.sh
spack env activate -d .
spack concretize -f
spack install
```

Download test matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/):

```bash
bash download_matrices.sh
```

## Build and Run

```bash
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh
spack env activate -d .

cargo check --all-targets

export RUSTFLAGS="-C target-cpu=native"
export OMP_NUM_THREADS=1
taskset -c 0 cargo bench
```

Results land in `target/criterion/`. Generate plots:

```bash
cd python && python3 plot.py
```

### Roofline

Measure single-core STREAM Triad bandwidth first:

```bash
bash stream_bench.sh
```

This writes `python/hw_config.json`. Re-running `plot.py` produces a roofline plot and per-matrix bar charts with bandwidth ceiling lines. Cold-cache compulsory traffic model: `AI = 2*nnz / ((nrows+1)*4 + nnz*12 + ncols*8 + nrows*16)` FLOP/byte.

## Architecture

### Data Sharing

All backends receive the same CSR/CSC arrays allocated once by Rust. PETSc (`MatCreateSeqAIJWithArrays`) and Eigen (`Eigen::Map`) wrap them without copying. MKL stores pointers at handle creation; `mkl_sparse_optimize` may build optimized internal representations. PSBLAS copies during assembly. Setup is excluded from the timed loop.

### Cross-Language LTO

C/C++ wrappers compile with clang `-flto`. Combined with Cargo `lto = "fat"`, the LLVM linker can inline FFI wrappers into the benchmark loop. Library kernels (PETSc, MKL, PSBLAS) are pre-compiled and not subject to cross-language LTO.

### Measurement

Criterion runs 100 samples per backend, 3 s warm-up, 20 s measurement. Throughput reported as GFLOP/s via `Throughput::Elements(2 * nnz)`. All backends accumulate `y += A*x` without resetting between iterations.

### Compiler Profile

| Component | Compiler | Flags |
|-----------|----------|-------|
| PETSc, PSBLAS | gcc / gfortran | `-O3 -march=native -mtune=native -flto` |
| FFI wrappers | clang / clang++ | `-O3 -march=native -mtune=native -ffast-math -flto` |
| Rust (faer + harness) | rustc (LLVM) | `opt-level=3, lto="fat", codegen-units=1` |
| MKL | Intel (precompiled) | — |

| Setting | Value |
|---------|-------|
| BLAS backend | Intel MKL |
| CPU affinity | `taskset -c 0` |
| Threading | `OMP_NUM_THREADS=1` |
| Cargo panic | `"abort"` |
