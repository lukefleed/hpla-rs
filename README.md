# Single-Threaded Sparse Kernel Benchmarks

Compares sparse matrix kernel performance across Rust and C/C++/Fortran numerical libraries on a single core. Three kernels are benchmarked:

1. **SpMV**: `y += A*x` (BLAS-style sparse matrix-vector product).
2. **One-pass Lanczos for `exp(-A)b`**: computes the matrix exponential applied to a vector by materializing the full Lanczos basis `V_m` in memory, then forming `V_m * exp(-T_m) * e_1 * ||b||`. Memory footprint O(n·m).
3. **Two-pass Lanczos for `exp(-A)b`**: same output as kernel 2, trading one extra pass of `m` matrix-vector products for O(n) memory (the basis is discarded in pass 1 and regenerated on the fly in pass 2).

Kernels 2 and 3 produce the same output vector, so the comparison isolates the memory/compute trade-off described in Section 2.4 of the companion paper.

## SpMV Backends

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

## Lanczos Backends

Two bench binaries, one per kernel. Both iterate the same dedicated matrix suite and share the Krylov dimension determined adaptively per matrix via the Saad (1992) a posteriori error estimate.

### Matrix suite

The Lanczos benches use a dedicated set of symmetric sparse matrices whose mean diagonal is zero or small, so the Saad a posteriori error estimator on `exp(-A)b` is meaningful at every Krylov dimension. This excludes raw FEM stiffness matrices, whose large self-stiffness diagonal entries make `exp(-alpha_1)` underflow at the first Lanczos step. The list lives in `src/lib.rs::LANCZOS_SUITE` and is shared between the benches and the equivalence tests.

| Matrix | SuiteSparse group | Class |
|---|---|---|
| `kron_g500-logn18` | `DIMACS10` | synthetic Kronecker graph |
| `coPapersDBLP` | `DIMACS10` | bibliometric co-citation graph |
| `thermal2` | `Schmid` | thermal diffusion PDE |
| `as-Skitter` | `SNAP` | internet AS topology |
| `roadNet-CA` | `SNAP` | road network |
| `delaunay_n22` | `DIMACS10` | random planar triangulation |

The first run of `./download_matrices.sh` after a clean checkout downloads any missing matrices.

### One-pass Lanczos for `exp(-A)b` (`cargo bench --bench lanczos`)

| Configuration | Library | Language |
|---------------|---------|----------|
| `faer/one_pass` | [faer](https://github.com/sarah-quinones/faer-rs) 0.24 | Rust |
| `psblas/one_pass` | [PSBLAS](https://github.com/sfilippone/psblas3) | Fortran |

Throughput metric: `m * (2*nnz + 11*n)` FLOPs per iteration, covering the `m`-step recurrence (`2*nnz + 9n` per step) and the final `V_m * g` gemv (`2*m*n`).

### Two-pass Lanczos for `exp(-A)b` (`cargo bench --bench lanczos_two_pass`)

| Configuration | Library | Language |
|---------------|---------|----------|
| `faer/two_pass` | [faer](https://github.com/sarah-quinones/faer-rs) 0.24 | Rust |
| `psblas/two_pass` | [PSBLAS](https://github.com/sfilippone/psblas3) | Fortran |

Throughput metric: `4k * (nnz + 4n)` FLOPs per iteration, covering both SpMV and vector recurrence work across the two Lanczos passes.

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

**SpMV:** Criterion runs 100 samples per backend, 3 s warm-up, 20 s measurement. Throughput: `2 * nnz` FLOPs per iteration.

**Lanczos (one-pass and two-pass):** 50 samples, 3 s warm-up, 30 s measurement. Throughput: `m * (2*nnz + 11*n)` for one-pass, `4k * (nnz + 4n)` for two-pass, where `m = k` is the adaptively determined Krylov dimension shared between the two benches.

### Compiler Profile

| Component | Compiler | Flags |
|-----------|----------|-------|
| PETSc, PSBLAS | gcc / gfortran | `-O3 -march=native -mtune=native -flto` |
| C/C++ FFI wrappers (`ffi/spmv/`) | clang / clang++ | `-O3 -march=native -mtune=native -ffast-math -flto` |
| Fortran FFI wrappers (`ffi/lanczos/`) | gfortran | `-O3 -march=native -mtune=native -ffast-math -ffat-lto-objects` |
| Rust (faer + harness) | rustc (LLVM) | `opt-level=3, lto="fat", codegen-units=1` |
| MKL | Intel (precompiled) | n/a |

| Setting | Value |
|---------|-------|
| BLAS backend | Intel MKL |
| CPU affinity | `taskset -c 0` |
| Threading | `OMP_NUM_THREADS=1` |
| Cargo panic | `"abort"` |
