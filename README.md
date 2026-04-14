# Single-Threaded Sparse Kernel Benchmarks

Benchmarks for three sparse matrix kernels comparing Rust ([faer](https://github.com/sarah-quinones/faer-rs)) against C/C++ libraries (PETSc, Eigen, MKL) on a single core.

| Kernel | What it computes | Memory |
|--------|-----------------|--------|
| SpMV | `y += A*x` | O(nnz) |
| One-pass Lanczos | `exp(-A)b` via full Krylov basis | O(n*m) |
| Two-pass Lanczos | `exp(-A)b` via basis-free reconstruction | O(n) |

The two Lanczos variants compute the same result. The difference is memory vs compute: one-pass stores the full basis V_m, two-pass discards it and replays the recurrence.

## Backends

### SpMV

| Backend | Library | Format | Language |
|---------|---------|--------|----------|
| `faer/csc` | faer 0.24 | CSC | Rust |
| `faer/csr` | faer 0.24 | CSR | Rust |
| `petsc/csr_inodes` | PETSc 3.24 | CSR + Inode | C |
| `petsc/csr_raw` | PETSc 3.24 | CSR scalar | C |
| `eigen/csc_map` | Eigen | CSC | C++ |
| `eigen/csr_map` | Eigen | CSR | C++ |
| `mkl/csr_ie` | Intel MKL | CSR IE | C |
| `mkl/csc_ie` | Intel MKL | CSC IE | C |

### Lanczos

Both benches share the same matrix suite and Krylov dimension (determined adaptively per matrix via the Saad 1992 a posteriori error estimate).

| Backend | Kernel | Library | Format | Language |
|---------|--------|---------|--------|----------|
| `faer_csc` | one-pass, two-pass | faer 0.24 | CSC | Rust |
| `faer_csr` | one-pass, two-pass | faer 0.24 | CSR | Rust |
| `eigen_csr` | one-pass, two-pass | Eigen | CSR | C++ |
| `eigen_csc` | one-pass, two-pass | Eigen | CSC | C++ |

Each library is tested in both CSR and CSC to separate format effects from library quality (2x2 factorial design).

### Matrix suite

Six symmetric matrices with small or zero mean diagonal, so the Saad estimator on `exp(-A)b` is meaningful at every Krylov dimension.

| Matrix | Group | Class |
|--------|-------|-------|
| `kron_g500-logn18` | DIMACS10 | synthetic Kronecker graph |
| `coPapersDBLP` | DIMACS10 | co-citation graph |
| `thermal2` | Schmid | thermal diffusion PDE |
| `as-Skitter` | SNAP | internet AS topology |
| `roadNet-CA` | SNAP | road network |
| `delaunay_n22` | DIMACS10 | Delaunay triangulation |

## Setup

Install dependencies via [Spack](https://spack.io/):

```bash
source ~/spack/share/spack/setup-env.sh
spack env activate -d .
spack concretize -f && spack install
```

Download matrices:

```bash
bash download_matrices.sh
```

## Build and run

```bash
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh
spack env activate -d .

cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test

export RUSTFLAGS="-C target-cpu=native"
export OMP_NUM_THREADS=1
taskset -c 0 cargo bench --bench spmv
taskset -c 0 cargo bench --bench lanczos
taskset -c 0 cargo bench --bench lanczos_two_pass
```

## Plotting

```bash
cd python
python3 plot.py spmv              # bar charts + roofline
python3 plot.py lanczos           # one-pass Lanczos bar charts
python3 plot.py lanczos_two_pass  # two-pass Lanczos bar charts
```

SpMV roofline requires STREAM Triad bandwidth. Measure it first:

```bash
bash stream_bench.sh   # writes python/hw_config.json
```

## Hardware

All benchmarks run pinned to a single core on a dual-socket Intel Xeon Gold 5318Y (Ice Lake-SP).

| | |
|---|---|
| CPU | Intel Xeon Gold 5318Y @ 2.10 GHz |
| Microarchitecture | Ice Lake-SP |
| L1d / L2 / L3 | 48 KB / 1.25 MB / 36 MB per socket |
| ISA | AVX-512F/BW/VL/VNNI |
| STREAM Triad (1 core) | 13.53 GB/s |

## Compiler flags

| Component | Compiler | Flags |
|-----------|----------|-------|
| C/C++ FFI wrappers | clang/clang++ | `-O3 -march=native -mtune=native -ffast-math -flto` |
| Rust (faer + harness) | rustc (LLVM) | `opt-level=3, lto="fat", codegen-units=1` |
| Eigen Lanczos wrappers | clang++ C++20 | same as above |

All backends receive the same CSR/CSC arrays allocated once by Rust. PETSc and Eigen wrap them zero-copy. MKL may build optimized internal copies. Setup cost is excluded from the timed loop.
