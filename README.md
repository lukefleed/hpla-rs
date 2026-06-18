# Evaluating Rust for Sparse Matrix Kernels in Scientific Computing

This repository contains the benchmark companion for the work [Evaluating Rust for Sparse Matrix Kernels in Scientific Computing](https://arxiv.org/abs/2606.19213). It compares native Rust sparse kernels against PETSc, Eigen, Intel oneMKL, and PSBLAS on one CPU core.

The benchmark covers three kernels:

| Kernel | Timed operation | Dominant matrix-dependent storage |
|--------|-----------------|-----------------------------------|
| SpMV | `y += A*x` | `O(nnz)` |
| One-pass Lanczos | `exp(-A)b`, storing the Krylov basis | `O(n*m)` |
| Two-pass Lanczos | `exp(-A)b`, reconstructing the basis in a second pass | `O(n)` |

The Rust implementation uses a [fork](https://codeberg.org/lukefleed/faer) of [`faer`](https://codeberg.org/sarah-quinones/faer) that contains the CSR SpMV specialization evaluated in this work. The dependency is declared in [`Cargo.toml`](Cargo.toml) and pinned by [`Cargo.lock`](Cargo.lock).

## Backend Configuration

### Sparse Matrix-Vector Product

| Benchmark ID | Library | Format | Implementation language |
|--------------|---------|--------|--------------------------|
| `faer/csc` | faer | CSC | Rust |
| `faer/csr` | faer | CSR | Rust |
| `petsc/csr` | PETSc | CSR | C |
| `eigen/csc_map` | Eigen | CSC | C++ |
| `eigen/csr_map` | Eigen | CSR | C++ |
| `psblas/csr` | PSBLAS | CSR | C++/Fortran |
| `psblas/csc` | PSBLAS | CSC | C++/Fortran |
| `mkl/csr_ie` | Intel oneMKL | CSR inspection-execution | C |
| `mkl/csc_ie` | Intel oneMKL | CSC inspection-execution | C |

### Lanczos Matrix-Function Evaluation

Both Lanczos benchmarks use the same starting vector and the same Krylov dimension for every backend on a given matrix. The Krylov dimension is selected with [Saad's a posteriori estimator](https://epubs.siam.org/doi/10.1137/0729014) after scaling the matrix by an estimated spectral radius.

| Benchmark ID prefix | Kernels | Library | Format | Implementation language |
|---------------------|---------|---------|--------|--------------------------|
| `faer_csc` | one-pass, two-pass | faer | CSC | Rust |
| `faer_csr` | one-pass, two-pass | faer | CSR | Rust |
| `eigen_csr` | one-pass, two-pass | Eigen | CSR | C++ |
| `eigen_csc` | one-pass, two-pass | Eigen | CSC | C++ |
| `petsc_csr` | one-pass, two-pass | PETSc | CSR | C |
| `psblas_csr` | one-pass, two-pass | PSBLAS | CSR | C++/Fortran |
| `psblas_csc` | one-pass, two-pass | PSBLAS | CSC | C++/Fortran |

## Matrix Suite

[`download_matrices.sh`](download_matrices.sh) downloads 28 [SuiteSparse](https://sparse.tamu.edu/) matrices used by the SpMV benchmark:

`amazon0302`, `atmosmodd`, `cant`, `inline_1`, `rajat31`, `thermal2`, `web-Google`, `audikw_1`, `Queen_4147`, `shipsec1`, `pdb1HYS`, `consph`, `mac_econ_fwd500`, `circuit5M`, `roadNet-CA`, `kron_g500-logn18`, `coPapersDBLP`, `as-Skitter`, `delaunay_n22`, `caidaRouterLevel`, `citationCiteseer`, `coAuthorsCiteseer`, `coPapersCiteseer`, `preferentialAttachment`, `smallworld`, `rgg_n_2_20_s0`, `belgium_osm`, and `auto`.

The Lanczos benchmarks use the 15-matrix symmetric subset declared in [`src/lib.rs`](src/lib.rs). These matrices have small or zero mean diagonal, which keeps the Saad estimator meaningful for `exp(-A)b` at the target tolerance.

| Matrix | Group | Class |
|--------|-------|-------|
| `kron_g500-logn18` | DIMACS10 | synthetic Kronecker graph |
| `coPapersDBLP` | DIMACS10 | co-citation graph |
| `thermal2` | Schmid | thermal diffusion PDE |
| `as-Skitter` | SNAP | internet AS topology |
| `roadNet-CA` | SNAP | road network |
| `delaunay_n22` | DIMACS10 | Delaunay triangulation |
| `caidaRouterLevel` | DIMACS10 | router-level topology |
| `citationCiteseer` | DIMACS10 | citation graph |
| `coAuthorsCiteseer` | DIMACS10 | co-author graph |
| `coPapersCiteseer` | DIMACS10 | co-paper graph |
| `preferentialAttachment` | DIMACS10 | scale-free benchmark graph |
| `smallworld` | DIMACS10 | small-world benchmark graph |
| `rgg_n_2_20_s0` | DIMACS10 | random geometric graph |
| `belgium_osm` | DIMACS10 | OpenStreetMap road network |
| `auto` | DIMACS10 | Walshaw graph-partitioning benchmark |

## Toolchain and Environment

The Rust toolchain is pinned in [`rust-toolchain.toml`](rust-toolchain.toml) to `nightly-2026-04-14`, whose bundled LLVM major version matches the Spack `llvm@22` toolchain used for C and C++ wrappers. [`build.rs`](build.rs) checks this match before compiling the wrappers.

Native dependencies are installed through the Spack environment in [`spack.yaml`](spack.yaml). The environment provides Intel oneMKL, Eigen, PETSc, PSBLAS, OpenMPI, OpenBLAS, GCC 14.3.0, and LLVM 22. PETSc is linked against oneMKL. PSBLAS is linked against OpenMPI and OpenBLAS.

On a new machine, bootstrap the Spack compiler once:

```bash
source ~/spack/share/spack/setup-env.sh
spack external find
spack install gcc@14.3.0 languages=fortran,c,c++
spack compiler add $(spack location -i gcc@14.3.0)
```

Then install the environment from the repository root:

```bash
source ~/spack/share/spack/setup-env.sh
spack env activate -d .
spack concretize -f
spack install -v
```

## Verification

These commands check that the Rust crate and native wrappers build in the active Spack environment:

```bash
source ~/spack/share/spack/setup-env.sh
spack env activate -d .

cargo check --all-targets
cargo clippy --all-targets
```

The equivalence tests in [`src/tests.rs`](src/tests.rs) link the native wrappers and run numerical checks on `kron_g500-logn18`, `coPapersDBLP`, and `thermal2`. `cargo test --release` checks SpMV plus one-pass and two-pass Lanczos: each backend result is compared with the `faer/CSC` reference by relative L2 error. These tests can take substantially longer than `cargo check`:

```bash
bash download_matrices.sh
cargo test --release
```

## Benchmark Execution

The benchmark harnesses live in [`benches/spmv.rs`](benches/spmv.rs), [`benches/lanczos.rs`](benches/lanczos.rs), and [`benches/lanczos_two_pass.rs`](benches/lanczos_two_pass.rs). Benchmark runs must be single-threaded and pinned to one core:

```bash
source ~/.cargo/env
source ~/spack/share/spack/setup-env.sh
spack env activate -d .

bash download_matrices.sh
export RUSTFLAGS="-C target-cpu=native"
export OMP_NUM_THREADS=1

taskset -c 0 cargo bench --bench spmv
taskset -c 0 cargo bench --bench lanczos
taskset -c 0 cargo bench --bench lanczos_two_pass
```

The three `cargo bench` commands write Criterion sample data under `target/criterion/`. `spmv` measures steady-state `y += A*x` for all SpMV backends on the downloaded matrix suite. `lanczos` measures the complete one-pass `exp(-A)b` evaluation, and `lanczos_two_pass` measures the same matrix-function workload with the two-pass reconstruction.

All Criterion groups use 50 samples, a 5 second warm-up, and a 100 second measurement window. Setup is outside the timed loop: matrix loading, format construction, backend context creation, MKL inspection, PSBLAS assembly, and workspace allocation are completed before Criterion enters `bench.iter`.

## Data and Figures

The repository keeps the figure inputs as CSV tables under [`python/data/`](python/data/):

- [`raw_samples.csv`](python/data/raw_samples.csv): one row per Criterion sample, with the benchmark, matrix, configuration, iteration count, sample time, and derived per-sample throughput.
- [`summary.csv`](python/data/summary.csv): one row per `(benchmark, matrix, configuration)`. [`python/plot.py`](python/plot.py) estimates time per iteration by an ordinary least-squares fit through the origin on `(iteration_count, sample_time)`, then derives throughput from that estimate.
- [`lanczos_accuracy.csv`](python/data/lanczos_accuracy.csv): relative L2 differences against the `faer/CSC` reference, generated by [`lanczos_accuracy.rs`](src/bin/lanczos_accuracy.rs).

The plotting pipeline is:

- To reproduce the included figures, run [`python/plot.py`](python/plot.py) on the committed CSV files in [`python/data/`](python/data/).
- After a new benchmark run, `cargo bench` writes raw samples under `target/criterion/**/new/raw.csv`, and `python3 python/plot.py export-csv` converts them into [`raw_samples.csv`](python/data/raw_samples.csv) and [`summary.csv`](python/data/summary.csv).
- Accuracy and roofline data use separate inputs: `cargo run --release --bin lanczos_accuracy -- --output python/data/lanczos_accuracy.csv` regenerates [`lanczos_accuracy.csv`](python/data/lanczos_accuracy.csv), while [`stream_bench.sh`](stream_bench.sh) writes the local `hw_config.json` used by the SpMV roofline.

Install plotting dependencies from [`python/requirements.txt`](python/requirements.txt):

```bash
python3 -m pip install -r python/requirements.txt
```

To reproduce the figures from the included data, run the plotting commands directly:

```bash
python3 python/plot.py lanczos_one_pass
python3 python/plot.py lanczos_two_pass
python3 python/plot.py perfprof
python3 python/plot.py violin
python3 python/plot.py accuracy
```

The `all` command in [`python/plot.py`](python/plot.py) runs the SpMV, Lanczos, performance-profile, and violin plot commands. It also requires the local `python/hw_config.json` written by [`stream_bench.sh`](stream_bench.sh), because the SpMV roofline is part of the `all` target.

## Hardware Environment

The measurements in this work were collected on one pinned core of this machine:

| | |
|---|---|
| CPU | Intel Xeon Gold 6418H, 24 cores per socket, 2 hardware threads per core |
| Base frequency | 2.10 GHz |
| Microarchitecture | Sapphire Rapids |
| System memory | 2 TiB |
| Cache | 48 KiB L1d, 2 MiB L2 per core, 60 MiB L3 per socket |
| ISA | AVX-512F/BW/VL/VNNI/BF16/FP16, AMX |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-111-generic, x86_64 |

The benchmark process is pinned with `taskset -c 0`, library threading is disabled with `OMP_NUM_THREADS=1`, and the MKL backend links against `mkl_sequential`.

## Compilers


| Component | Compiler | Flags |
|-----------|----------|-------|
| C/C++ FFI wrappers | Spack `clang`/`clang++` from LLVM 22 | `-O3 -march=native -mtune=native -flto` |
| Eigen Lanczos wrappers | Spack `clang++` from LLVM 22, C++20 | `-O3 -march=native -mtune=native -flto` |
| Fortran PSBLAS wrappers | Spack `gfortran` from GCC 14.3.0 | `-O3 -march=native -mtune=native -ffat-lto-objects` |
| Rust harness and native kernels | `rustc` from `nightly-2026-04-14` | `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

All backends receive sparse arrays derived from the same [`RawMatrix`](src/lib.rs). PETSc and Eigen wrap Rust-owned arrays without copying. MKL may build optimized internal data during inspection. PSBLAS assembles its own sparse descriptor. The timed loop measures only backend execution on already constructed operands.

## Cite this work

You can cite this work with the following BibTeX entry:

```bibtex
@misc{lombardo2026evaluatingrustsparsematrix,
      title={Evaluating Rust for Sparse Matrix Kernels in Scientific Computing},
      author={Luca Lombardo and Fabio Durastante},
      year={2026},
      eprint={2606.19213},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2606.19213},
}
```
