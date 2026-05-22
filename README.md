# Single-Threaded Sparse Kernel Benchmarks

Benchmarks for three sparse matrix kernels comparing [faer](https://github.com/sarah-quinones/faer-rs) with PETSc, Eigen, MKL, and PSBLAS on a single core.

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
| `faer/csc` | faer | CSC | Rust |
| `faer/csr` | faer | CSR | Rust |
| `petsc/csr` | PETSc | CSR | C |
| `eigen/csc_map` | Eigen | CSC | C++ |
| `eigen/csr_map` | Eigen | CSR | C++ |
| `psblas/csr` | PSBLAS | CSR | C++/Fortran |
| `psblas/csc` | PSBLAS | CSC | C++/Fortran |
| `mkl/csr_ie` | Intel MKL | CSR IE | C |
| `mkl/csc_ie` | Intel MKL | CSC IE | C |

### Lanczos

Both Lanczos benches use the same matrix suite and the same Krylov dimension for a given matrix. The Krylov dimension is selected adaptively with the Saad a posteriori error estimate.

| Benchmark ID prefix | Kernels | Library | Format | Language |
|---------------------|---------|---------|--------|----------|
| `faer_csc` | one-pass, two-pass | faer | CSC | Rust |
| `faer_csr` | one-pass, two-pass | faer | CSR | Rust |
| `eigen_csr` | one-pass, two-pass | Eigen | CSR | C++ |
| `eigen_csc` | one-pass, two-pass | Eigen | CSC | C++ |
| `petsc_csr` | one-pass, two-pass | PETSc | CSR | C |
| `psblas_csr` | one-pass, two-pass | PSBLAS | CSR | C++/Fortran |
| `psblas_csc` | one-pass, two-pass | PSBLAS | CSC | C++/Fortran |

### Matrix suite

Fifteen symmetric matrices with small or zero mean diagonal, so the Saad estimator on `exp(-A)b` is meaningful at every Krylov dimension.

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
cargo test --release

export RUSTFLAGS="-C target-cpu=native"
export OMP_NUM_THREADS=1
taskset -c 0 cargo bench --bench spmv
taskset -c 0 cargo bench --bench lanczos
taskset -c 0 cargo bench --bench lanczos_two_pass
```

## Plotting

Criterion writes benchmark artifacts under `target/criterion/`. These files are _not_ committed. With Criterion's `csv_output` feature, the repository stores the data used for the figures as CSV files under `python/data/`:

- `raw_samples.csv`: Criterion raw samples collected from all `target/criterion/**/new/raw.csv` files.
- `summary.csv`: one row per `(benchmark, matrix, configuration)`, with execution time estimated by an ordinary least-squares fit through the origin on `(iteration_count, sample_time)` and throughput derived from that estimate.
- `lanczos_accuracy.csv`: relative L2 output error for Lanczos backends, measured against the `faer/CSC` reference and generated outside Criterion.

The plotting script reads only these CSV files and does not parse Criterion JSON artifacts.

Install the plotting dependencies if they are not already available:

```bash
python3 -m pip install -r python/requirements.txt
```

After running Criterion, generate `raw_samples.csv` and `summary.csv` with:

```bash
python3 python/plot.py export-csv
```

SpMV roofline requires STREAM Triad bandwidth. Measure it before running
`plot.py spmv` or `plot.py all`:

```bash
bash stream_bench.sh   # writes python/hw_config.json
```

To regenerate figures from committed CSV data:

```bash
python3 python/plot.py spmv              # per-matrix SpMV bar charts + roofline
python3 python/plot.py lanczos_one_pass  # one-pass Lanczos bar charts
python3 python/plot.py lanczos_two_pass  # two-pass Lanczos bar charts
python3 python/plot.py perfprof          # performance profiles
python3 python/plot.py violin            # normalized-throughput violin plots
python3 python/plot.py accuracy          # Lanczos output-agreement plots
python3 python/plot.py all               # performance figures above
```

The Lanczos accuracy plots report `-log10` of the relative L2 output error against `faer/CSC`. Larger values indicate closer numerical agreement.

To regenerate the Lanczos accuracy CSV and plots:

```bash
cargo run --release --bin lanczos_accuracy -- --output python/data/lanczos_accuracy.csv
python3 python/plot.py accuracy
```

To reproduce the benchmark data from scratch:

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

python3 python/plot.py export-csv
cargo run --release --bin lanczos_accuracy -- --output python/data/lanczos_accuracy.csv
python3 python/plot.py all
python3 python/plot.py accuracy
```

## Hardware

All benchmarks run pinned to a single core on a quad-socket Intel Xeon Gold 6418H (Sapphire Rapids).

| | |
|---|---|
| CPU | Intel Xeon Gold 6418H @ 4.0 GHz (24 cores, 2 threads per core) |
| Microarchitecture | Sapphire Rapids |
| Memory | 2 TiB |
| L1d / L2 / L3 | 48 KB / 2 MB / 60 MB per socket |
| ISA | AVX-512F/BW/VL/VNNI/BF16/FP16, AMX |

## Compiler flags

| Component | Compiler | Flags |
|-----------|----------|-------|
| C/C++ FFI wrappers | clang/clang++ | `-O3 -march=native -mtune=native -flto` |
| Fortran PSBLAS wrappers | gfortran | `-O3 -march=native -mtune=native -ffat-lto-objects` |
| Rust (faer + harness) | rustc (LLVM) | `opt-level=3, lto="fat", codegen-units=1, panic="abort"` |
| Eigen Lanczos wrappers | clang++ C++20 | `-O3 -march=native -mtune=native -flto` |

All backends receive the same CSR and CSC arrays allocated once by Rust. PETSc and Eigen wrap those arrays without copying. MKL may build optimized internal copies. PSBLAS assembles its own sparse matrix. Setup cost is excluded from the timed loop.
