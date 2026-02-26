# SpMV Benchmark 

## File Structure

- `Cargo.toml`: The Rust package configuration. It specifies `criterion` for statistical benchmarking and sets aggressive compiler optimization profiles (LTO, codegen-units=1) to ensure state-of-the-art performance.
- `build.rs`: The Rust build script. It uses `spack` to automatically locate your PETSc installation, injects optimal C compiler flags (`-O3 -march=native -ffast-math`), and compiles the C wrapper to link it into the Rust binary.
- `petsc_wrapper.c`: A minimal C interface. It takes raw CSR arrays allocated by Rust and uses PETSc's `MatCreateSeqAIJWithArrays` and `MatMultAdd` to perform the matrix-vector products without copying the data.
- `src/petsc.rs`: The Rust FFI bindings. It defines the `extern "C"` interfaces used to call the C functions from `petsc_wrapper.c`.
- `src/lib.rs`: The core logic. It handles parsing Matrix Market (`.mtx`) files from disk, converting them into flat CSR arrays, and exposes the Faer SpMV kernel.
- `benches/spmv.rs`: The Criterion benchmark harness. It iterates over the loaded matrices, executes both Faer and PETSc side-by-side, times the pure multiplication loops, and calculates the memory bandwidth throughput (GB/s).

## How to Run

1. Ensure PETSc is installed via Spack.
2. Load the necessary environments:
   ```bash
   source ~/.cargo/env
   source ~/spack/share/spack/setup-env.sh
   ```
3. Execute the benchmarking suite:
   ```bash
   cargo bench
   ```

Criterion will automatically measure cycle-accurate timings and generate plots/reports in the `target/criterion/` directory.
