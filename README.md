# SpMV Benchmark 

## How to Run

1. Ensure PETSc is installed via Spack.

   ```bash
   spack install petsc \~mpi \~hdf5 \~superlu-dist \~mumps \~suite-sparse \~scalapack \~strumpack \~ptscotch \~hwloc \~X cflags='-g -O3 -march=native -mtune=native -flto' cxxflags='-g -O3 -march=native -mtune=native -flto' fflags='-g -O3 -march=native -mtune=native -flto -ffree-line-length-none'
   ```   

2. Ensure Eigen is installed via Spack

   ```bash
   spack install eigen
   ```

3. Ensure Intel MKL is installed via Spack

   ```bash
   spack install intel-oneapi-mkl
   ```

4. Download some matrices from the Sparse Matrix Collection. For example [`atmosmood.mtx`](https://sparse.tamu.edu/Bourchtein/atmosmodd) and [`thermal2.mtx`](https://sparse.tamu.edu/Schmid/thermal2)

   ```bash
   mkdir matrices
   cd matrices
   wget https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodd.tar.gz
   wget https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz
   tar -xvf atmosmodd.tar.gz
   tar -xvf thermal2.tar.gz
   ```   

5. Load the necessary environments and build the PSBLAS C/C++ wrappers:

   ```bash
   source ~/.cargo/env
   source ~/spack/share/spack/setup-env.sh
   spack load intel-oneapi-mkl
   spack load openmpi
   
   # Build PSBLAS locally using Clang with LTO and Position Independent Code
   bash build_psblas.sh
   ```

6. Execute the benchmarking suite in a strictly isolated, single-threaded SOTA environment:

   ```bash
   cargo check --all-targets --all-features
   taskset -c 0 cargo bench 
   ```

Criterion will automatically measure cycle-accurate timings and generate plots/reports in the `target/criterion/` directory. You can also plot with

```bash
cd python
python3 plot.py
```

## Benchmarks Configuration

The `cargo bench` suite compares different libraries performing the Sparse General Matrix-Vector (SpGEMV) multiplication: `y += A * x`. All external backends are compiled via Clang/LLVM with `-O3 -march=native -ffast-math -flto` to ensure maximal SIMD autovectorization and operate via a Zero-Copy memory interface.

The harness evaluates the following configurations:

* **`faer/csc`**: Native pure-Rust implementation using [Faer](https://github.com/sarah-quinones/faer-rs). It iterates directly over a standard Compressed Sparse Column (CSC) memory layout using safe Rust accumulators (`y.write(...)`).
* **`petsc/csr_inodes`**: C FFI bindings to [PETSc](https://petsc.org/). It uses a Compressed Sparse Row (CSR) layout with the **Inode** kernel optimization enabled (`MatCreateSeqAIJWithArrays`). This dynamically scans the matrix structure at runtime to find identical non-zero patterns across adjacent rows, grouping them to apply aggressive compiler unrolling and SIMD vectorization.
* **`petsc/csr_raw`**: C FFI bindings to PETSc. It operates on the same CSR layout but explicitly forces the Inode routine off (scalar `MatMultAdd`). This is useful to measure the precise impact of the Inode heuristic vs the baseline C loop.
* **`eigen/csc_map`**: C++ FFI bindings to [Eigen](https://eigen.tuxfamily.org/). Uses an `Eigen::Map` to project the CSC memory buffer into an `Eigen::SparseMatrix` without copies. The multiplication is evaluated purely via C++ Template Expressions (`*y += (*A) * (*x);`).
* **`mkl/csr_ie`**: C FFI bindings to [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) Sparse BLAS. Uses the  Inspection-Execution API (`mkl_sparse_d_create_csr`, `mkl_sparse_optimize`) over a CSR layout. This routine heuristically inspects the matrix topology ahead of time, rearranging its representation internally to map perfectly onto the CPU's L1/L2 caches and AVX-512 FMA registers before invoking `mkl_sparse_d_mv`.
* **`psblas/csr`**: C++ FFI bindings to [PSBLAS](https://github.com/sfilippone/psblas3) (Parallel Sparse BLAS). This framework handles both serial and parallel environments, internally executing OpenMPI. To preserve L3 Cache coherency on a single-node run, CPU affinity is programmatically locked via `sched_setaffinity` overriding MPI setup daemons, ensuring an exact cycle-accurate comparison with serial libraries.
