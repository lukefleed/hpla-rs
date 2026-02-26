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
3. Load the necessary environments:
   ```bash
   source ~/.cargo/env
   source ~/spack/share/spack/setup-env.sh
   ```
4. Execute the benchmarking suite:
   ```bash
   cargo bench
   ```

Criterion will automatically measure cycle-accurate timings and generate plots/reports in the `target/criterion/` directory.
