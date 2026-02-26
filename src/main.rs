use faer::col::Col;
use faer::sparse::SparseColMat;
use spmv_bench::{load_mtx_raw, spmv_faer};

fn main() {
    println!("Spmv-Bench unified benchmarking suite.");
    println!("Run `cargo bench` to execute Criterion benchmarks comparing Faer vs PETSc.");
}
