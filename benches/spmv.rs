//! Criterion benchmarking harness for SpMV.
//!
//! Iterates over all `.mtx` matrices in the `matrices/` directory,
//! sets up the exact same raw memory structures, and executes
//! Faer (CSC) and PETSc (CSR) back-to-back to guarantee perfectly
//! isolated, cycle-accurate performance comparisons.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use faer::col::Col;
use faer::sparse::SparseColMat;
use spmv_bench::eigen::{libeigen_spmv_execute, libeigen_spmv_setup, libeigen_spmv_teardown};
use spmv_bench::petsc::{libpetsc_spmv_execute, libpetsc_spmv_setup, libpetsc_spmv_teardown};
use spmv_bench::{load_mtx_raw, spmv_faer};
use std::fs;
use std::path::PathBuf;

/// Discovers Matrix Market `.mtx` files available for benchmarking.
fn get_matrices() -> Vec<PathBuf> {
    let dir = PathBuf::from("../matrices"); // Assumes we run from spmv-bench/
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "mtx") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Core Criterion benchmarking loop.
/// Sets up the benchmark group per matrix and defines custom memory throughputs.
fn bench_spmv(c: &mut Criterion) {
    let matrices = get_matrices();

    for path in matrices {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let raw = load_mtx_raw(&path).expect("Failed to load matrix");

        let mut group = c.benchmark_group(format!("spmv_{}", name));

        // Setup Throughput purely for memory bandwidth or flop/s representation
        // For SpMV: 2*NNZ ops
        // Bandwidth: (rows+1)*4 + nnz*4 + nnz*8 + cols*8 + rows*8
        let bytes =
            ((raw.nrows + 1) * 4 + raw.nnz * 4 + raw.nnz * 8 + raw.ncols * 8 + raw.nrows * 8)
                as u64;
        group.throughput(Throughput::Bytes(bytes));

        // ----------------------------------------------------
        // Faer
        // ----------------------------------------------------
        let a_faer =
            SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let x_faer: Col<f64> = Col::from_fn(raw.ncols, |_| 1.0);
        let y_init_faer: Col<f64> = Col::from_fn(raw.nrows, |i| (i as f64) * 1e-9);
        let mut y_faer = y_init_faer.clone();

        group.bench_with_input(BenchmarkId::new("faer", "csc"), &(), |b, _| {
            b.iter(|| {
                // We reload y in the bench loop to prevent accumulation issues,
                // but note that this slightly adds overhead if not cache-hot.
                // However, doing y = Ax + y modifies y each time.
                y_faer.copy_from(&y_init_faer);
                spmv_faer(&a_faer, &x_faer, &mut y_faer);
                criterion::black_box(&mut y_faer);
            });
        });

        // ----------------------------------------------------
        // PETSc (Inodes)
        // ----------------------------------------------------
        unsafe {
            let ctx = libpetsc_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
                0, // inodes enabled
            );

            group.bench_with_input(BenchmarkId::new("petsc", "csr_inodes"), &(), |b, _| {
                b.iter(|| {
                    libpetsc_spmv_execute(ctx);
                });
            });

            libpetsc_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // PETSc (Raw Scalar)
        // ----------------------------------------------------
        unsafe {
            let ctx = libpetsc_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
                1, // inodes disabled
            );

            group.bench_with_input(BenchmarkId::new("petsc", "csr_raw"), &(), |b, _| {
                b.iter(|| {
                    libpetsc_spmv_execute(ctx);
                });
            });

            libpetsc_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // Eigen (C++ CSC)
        // ----------------------------------------------------
        unsafe {
            let ctx = libeigen_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.col_ptr.as_ptr(),
                raw.row_idx.as_ptr(),
                raw.csc_values.as_ptr(),
            );

            group.bench_with_input(BenchmarkId::new("eigen", "csc_map"), &(), |b, _| {
                b.iter(|| {
                    libeigen_spmv_execute(ctx);
                });
            });

            libeigen_spmv_teardown(ctx);
        }

        group.finish();
    }
}

criterion_group!(benches, bench_spmv);
criterion_main!(benches);
