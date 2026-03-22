//! Criterion benchmarking harness for single-threaded SpMV (y += A*x).
//!
//! Iterates over all `.mtx` matrices in the `matrices/` directory,
//! sets up the exact same raw memory structures, and benchmarks all
//! backends (Faer, PETSc, Eigen, MKL, PSBLAS) under identical conditions
//! for cycle-accurate performance comparisons.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::col::Col;
use faer::sparse::SparseColMat;
use hpla_rs::eigen::{
    libeigen_csr_spmv_execute, libeigen_csr_spmv_setup, libeigen_csr_spmv_teardown,
    libeigen_spmv_execute, libeigen_spmv_setup, libeigen_spmv_teardown,
};
use hpla_rs::mkl::{
    libmkl_csc_spmv_execute, libmkl_csc_spmv_setup, libmkl_csc_spmv_teardown, libmkl_spmv_execute,
    libmkl_spmv_setup, libmkl_spmv_teardown,
};
use hpla_rs::petsc::{libpetsc_spmv_execute, libpetsc_spmv_setup, libpetsc_spmv_teardown};
use hpla_rs::psblas::{libpsblas_spmv_execute, libpsblas_spmv_setup, libpsblas_spmv_teardown};
use hpla_rs::{load_mtx_raw, spmv_faer};
use std::fs;
use std::path::PathBuf;

/// Discovers Matrix Market `.mtx` files available for benchmarking.
fn get_matrices() -> Vec<PathBuf> {
    let dir = PathBuf::from("matrices");
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

    // y is not reset between iterations — avoids injecting a memcpy.
    // SpMV is memory-bound; y values don't affect cost.
    for path in matrices {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let raw = load_mtx_raw(&path).expect("Failed to load matrix");

        let mut group = c.benchmark_group(format!("spmv_{}", name));

        // Setup Throughput for memory bandwidth or flop/s representation
        // For SpMV (y += A*x): 2*NNZ ops
        // Bandwidth: (rows+1)*4 + nnz*4 + nnz*8 + cols*8 + rows*16 (read y + write y)
        // let bytes =
        //     ((raw.nrows + 1) * 4 + raw.nnz * 4 + raw.nnz * 8 + raw.ncols * 8 + raw.nrows * 16)
        //         as u64;
        // group.throughput(Throughput::Bytes(bytes));

        // Setup Throughput for computational performance (GFLOP/s)
        // For SpMV (y += A*x), we perform one multiply and one add per non-zero: 2*NNZ FLOPs.
        group.throughput(Throughput::Elements(2 * raw.nnz as u64));

        // ----------------------------------------------------
        // Faer
        // ----------------------------------------------------
        let a_faer =
            SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let x_faer: Col<f64> = Col::from_fn(raw.ncols, |_| 1.0);
        let y_init_faer: Col<f64> = Col::zeros(raw.nrows);
        let mut y_faer = y_init_faer.clone();

        // NOTE: black_box applied uniformly. With fat LTO (clang -flto +
        // cargo lto="fat"), LLVM sees across the FFI boundary. Dynamically-linked
        // library calls (PETSc, MKL) act as the real DCE barriers, but black_box
        // is a zero-cost compiler fence that removes any ambiguity.
        group.bench_with_input(BenchmarkId::new("faer", "csc"), &(), |b, _| {
            b.iter(|| {
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
                    criterion::black_box(ctx);
                });
            });

            libpetsc_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // PETSc
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
                    criterion::black_box(ctx);
                });
            });

            libpetsc_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // Intel MKL
        // ----------------------------------------------------
        unsafe {
            let ctx = libmkl_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
            );

            group.bench_with_input(BenchmarkId::new("mkl", "csr_ie"), &(), |b, _| {
                b.iter(|| {
                    libmkl_spmv_execute(ctx);
                    criterion::black_box(ctx);
                });
            });

            libmkl_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // Intel MKL CSC
        // ----------------------------------------------------
        unsafe {
            let ctx = libmkl_csc_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.col_ptr.as_ptr(),
                raw.row_idx.as_ptr(),
                raw.csc_values.as_ptr(),
            );

            group.bench_with_input(BenchmarkId::new("mkl", "csc_ie"), &(), |b, _| {
                b.iter(|| {
                    libmkl_csc_spmv_execute(ctx);
                    criterion::black_box(ctx);
                });
            });

            libmkl_csc_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // Eigen
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
                    criterion::black_box(ctx);
                });
            });

            libeigen_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // Eigen (C++ CSR) — cross-format control
        // ----------------------------------------------------
        unsafe {
            let ctx = libeigen_csr_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
            );

            group.bench_with_input(BenchmarkId::new("eigen", "csr_map"), &(), |b, _| {
                b.iter(|| {
                    libeigen_csr_spmv_execute(ctx);
                    criterion::black_box(ctx);
                });
            });

            libeigen_csr_spmv_teardown(ctx);
        }

        // ----------------------------------------------------
        // PSBLAS
        // ----------------------------------------------------
        unsafe {
            let ctx = libpsblas_spmv_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
            );

            group.bench_with_input(BenchmarkId::new("psblas", "csr"), &(), |b, _| {
                b.iter(|| {
                    libpsblas_spmv_execute(ctx);
                    criterion::black_box(ctx);
                });
            });

            libpsblas_spmv_teardown(ctx);
        }

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(8));
    targets = bench_spmv
);
criterion_main!(benches);
