//! Criterion harness for the two-pass Lanczos kernel computing `exp(-A)b`.
//!
//! Iterates over symmetric `.mtx` matrices in `matrices/`, determines the
//! Krylov dimension via the a posteriori residual estimator, and benchmarks
//! all backends under identical conditions (same matrix, same starting vector
//! `b`, same number of Lanczos iterations).

#[path = "lanczos_common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::sparse::{SparseColMat, SparseRowMat};
use hpla_rs::eigen::{
    libeigen_csc_lanczos_two_pass_execute, libeigen_csc_lanczos_two_pass_setup,
    libeigen_csc_lanczos_two_pass_teardown, libeigen_lanczos_two_pass_execute,
    libeigen_lanczos_two_pass_setup, libeigen_lanczos_two_pass_teardown,
};
use hpla_rs::lanczos::{
    ProjectedTridiagonalWorkspace, TwoPassWorkspace, estimate_spectral_radius,
    lanczos_two_pass_into,
};
use hpla_rs::petsc::{
    libpetsc_lanczos_two_pass_execute, libpetsc_lanczos_two_pass_setup,
    libpetsc_lanczos_two_pass_teardown,
};
use hpla_rs::psblas::{
    libpsblas_csc_lanczos_two_pass_execute, libpsblas_csc_lanczos_two_pass_setup,
    libpsblas_csc_lanczos_two_pass_teardown, libpsblas_csr_lanczos_two_pass_execute,
    libpsblas_csr_lanczos_two_pass_setup, libpsblas_csr_lanczos_two_pass_teardown,
};
use hpla_rs::{load_mtx_raw, scale_values};

use common::{lanczos_matrices, probe_krylov_dim};
use hpla_rs::lanczos::deterministic_rhs;

fn bench_lanczos_two_pass(c: &mut Criterion) {
    for (name, path) in lanczos_matrices() {
        let mut raw = load_mtx_raw(&path).expect("Failed to load matrix");

        // Estimate rho(A) via a short Lanczos probe, then absorb tau = -1/rho
        // into the matrix: the kernel then computes exp(-A')v on A' = A/rho,
        // whose spectrum sits in [-1, 1], so the a posteriori estimator is
        // meaningful at the target tolerance.
        let b_vec = deterministic_rhs(raw.nrows);
        let scale = {
            let a_tmp =
                SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
            let b_tmp = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);
            let scratch_req = a_tmp.as_ref().apply_scratch(1, Par::Seq);
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(
                &a_tmp.as_ref(),
                b_tmp.as_ref(),
                common::SPECTRAL_PROBE_STEPS,
                Par::Seq,
                stack,
            )
            .expect("spectral radius probe failed")
        };
        scale_values(&mut raw, scale);

        let a_faer =
            SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let scratch_req = a_faer.as_ref().apply_scratch(1, Par::Seq);

        let b_mat = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);

        let (krylov_dim, spectral_radius) = probe_krylov_dim(&a_faer, b_mat.as_ref());
        let max_k = ((spectral_radius.ceil() as usize) + common::KRYLOV_MARGIN)
            .min(common::KRYLOV_HARD_LIMIT);
        let converged = krylov_dim < max_k;
        eprintln!(
            "{name}: nrows={}, nnz={}, rho(A)~{scale:.3e}, rho(A/s)~{spectral_radius:.3}, max_k={max_k}, m={krylov_dim}{}",
            raw.nrows,
            raw.nnz,
            if converged {
                ""
            } else {
                " [WARNING: Saad tolerance not reached]"
            }
        );

        let mut group = c.benchmark_group(format!("lanczos_two_pass_{name}"));

        // Two-pass Lanczos FLOP count, derived from the implementation
        // (src/lanczos/algorithms/lanczos_two_pass.rs):
        //
        // Pass 1 (k steps), per step:
        //   SpMV:               2*nnz   (1 mul + 1 add per nonzero)
        //   w -= beta*v_prev:   2n
        //   alpha = dot(v,w):   2n
        //   w -= alpha*v_curr:  2n
        //   ||w||_2:            2n
        //   w *= 1/beta:        n
        //   Step total:         2*nnz + 9n
        //
        // Pass 2 (k-1 steps), per step:
        //   SpMV:               2*nnz
        //   w -= beta*v_prev:   2n
        //   w -= alpha*v_curr:  2n
        //   w *= 1/beta:        n
        //   x += coeff*v:       2n
        //   Step total:         2*nnz + 7n
        //
        // Total: k*(2*nnz + 9n) + (k-1)*(2*nnz + 7n)
        //      = (4k-2)*nnz + (16k-7)*n
        //      ≈ 4k*(nnz + 4n)   for k >> 1 (lower-order terms dropped)
        group.throughput(Throughput::Elements(
            4 * krylov_dim as u64 * (raw.nnz as u64 + 4 * raw.nrows as u64),
        ));

        // --------------------------------------------------------
        // faer (two-pass Lanczos)
        // --------------------------------------------------------
        // Workspace is built once per matrix, outside `bench.iter`, so
        // the timing window measures only the kernel: no rolling-vector
        // allocation per iteration. Matches the SpMV steady-state
        // accumulation pattern and the PSBLAS `_setup`/`_execute`
        // contract.
        let mut ws = TwoPassWorkspace::new(raw.nrows, krylov_dim);
        let mut projected = ProjectedTridiagonalWorkspace::new(krylov_dim, Par::Seq);
        let mut mem = MemBuffer::new(scratch_req);
        group.bench_with_input(BenchmarkId::new("faer_csc", "two_pass"), &(), |bench, _| {
            bench.iter(|| {
                let stack = MemStack::new(&mut mem);
                let result = lanczos_two_pass_into(
                    &mut ws,
                    &a_faer.as_ref(),
                    b_mat.as_ref(),
                    krylov_dim,
                    Par::Seq,
                    stack,
                    |alphas, betas, out| projected.exp_neg_tk(alphas, betas, out),
                );
                let _ = criterion::black_box(result);
            });
        });

        // --------------------------------------------------------
        // faer CSR (two-pass Lanczos, cross-format control)
        // --------------------------------------------------------
        let a_faer_csr =
            SparseRowMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let scratch_req_csr = a_faer_csr.as_ref().apply_scratch(1, Par::Seq);
        let mut ws_csr = TwoPassWorkspace::new(raw.nrows, krylov_dim);
        let mut projected_csr = ProjectedTridiagonalWorkspace::new(krylov_dim, Par::Seq);
        let mut mem_csr = MemBuffer::new(scratch_req_csr);
        group.bench_with_input(BenchmarkId::new("faer_csr", "two_pass"), &(), |bench, _| {
            bench.iter(|| {
                let stack = MemStack::new(&mut mem_csr);
                let result = lanczos_two_pass_into(
                    &mut ws_csr,
                    &a_faer_csr.as_ref(),
                    b_mat.as_ref(),
                    krylov_dim,
                    Par::Seq,
                    stack,
                    |alphas, betas, out| projected_csr.exp_neg_tk(alphas, betas, out),
                );
                let _ = criterion::black_box(result);
            });
        });

        // --------------------------------------------------------
        // Eigen CSR (two-pass Lanczos)
        // --------------------------------------------------------
        unsafe {
            let ctx = libeigen_lanczos_two_pass_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
                b_vec.as_ptr(),
                krylov_dim as i32,
            );

            if !ctx.is_null() {
                group.bench_with_input(
                    BenchmarkId::new("eigen_csr", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libeigen_lanczos_two_pass_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libeigen_lanczos_two_pass_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // Eigen CSC (two-pass Lanczos, cross-format control)
        // --------------------------------------------------------
        unsafe {
            let ctx = libeigen_csc_lanczos_two_pass_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.col_ptr.as_ptr(),
                raw.row_idx.as_ptr(),
                raw.csc_values.as_ptr(),
                b_vec.as_ptr(),
                krylov_dim as i32,
            );

            if !ctx.is_null() {
                group.bench_with_input(
                    BenchmarkId::new("eigen_csc", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libeigen_csc_lanczos_two_pass_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libeigen_csc_lanczos_two_pass_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PETSc CSR (two-pass Lanczos for exp(-A)b)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpetsc_lanczos_two_pass_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
                b_vec.as_ptr(),
                krylov_dim as i32,
            );

            if !ctx.is_null() {
                group.bench_with_input(
                    BenchmarkId::new("petsc_csr", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpetsc_lanczos_two_pass_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpetsc_lanczos_two_pass_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PSBLAS CSR (two-pass Lanczos)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpsblas_csr_lanczos_two_pass_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.row_ptr.as_ptr(),
                raw.col_idx.as_ptr(),
                raw.values.as_ptr(),
                b_vec.as_ptr(),
                krylov_dim as i32,
            );

            if !ctx.is_null() {
                group.bench_with_input(
                    BenchmarkId::new("psblas_csr", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpsblas_csr_lanczos_two_pass_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpsblas_csr_lanczos_two_pass_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PSBLAS CSC (two-pass Lanczos)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpsblas_csc_lanczos_two_pass_setup(
                raw.nrows as i32,
                raw.ncols as i32,
                raw.nnz as i32,
                raw.col_ptr.as_ptr(),
                raw.row_idx.as_ptr(),
                raw.csc_values.as_ptr(),
                b_vec.as_ptr(),
                krylov_dim as i32,
            );

            if !ctx.is_null() {
                group.bench_with_input(
                    BenchmarkId::new("psblas_csc", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpsblas_csc_lanczos_two_pass_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpsblas_csc_lanczos_two_pass_teardown(ctx);
            }
        }

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(5))
        .measurement_time(std::time::Duration::from_secs(100));
    targets = bench_lanczos_two_pass
);
criterion_main!(benches);
