//! Criterion harness for the one-pass Lanczos variant of `exp(-A)b`.
//!
//! Computes `y ~= exp(-A)b` by building the full Lanczos basis `V_m` in
//! memory, solving the projected problem `g = exp(-T_m)*e_1` and returning
//! `y = ||b|| * V_m * g`. Memory footprint is O(n*m).
//!
//! Pairs with `benches/lanczos_two_pass.rs`, which computes the same output
//! vector with memory O(n) at the cost of a second Lanczos pass. Both benches
//! share the Krylov dimension picked by the a posteriori residual estimator,
//! so any throughput difference isolates the memory/compute trade-off between
//! the two variants.

#[path = "lanczos_common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::sparse::{SparseColMat, SparseRowMat};
use hpla_rs::eigen::{
    libeigen_csc_lanczos_execute, libeigen_csc_lanczos_setup, libeigen_csc_lanczos_teardown,
    libeigen_lanczos_execute, libeigen_lanczos_setup, libeigen_lanczos_teardown,
};
use hpla_rs::lanczos::{
    LanczosWorkspace, ProjectedTridiagonalWorkspace, Reorthogonalization, estimate_spectral_radius,
    lanczos_into,
};
use hpla_rs::petsc::{libpetsc_lanczos_execute, libpetsc_lanczos_setup, libpetsc_lanczos_teardown};
use hpla_rs::psblas::{
    libpsblas_csc_lanczos_execute, libpsblas_csc_lanczos_setup, libpsblas_csc_lanczos_teardown,
    libpsblas_csr_lanczos_execute, libpsblas_csr_lanczos_setup, libpsblas_csr_lanczos_teardown,
};
use hpla_rs::{load_mtx_raw, scale_values};

use common::{lanczos_matrices, probe_krylov_dim};
use hpla_rs::lanczos::deterministic_rhs;

fn bench_lanczos(c: &mut Criterion) {
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

        let mut group = c.benchmark_group(format!("lanczos_{name}"));

        // One-pass Lanczos FLOP count for f(A)b.
        //
        // Pass 1 (m steps), per step, from
        // src/lanczos/algorithms/mod.rs::lanczos_recurrence_step:
        //   SpMV:                                          2*nnz
        //   fused w -= beta*v_prev + alpha = v^T w:        4n
        //   w -= alpha*v_curr:                             2n
        //   ||w||_2:                                       2n
        //   w *= 1/beta (normalize) + copy into V_m col:   n
        //   Step total:                                    2*nnz + 9n
        //
        // Post-recurrence: V_m * g final accumulation (gemv, column-major):
        //   2*m*n
        //
        // Scaling by ||b||: folded into the gemv alpha, 0 extra FLOPs.
        // Projected problem exp(-T_m)*e_1: O(m^3), negligible for m << n.
        //
        // Total: m*(2*nnz + 9n) + 2*m*n = m*(2*nnz + 11n)
        group.throughput(Throughput::Elements(
            krylov_dim as u64 * (2 * raw.nnz as u64 + 11 * raw.nrows as u64),
        ));

        // --------------------------------------------------------
        // faer (one-pass Lanczos for exp(-A)b)
        // --------------------------------------------------------
        // Workspace is built once per matrix, outside `bench.iter`, so
        // the timing window measures only the kernel: no `V_k`/scratch
        // allocation or zero-fill per iteration. Matches the SpMV
        // the PSBLAS `_setup`/`_execute` contract.
        let mut ws = LanczosWorkspace::new(raw.nrows, krylov_dim);
        let mut projected = ProjectedTridiagonalWorkspace::new(krylov_dim, Par::Seq);
        let mut mem = MemBuffer::new(scratch_req);
        group.bench_with_input(BenchmarkId::new("faer_csc", "one_pass"), &(), |bench, _| {
            bench.iter(|| {
                let stack = MemStack::new(&mut mem);
                let result = lanczos_into(
                    &mut ws,
                    &a_faer.as_ref(),
                    b_mat.as_ref(),
                    krylov_dim,
                    Par::Seq,
                    Reorthogonalization::None,
                    stack,
                    |alphas, betas, out| projected.exp_neg_tk(alphas, betas, out),
                );
                let _ = criterion::black_box(result);
            });
        });

        // --------------------------------------------------------
        // faer CSR (one-pass Lanczos, cross-format control)
        // --------------------------------------------------------
        let a_faer_csr =
            SparseRowMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let scratch_req_csr = a_faer_csr.as_ref().apply_scratch(1, Par::Seq);
        let mut ws_csr = LanczosWorkspace::new(raw.nrows, krylov_dim);
        let mut projected_csr = ProjectedTridiagonalWorkspace::new(krylov_dim, Par::Seq);
        let mut mem_csr = MemBuffer::new(scratch_req_csr);
        group.bench_with_input(BenchmarkId::new("faer_csr", "one_pass"), &(), |bench, _| {
            bench.iter(|| {
                let stack = MemStack::new(&mut mem_csr);
                let result = lanczos_into(
                    &mut ws_csr,
                    &a_faer_csr.as_ref(),
                    b_mat.as_ref(),
                    krylov_dim,
                    Par::Seq,
                    Reorthogonalization::None,
                    stack,
                    |alphas, betas, out| projected_csr.exp_neg_tk(alphas, betas, out),
                );
                let _ = criterion::black_box(result);
            });
        });

        // --------------------------------------------------------
        // Eigen CSR (one-pass Lanczos)
        // --------------------------------------------------------
        unsafe {
            let ctx = libeigen_lanczos_setup(
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
                    BenchmarkId::new("eigen_csr", "one_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libeigen_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libeigen_lanczos_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // Eigen CSC (one-pass Lanczos, cross-format control)
        // --------------------------------------------------------
        unsafe {
            let ctx = libeigen_csc_lanczos_setup(
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
                    BenchmarkId::new("eigen_csc", "one_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libeigen_csc_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libeigen_csc_lanczos_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PSBLAS CSR (one-pass Lanczos for exp(-A)b)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpsblas_csr_lanczos_setup(
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
                    BenchmarkId::new("psblas_csr", "one_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpsblas_csr_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpsblas_csr_lanczos_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PSBLAS CSC (one-pass Lanczos for exp(-A)b)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpsblas_csc_lanczos_setup(
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
                    BenchmarkId::new("psblas_csc", "one_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpsblas_csc_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpsblas_csc_lanczos_teardown(ctx);
            }
        }

        // --------------------------------------------------------
        // PETSc CSR (one-pass Lanczos for exp(-A)b)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpetsc_lanczos_setup(
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
                    BenchmarkId::new("petsc_csr", "one_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpetsc_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpetsc_lanczos_teardown(ctx);
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
    targets = bench_lanczos
);
criterion_main!(benches);
