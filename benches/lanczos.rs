//! Criterion benchmarking harness for two-pass Lanczos computing exp(-A)b.
//!
//! Iterates over symmetric `.mtx` matrices in `matrices/`, determines the
//! Krylov dimension via the Saad (1992) a posteriori error estimate, and
//! benchmarks all backends under identical conditions (same matrix, same
//! starting vector b, same number of Lanczos iterations).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::sparse::SparseColMat;
use faer::Par;
use hpla_rs::lanczos::{
    determine_krylov_dim, estimate_spectral_radius, exp_neg_tk_solver, lanczos_two_pass,
};
use hpla_rs::psblas::{
    libpsblas_lanczos_execute, libpsblas_lanczos_setup, libpsblas_lanczos_teardown,
};
use hpla_rs::{detect_symmetry, load_mtx_raw};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::PathBuf;

/// Number of Lanczos steps for the spectral radius probe (Ritz values of T_k).
const SPECTRAL_PROBE_STEPS: usize = 20;

/// Tolerance for the Saad a posteriori error estimate.
const SAAD_TOL: f64 = 1e-10;

/// Safety margin added to the spectral radius estimate when computing max_k.
const KRYLOV_MARGIN: usize = 50;

/// Absolute upper bound on the Krylov subspace dimension.
const KRYLOV_HARD_LIMIT: usize = 500;

/// Discovers symmetric Matrix Market files for Lanczos benchmarking.
fn get_symmetric_matrices() -> Vec<PathBuf> {
    let dir = PathBuf::from("matrices");
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "mtx") {
                let (is_sym, _) = detect_symmetry(&path);
                if is_sym {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

/// Generates a deterministic starting vector shared across all backends.
fn deterministic_b(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random::<f64>()).collect()
}

fn bench_lanczos(c: &mut Criterion) {
    let matrices = get_symmetric_matrices();

    for path in matrices {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let raw = load_mtx_raw(&path).expect("Failed to load matrix");

        let b_vec = deterministic_b(raw.nrows);

        // Build faer sparse matrix (CSC, used as LinOp)
        let a_faer =
            SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets).unwrap();
        let scratch_req = a_faer.as_ref().apply_scratch(1, Par::Seq);

        // Determine Krylov dimension via Saad error estimate (probe phase, not timed)
        let b_mat = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);

        // Estimate spectral radius via short Lanczos probe (Ritz values of T_20).
        let spectral_radius = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(&a_faer.as_ref(), b_mat.as_ref(), SPECTRAL_PROBE_STEPS, Par::Seq, stack)
                .unwrap_or(100.0) // fallback if probe fails
        };

        let max_k = ((spectral_radius.ceil() as usize) + KRYLOV_MARGIN).min(KRYLOV_HARD_LIMIT);

        let krylov_dim = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            let (m, _decomp) =
                determine_krylov_dim(&a_faer.as_ref(), b_mat.as_ref(), max_k, SAAD_TOL, Par::Seq, stack)
                    .expect("Lanczos probe failed");
            m.max(1)
        };

        let converged = krylov_dim < max_k;
        eprintln!(
            "{name}: nrows={}, nnz={}, rho~{spectral_radius:.1}, max_k={max_k}, m={krylov_dim}{}",
            raw.nrows, raw.nnz,
            if converged { "" } else { " [WARNING: Saad tolerance not reached]" }
        );

        let mut group = c.benchmark_group(format!("lanczos_{name}"));

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
        group.bench_with_input(BenchmarkId::new("faer", "two_pass"), &(), |bench, _| {
            let mut mem = MemBuffer::new(scratch_req);
            bench.iter(|| {
                let stack = MemStack::new(&mut mem);
                let result = lanczos_two_pass(
                    &a_faer.as_ref(),
                    b_mat.as_ref(),
                    krylov_dim,
                    Par::Seq,
                    stack,
                    exp_neg_tk_solver,
                );
                let _ = criterion::black_box(result);
            });
        });

        // --------------------------------------------------------
        // PSBLAS (two-pass Lanczos)
        // --------------------------------------------------------
        unsafe {
            let ctx = libpsblas_lanczos_setup(
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
                    BenchmarkId::new("psblas", "two_pass"),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            libpsblas_lanczos_execute(ctx);
                            criterion::black_box(ctx);
                        });
                    },
                );

                libpsblas_lanczos_teardown(ctx);
            }
        }

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(30));
    targets = bench_lanczos
);
criterion_main!(benches);
