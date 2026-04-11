//! Shared helpers for the `lanczos` and `lanczos_two_pass` Criterion harnesses.
//! Pulled in by each bench file via `#[path = "lanczos_common/mod.rs"] mod common;`.

use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::prelude::MatRef;
use faer::sparse::SparseColMat;
use hpla_rs::LANCZOS_SUITE;
use hpla_rs::lanczos::estimate_spectral_radius;
pub use hpla_rs::lanczos::{KRYLOV_HARD_LIMIT, KRYLOV_MARGIN, SAAD_TOL, SPECTRAL_PROBE_STEPS};
use std::path::PathBuf;

/// Iterates [`hpla_rs::LANCZOS_SUITE`] and returns `(name, path)` for every
/// matrix that is present on disk under `matrices/`. Missing ones are logged
/// and skipped so a partially-downloaded suite still runs.
pub fn lanczos_matrices() -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    for name in LANCZOS_SUITE {
        let path = PathBuf::from(format!("matrices/{name}.mtx"));
        if path.exists() {
            out.push((*name, path));
        } else {
            eprintln!(
                "  warning: {name}.mtx not available; skipping. \
                 Run `bash download_matrices.sh` from the repo root."
            );
        }
    }
    out
}

/// Runs the spectral radius probe (short Lanczos on Ritz values of `T_k`)
/// followed by the Saad a posteriori error estimate to pick the minimum
/// Krylov dimension `m` such that `err < SAAD_TOL`.
///
/// The caller owns `b_mat`; the helper never allocates a fresh starting
/// vector, so the Saad estimate and the vector later passed to the
/// benchmark backends are guaranteed to match.
///
/// Returns `(m, rho)` where `rho` is the estimated spectral radius.
pub fn probe_krylov_dim(
    a_faer: &SparseColMat<u32, f64>,
    b_mat: MatRef<'_, f64>,
) -> (usize, f64) {
    let scratch_req = a_faer.as_ref().apply_scratch(1, Par::Seq);

    let rho = {
        let mut mem = MemBuffer::new(scratch_req);
        let stack = MemStack::new(&mut mem);
        estimate_spectral_radius(
            &a_faer.as_ref(),
            b_mat,
            SPECTRAL_PROBE_STEPS,
            Par::Seq,
            stack,
        )
        .unwrap_or(100.0)
    };

    let max_k = ((rho.ceil() as usize) + KRYLOV_MARGIN).min(KRYLOV_HARD_LIMIT);

    let m = {
        let mut mem = MemBuffer::new(scratch_req);
        let stack = MemStack::new(&mut mem);
        let (m, _decomp) = hpla_rs::lanczos::adaptive_krylov_dim(
            &a_faer.as_ref(),
            b_mat,
            max_k,
            SAAD_TOL,
            Par::Seq,
            stack,
        )
        .expect("Lanczos probe failed");
        m.max(1)
    };

    (m, rho)
}
