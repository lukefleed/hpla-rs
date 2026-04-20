//! Symmetric Lanczos for `f(A) b` with `f(z) = exp(-z)`. Hosts the high-level
//! one-pass and two-pass solvers, the projected-problem helper `exp_neg_tk`,
//! the a posteriori residual estimator, the adaptive Krylov-dimension driver,
//! and the spectral-radius probe used by the benches to absorb `tau = -1/rho`
//! into the matrix.

pub(crate) mod algorithms;
#[cfg(test)]
pub(crate) mod alloc_counter;
pub mod error;
mod projected;
pub mod solvers;

pub use algorithms::{LanczosDecomposition, Reorthogonalization, lanczos_scratch};
pub use projected::ProjectedTridiagonalWorkspace;
pub use solvers::{
    LanczosWorkspace, TwoPassWorkspace, lanczos, lanczos_into, lanczos_two_pass,
    lanczos_two_pass_into,
};

use faer::{Par, dyn_stack::MemStack, matrix_free::LinOp, prelude::*};

use crate::lanczos::algorithms::lanczos_two_pass::lanczos_pass_one_into;
use crate::lanczos::error::LanczosError;

/// Lanczos steps used by [`estimate_spectral_radius`] to probe rho(A).
pub const SPECTRAL_PROBE_STEPS: usize = 20;

/// Tolerance for the Saad a posteriori estimator used in [`adaptive_krylov_dim`].
pub const SAAD_TOL: f64 = 1e-10;

/// Extra steps added on top of `ceil(rho)` when sizing `max_k` for the
/// adaptive driver.
pub const KRYLOV_MARGIN: usize = 50;

/// Absolute upper bound on the Krylov subspace dimension.
pub const KRYLOV_HARD_LIMIT: usize = 500;

/// Deterministic zero-mean starting vector shared by benches and tests.
/// Components are uniform in `[-1, 1)` seeded from `StdRng::seed_from_u64(42)`.
/// Zero expected Rayleigh quotient on zero-diagonal adjacency matrices, so
/// the Saad estimator is meaningful at `m = 1` even on graph adjacencies.
pub fn deterministic_rhs(n: usize) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| 2.0 * rng.random::<f64>() - 1.0).collect()
}

/// Computes `exp(-T_k) e_1` for the tridiagonal `T_k` defined by `alphas`
/// (diagonal) and `betas` (off-diagonal).
///
/// Allocates a fresh projected-solve workspace. Hot loops should instead
/// reuse [`ProjectedTridiagonalWorkspace`] and call
/// [`ProjectedTridiagonalWorkspace::exp_neg_tk`] directly.
///
/// # Errors
/// Returns an error if the self-adjoint eigendecomposition fails.
pub fn exp_neg_tk(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>, anyhow::Error> {
    let k = alphas.len();
    if k == 0 {
        return Ok(Mat::zeros(0, 1));
    }

    let mut result = Mat::<f64>::zeros(k, 1);
    let mut projected = ProjectedTridiagonalWorkspace::new(k, Par::Seq);
    projected.exp_neg_tk(alphas, betas, result.as_mut())?;
    Ok(result)
}

/// A posteriori residual estimate for the Lanczos approximation of
/// `exp(-A) b` at step `m`: `err = beta_next * |[exp(-T_m) e_1]_m| * ||b||`,
/// where `beta_next = h_{m+1,m}` is the coupling element produced by the
/// `m`-th step of the recurrence.
pub fn saad_error_estimate(alphas: &[f64], betas: &[f64], beta_next: f64, b_norm: f64) -> f64 {
    let mut projected = ProjectedTridiagonalWorkspace::new(alphas.len(), Par::Seq);
    saad_error_estimate_with(&mut projected, alphas, betas, beta_next, b_norm)
}

fn saad_error_estimate_with(
    projected: &mut ProjectedTridiagonalWorkspace,
    alphas: &[f64],
    betas: &[f64],
    beta_next: f64,
    b_norm: f64,
) -> f64 {
    let m = alphas.len();
    if m == 0 {
        return 0.0;
    }

    let mut exp_tm_e1 = Mat::<f64>::zeros(m, 1);
    let exp_tm_e1 = match projected.exp_neg_tk(alphas, betas, exp_tm_e1.as_mut()) {
        Ok(()) => exp_tm_e1,
        Err(_) => return f64::INFINITY,
    };

    let last_component = exp_tm_e1[(m - 1, 0)];
    beta_next * last_component.abs() * b_norm
}

/// Returns the smallest `m <= max_k` for which the a posteriori residual
/// estimate drops below `tol`, along with the truncated decomposition. Falls
/// back to `max_k` and the full decomposition if the tolerance is never met.
///
/// # Errors
/// Returns [`LanczosError`] if the underlying iteration fails.
pub fn adaptive_krylov_dim(
    operator: &impl LinOp<f64>,
    b: MatRef<'_, f64>,
    max_k: usize,
    tol: f64,
    par: Par,
    stack: &mut MemStack,
) -> Result<(usize, LanczosDecomposition), LanczosError> {
    // Setup-phase helper: builds its own workspace. Not on the bench hot path.
    let probe_k = max_k + 1;
    let mut probe_ws = TwoPassWorkspace::new(b.nrows(), probe_k);
    // Run max_k + 1 steps so betas[max_k - 1] exists for the estimate at m = max_k.
    let probe_steps_taken = lanczos_pass_one_into(&mut probe_ws, operator, b, probe_k, par, stack)?;

    let b_norm = probe_ws.b_norm();
    let probe_alphas = probe_ws.alphas();
    let probe_betas = probe_ws.betas();

    let steps = probe_steps_taken.min(max_k);
    if steps == 0 {
        return Ok((
            0,
            LanczosDecomposition {
                alphas: Vec::new(),
                betas: Vec::new(),
                steps_taken: 0,
                b_norm,
            },
        ));
    }

    let mut projected = ProjectedTridiagonalWorkspace::new(steps, par);
    for m in 1..=steps {
        if m > probe_betas.len() {
            break;
        }
        let beta_next = probe_betas[m - 1];
        let err = saad_error_estimate_with(
            &mut projected,
            &probe_alphas[..m],
            &probe_betas[..m.saturating_sub(1)],
            beta_next,
            b_norm,
        );
        if err < tol {
            let truncated = LanczosDecomposition {
                alphas: probe_alphas[..m].to_vec(),
                betas: probe_betas[..m.saturating_sub(1)].to_vec(),
                steps_taken: m,
                b_norm,
            };
            return Ok((m, truncated));
        }
    }

    let truncated = LanczosDecomposition {
        alphas: probe_alphas[..steps].to_vec(),
        betas: probe_betas[..steps.saturating_sub(1)].to_vec(),
        steps_taken: steps,
        b_norm,
    };
    Ok((steps, truncated))
}

/// Estimates `rho(A)` as the largest absolute Ritz value from a short
/// `probe_steps`-iteration Lanczos run on the symmetric operator `A`.
///
/// # Errors
/// Returns [`LanczosError`] if the underlying iteration fails.
pub fn estimate_spectral_radius(
    operator: &impl LinOp<f64>,
    b: MatRef<'_, f64>,
    probe_steps: usize,
    par: Par,
    stack: &mut MemStack,
) -> Result<f64, LanczosError> {
    // Setup-phase helper: builds its own workspace. Not on the bench hot path.
    let mut probe_ws = TwoPassWorkspace::new(b.nrows(), probe_steps);
    let steps_taken = lanczos_pass_one_into(&mut probe_ws, operator, b, probe_steps, par, stack)?;

    if steps_taken == 0 {
        return Ok(0.0);
    }

    let alphas = probe_ws.alphas();
    let betas = probe_ws.betas();
    let k = steps_taken;
    let mut projected = ProjectedTridiagonalWorkspace::new(k, par);
    projected
        .spectral_radius(&alphas[..k], &betas[..betas.len().min(k.saturating_sub(1))])
        .map_err(|e| LanczosError::from(error::LanczosErrorKind::SolverError(e.to_string())))
}
