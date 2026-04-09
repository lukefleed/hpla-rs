//! Lanczos algorithm implementations for computing f(A)b via Krylov subspace projection.
//!
//! This module provides both one-pass and two-pass symmetric Lanczos methods, along with
//! utility functions for the matrix exponential application exp(-A)b. The two-pass variant
//! trades 2x matrix-vector products for O(n) memory instead of O(nk).
//!
//! The module also includes the Saad (1992) a posteriori error estimate for exp(-T_k)*e_1
//! and adaptive Krylov dimension selection based on that estimate.

pub mod algorithms;
pub mod error;
pub mod solvers;

pub use algorithms::{Reorthogonalization, lanczos_scratch};
pub use solvers::{lanczos, lanczos_two_pass};

use faer::{
    Par, Side,
    dyn_stack::MemStack,
    linalg::solvers::SelfAdjointEigen,
    matrix_free::LinOp,
    prelude::*,
};

use crate::lanczos::{
    algorithms::LanczosDecomposition,
    algorithms::lanczos_two_pass::lanczos_pass_one,
    error::LanczosError,
};

/// Computes exp(-T_k) * e_1 for the k x k tridiagonal matrix T_k defined by
/// the given diagonal (`alphas`) and sub/super-diagonal (`betas`) coefficients.
///
/// This is the standard projected problem solver for the Lanczos approximation
/// of exp(-A)b. The tridiagonal matrix is negated, eigendecomposed, and the
/// matrix exponential is applied in the eigenbasis:
///
///   exp(-T_k) * e_1 = Q * diag(exp(lambda_i)) * Q^T * e_1
///
/// where lambda_i are the eigenvalues of -T_k and Q is the orthogonal eigenvector matrix.
///
/// # Arguments
/// * `alphas` - Diagonal elements of T_k. Length determines the dimension k.
/// * `betas` - Sub/super-diagonal elements of T_k. Length must be `k - 1`.
///
/// # Returns
/// A k x 1 dense matrix containing exp(-T_k) * e_1, or an error if the
/// eigendecomposition fails.
///
/// # Errors
/// Returns `anyhow::Error` if the self-adjoint eigendecomposition does not converge.
pub fn exp_neg_tk_solver(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>, anyhow::Error> {
    let k = alphas.len();
    if k == 0 {
        return Ok(Mat::zeros(0, 1));
    }

    // Build the k x k tridiagonal matrix -T_k.
    let mut neg_t_k = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        neg_t_k[(i, i)] = -alphas[i];
    }
    for i in 0..betas.len() {
        neg_t_k[(i, i + 1)] = -betas[i];
        neg_t_k[(i + 1, i)] = -betas[i];
    }

    // Eigendecompose -T_k. Since T_k is real symmetric, -T_k is also real symmetric.
    let evd = SelfAdjointEigen::new(neg_t_k.as_ref(), Side::Lower)
        .map_err(|e| anyhow::anyhow!("eigendecomposition of -T_k failed: {e:?}"))?;
    let q = evd.U(); // k x k orthogonal eigenvector matrix
    let s = evd.S(); // diagonal of eigenvalues

    // Compute exp(-T_k) * e_1 = Q * diag(exp(lambda_i)) * Q^T * e_1.
    // Since e_1 = [1, 0, ..., 0]^T, Q^T * e_1 is simply the first row of Q^T,
    // i.e., the first column of Q read as a column (since Q is real, Q^T[..,0] = Q[0,..]).
    let eigenvalues = s.column_vector();

    let mut result = Mat::<f64>::zeros(k, 1);
    for j in 0..k {
        let exp_lambda = eigenvalues.get(j).exp();
        let q_0j = q[(0, j)]; // (Q^T * e_1)_j = Q[0, j]
        // result += exp(lambda_j) * q_0j * q[.., j]
        for i in 0..k {
            result[(i, 0)] += exp_lambda * q_0j * q[(i, j)];
        }
    }

    Ok(result)
}

/// Computes the Saad (1992) a posteriori error estimate for the Lanczos
/// approximation of exp(-A)b.
///
/// From the Lanczos relation `A * V_m = V_m * T_m + beta_m * v_{m+1} * e_m^T`,
/// the estimate is:
///   err = beta_m * |[exp(-T_m) * e_1]_m| * ||b||
///
/// where `[exp(-T_m) * e_1]_m` denotes the last component (index m-1) of the
/// vector exp(-T_m) * e_1, and `beta_m` is the off-diagonal coupling element
/// produced at step m of the Lanczos recurrence (i.e., `h_{m+1,m}`).
///
/// # Arguments
/// * `alphas` - Diagonal elements of T_m (length m).
/// * `betas` - Sub/super-diagonal elements of T_m (length m-1).
/// * `beta_next` - The coupling element beta_m (= `h_{m+1,m}`) from the Lanczos decomposition.
/// * `b_norm` - The L2 norm of the original vector b.
///
/// # Returns
/// The scalar error estimate. Returns 0.0 if `alphas` is empty.
pub fn saad_error_estimate(alphas: &[f64], betas: &[f64], beta_next: f64, b_norm: f64) -> f64 {
    let m = alphas.len();
    if m == 0 {
        return 0.0;
    }

    // Compute exp(-T_m) * e_1 via eigendecomposition, reusing the same logic.
    let exp_tm_e1 = match exp_neg_tk_solver(alphas, betas) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };

    // The last component of exp(-T_m) * e_1.
    let last_component = exp_tm_e1[(m - 1, 0)];

    beta_next * last_component.abs() * b_norm
}

/// Determines the minimum Krylov subspace dimension m such that the Saad error
/// estimate for exp(-A)b drops below a given tolerance.
///
/// Runs `lanczos_pass_one` with `max_k` steps, then scans the resulting
/// decomposition from m = 1 upward to find the smallest m where:
///   beta_{m+1} * |[exp(-T_m) * e_1]_m| * ||b|| < tol
///
/// # Arguments
/// * `operator` - The linear operator A.
/// * `b` - The starting vector.
/// * `max_k` - Upper bound on the Krylov dimension.
/// * `tol` - Target error tolerance.
/// * `par` - Parallelism strategy.
/// * `stack` - Workspace for operator application.
///
/// # Returns
/// A tuple `(m, decomposition)` where `m` is the selected dimension and
/// `decomposition` contains the full Lanczos coefficients from the pass-one run.
/// If no m <= max_k satisfies the tolerance, returns `max_k` and the full decomposition.
///
/// # Errors
/// Returns [`LanczosError`] if the Lanczos iteration fails (e.g., zero input vector).
pub fn determine_krylov_dim(
    operator: &impl LinOp<f64>,
    b: MatRef<'_, f64>,
    max_k: usize,
    tol: f64,
    par: Par,
    stack: &mut MemStack,
) -> Result<(usize, LanczosDecomposition<f64>), LanczosError> {
    // Run max_k + 1 steps so that betas stores the coupling element for step
    // max_k, enabling the Saad estimate at m = max_k.
    let probe = lanczos_pass_one::<f64>(operator, b, max_k + 1, par, stack)?;

    let steps = probe.steps_taken.min(max_k);
    if steps == 0 {
        return Ok((0, probe));
    }

    let b_norm = probe.b_norm;

    // Scan from m = 1 to steps. At each m:
    // - T_m uses alphas[0..m] (diagonal) and betas[0..m-1] (off-diagonal)
    // - beta_next = betas[m-1] = beta_m (the coupling element h_{m+1,m})
    // The extra step guarantees betas[steps-1] exists when steps <= max_k.
    for m in 1..=steps {
        if m > probe.betas.len() {
            break;
        }
        let beta_next = probe.betas[m - 1];
        let err = saad_error_estimate(
            &probe.alphas[..m],
            &probe.betas[..m.saturating_sub(1)],
            beta_next,
            b_norm,
        );
        if err < tol {
            // Truncate the decomposition to the selected dimension for the caller.
            let truncated = LanczosDecomposition {
                alphas: probe.alphas[..m].to_vec(),
                betas: probe.betas[..m.saturating_sub(1)].to_vec(),
                steps_taken: m,
                b_norm,
            };
            return Ok((m, truncated));
        }
    }

    // No dimension satisfied the tolerance; return max_k with a truncated decomposition.
    let truncated = LanczosDecomposition {
        alphas: probe.alphas[..steps].to_vec(),
        betas: probe.betas[..steps.saturating_sub(1)].to_vec(),
        steps_taken: steps,
        b_norm,
    };
    Ok((steps, truncated))
}

/// Estimates the spectral radius of a symmetric linear operator by running a short
/// Lanczos factorization and eigendecomposing the resulting tridiagonal matrix.
///
/// The spectral radius rho(A) = max_i |lambda_i(A)| is approximated by the largest
/// absolute eigenvalue of the k x k tridiagonal Ritz matrix T_k produced by
/// `probe_steps` Lanczos iterations. Convergence of the extremal Ritz values is
/// typically fast for symmetric operators, so 10--30 steps often suffice.
///
/// # Arguments
/// * `operator` - The symmetric linear operator A.
/// * `b` - Starting vector for the Krylov subspace. Must be nonzero.
/// * `probe_steps` - Number of Lanczos iterations to run (e.g., 20).
/// * `par` - Parallelism strategy for operator applications.
/// * `stack` - Workspace for operator application scratch space.
///
/// # Returns
/// An estimate of rho(A), the spectral radius.
///
/// # Errors
/// Returns [`LanczosError`] if the Lanczos iteration fails (e.g., zero input vector).
pub fn estimate_spectral_radius(
    operator: &impl LinOp<f64>,
    b: MatRef<'_, f64>,
    probe_steps: usize,
    par: Par,
    stack: &mut MemStack,
) -> Result<f64, LanczosError> {
    let decomposition = lanczos_pass_one::<f64>(operator, b, probe_steps, par, stack)?;

    let k = decomposition.steps_taken;
    if k == 0 {
        return Ok(0.0);
    }

    // Build the k x k tridiagonal matrix T_k (no negation -- we want eigenvalues of T).
    let mut t_k = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        t_k[(i, i)] = decomposition.alphas[i];
    }
    for i in 0..decomposition.betas.len().min(k.saturating_sub(1)) {
        t_k[(i, i + 1)] = decomposition.betas[i];
        t_k[(i + 1, i)] = decomposition.betas[i];
    }

    // Eigendecompose T_k. The spectral radius is the largest absolute eigenvalue.
    let evd = SelfAdjointEigen::new(t_k.as_ref(), Side::Lower)
        .map_err(|e| LanczosError::from(error::LanczosErrorKind::SolverError(
            format!("eigendecomposition of T_k failed: {e:?}"),
        )))?;
    let eigenvalues = evd.S().column_vector();

    Ok(eigenvalues
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f64, f64::max))
}
