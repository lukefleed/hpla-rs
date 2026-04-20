//! Reusable workspaces for projected tridiagonal problems arising in Lanczos.
//!
//! The hot path needs `exp(-T_k)e_1` with a fixed Krylov cap `k_cap`. faer
//! exposes a tridiagonal self-adjoint eigensolver, but its low-level API
//! expects caller-owned eigenspace buffers and scratch. This module packages
//! those allocations so the projected solve can be reused across iterations.

use anyhow::ensure;
use faer::{
    Par,
    diag::{Diag, DiagRef},
    dyn_stack::{MemBuffer, MemStack},
    linalg::evd::{ComputeEigenvectors, self_adjoint_evd_scratch, tridiagonal_self_adjoint_evd},
    prelude::*,
};

/// Reusable buffers for projected tridiagonal problems at a fixed `k_cap`.
///
/// The workspace owns the eigenspace matrix, the eigenvalue buffer, and the
/// faer scratch memory required by the tridiagonal self-adjoint eigensolver.
/// Once built, repeated solves of `exp(-T_k)e_1` do not allocate.
pub struct ProjectedTridiagonalWorkspace {
    k_cap: usize,
    par: Par,
    eigenvectors: Mat<f64>,
    eigenvalues: Diag<f64>,
    scratch: MemBuffer,
}

impl ProjectedTridiagonalWorkspace {
    /// Allocates every buffer for projected tridiagonal problems of size at
    /// most `k`.
    #[must_use]
    pub fn new(k: usize, par: Par) -> Self {
        let scratch = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
            k,
            ComputeEigenvectors::Yes,
            par,
            Default::default(),
        ));
        Self {
            k_cap: k,
            par,
            eigenvectors: Mat::zeros(k, k),
            eigenvalues: Diag::zeros(k),
            scratch,
        }
    }

    /// Returns `exp(-T_k)e_1`, where `T_k` is the tridiagonal defined by
    /// `alphas` (diagonal) and `betas` (off-diagonal), writing the result into
    /// `out`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tridiagonal dimensions are inconsistent, if
    /// `out` has the wrong shape, or if the eigendecomposition fails.
    pub fn exp_neg_tk(
        &mut self,
        alphas: &[f64],
        betas: &[f64],
        mut out: MatMut<'_, f64>,
    ) -> Result<(), anyhow::Error> {
        let k = self.validate_tridiagonal(alphas, betas)?;
        ensure!(
            out.nrows() == k,
            "projected solve output has {} rows, expected {k}",
            out.nrows(),
        );
        ensure!(
            out.ncols() == 1,
            "projected solve output has {} columns, expected 1",
            out.ncols(),
        );

        if k == 0 {
            return Ok(());
        }

        {
            let stack = MemStack::new(&mut self.scratch);
            tridiagonal_self_adjoint_evd(
                DiagRef::from_slice(alphas),
                DiagRef::from_slice(betas),
                self.eigenvalues
                    .column_vector_mut()
                    .subrows_mut(0, k)
                    .as_diagonal_mut(),
                Some(self.eigenvectors.as_mut().submatrix_mut(0, 0, k, k)),
                self.par,
                stack,
                Default::default(),
            )
            .map_err(|e| anyhow::anyhow!("eigendecomposition of T_k failed: {e:?}"))?;
        }

        for i in 0..k {
            out[(i, 0)] = 0.0;
        }
        for j in 0..k {
            let scale = (-self.eigenvalues[j]).exp() * self.eigenvectors[(0, j)];
            for i in 0..k {
                out[(i, 0)] += scale * self.eigenvectors[(i, j)];
            }
        }

        Ok(())
    }

    /// Returns the spectral radius of the tridiagonal defined by `alphas` and
    /// `betas`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tridiagonal dimensions are inconsistent or if
    /// the eigendecomposition fails.
    pub fn spectral_radius(&mut self, alphas: &[f64], betas: &[f64]) -> Result<f64, anyhow::Error> {
        let k = self.validate_tridiagonal(alphas, betas)?;
        if k == 0 {
            return Ok(0.0);
        }

        {
            let stack = MemStack::new(&mut self.scratch);
            tridiagonal_self_adjoint_evd(
                DiagRef::from_slice(alphas),
                DiagRef::from_slice(betas),
                self.eigenvalues
                    .column_vector_mut()
                    .subrows_mut(0, k)
                    .as_diagonal_mut(),
                None,
                self.par,
                stack,
                Default::default(),
            )
            .map_err(|e| anyhow::anyhow!("eigendecomposition of T_k failed: {e:?}"))?;
        }

        Ok((0..k)
            .map(|i| self.eigenvalues[i].abs())
            .fold(0.0_f64, f64::max))
    }

    fn validate_tridiagonal(&self, alphas: &[f64], betas: &[f64]) -> Result<usize, anyhow::Error> {
        let k = alphas.len();
        ensure!(
            k <= self.k_cap,
            "tridiagonal size {k} exceeds workspace capacity {}",
            self.k_cap,
        );
        ensure!(
            betas.len() == k.saturating_sub(1),
            "off-diagonal has length {}, expected {}",
            betas.len(),
            k.saturating_sub(1),
        );
        Ok(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use faer::{Side, linalg::solvers::SelfAdjointEigen};

    fn dense_exp_neg_tk(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>> {
        let k = alphas.len();
        let mut t_k = Mat::<f64>::zeros(k, k);
        for i in 0..k {
            t_k[(i, i)] = alphas[i];
        }
        for i in 0..betas.len() {
            t_k[(i, i + 1)] = betas[i];
            t_k[(i + 1, i)] = betas[i];
        }

        let evd = SelfAdjointEigen::new(t_k.as_ref(), Side::Lower)
            .map_err(|e| anyhow::anyhow!("dense eigendecomposition failed: {e:?}"))?;
        let q = evd.U();
        let s = evd.S();
        let mut out = Mat::<f64>::zeros(k, 1);
        for j in 0..k {
            let scale = (-s[j]).exp() * q[(0, j)];
            for i in 0..k {
                out[(i, 0)] += scale * q[(i, j)];
            }
        }
        Ok(out)
    }

    #[test]
    fn test_projected_workspace_exp_neg_tk_matches_dense_reference() -> Result<()> {
        let alphas = [2.0, 3.0, 5.0, 7.0];
        let betas = [0.5, 1.25, 0.75];
        let expected = dense_exp_neg_tk(&alphas, &betas)?;

        let mut projected = ProjectedTridiagonalWorkspace::new(alphas.len(), Par::Seq);
        let mut actual = Mat::<f64>::zeros(alphas.len(), 1);
        projected.exp_neg_tk(&alphas, &betas, actual.as_mut())?;

        let err = (0..alphas.len())
            .map(|i| (actual[(i, 0)] - expected[(i, 0)]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12, "relative mismatch in exp(-T_k)e_1: {err:.3e}");
        Ok(())
    }

    #[test]
    fn test_projected_workspace_spectral_radius_matches_dense_reference() -> Result<()> {
        let alphas = [2.0, 3.0, 5.0, 7.0];
        let betas = [0.5, 1.25, 0.75];

        let mut t_k = Mat::<f64>::zeros(alphas.len(), alphas.len());
        for i in 0..alphas.len() {
            t_k[(i, i)] = alphas[i];
        }
        for i in 0..betas.len() {
            t_k[(i, i + 1)] = betas[i];
            t_k[(i + 1, i)] = betas[i];
        }
        let expected = SelfAdjointEigen::new(t_k.as_ref(), Side::Lower)
            .map_err(|e| anyhow::anyhow!("dense eigendecomposition failed: {e:?}"))?
            .S()
            .column_vector()
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);

        let mut projected = ProjectedTridiagonalWorkspace::new(alphas.len(), Par::Seq);
        let actual = projected.spectral_radius(&alphas, &betas)?;
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual:.3e}, expected={expected:.3e}"
        );
        Ok(())
    }
}
