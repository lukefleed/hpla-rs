//! One-pass and two-pass Lanczos drivers for `f(A) b` on a Hermitian `A`.
//!
//! The public API separates setup from execute: [`LanczosWorkspace`] and
//! [`TwoPassWorkspace`] own every n-sized and n*k-sized buffer the kernels
//! touch, and are constructed once per `(n, k)` pair. The `_into` driver
//! variants reuse the workspace without any hot-path allocation. Thin
//! allocating wrappers [`lanczos`] and [`lanczos_two_pass`] are retained for
//! call sites that are not in a timing window (tests, probes).

use crate::lanczos::{
    algorithms::{
        Reorthogonalization,
        lanczos::lanczos_one_pass_into,
        lanczos_two_pass::{lanczos_pass_one_into, lanczos_pass_two_into},
    },
    error::{LanczosError, LanczosErrorKind},
};
use faer::{
    Par,
    dyn_stack::MemStack,
    matrix_free::LinOp,
    prelude::*,
};

/// Working buffers for a one-pass Lanczos run at a fixed `(n, k_cap)`.
///
/// Owns the basis `V_k` (`n x k_cap`), the three rolling vectors
/// `v_prev`, `v_curr`, `work` (each `n x 1`), the output `x_k` (`n x 1`),
/// and the scalar tridiagonal `alphas`/`betas` with capacity `k_cap`.
/// A single instance is reused across many `lanczos_into` calls with the
/// same matrix dimensions.
#[derive(Debug)]
pub struct LanczosWorkspace {
    v_k: Mat<f64>,
    v_prev: Mat<f64>,
    v_curr: Mat<f64>,
    work: Mat<f64>,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    x_k: Mat<f64>,
    b_norm: f64,
    k_cap: usize,
}

impl LanczosWorkspace {
    /// Allocates every buffer for a problem of size `n` and Krylov
    /// dimension at most `k`.
    #[must_use]
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            v_k: Mat::zeros(n, k),
            v_prev: Mat::zeros(n, 1),
            v_curr: Mat::zeros(n, 1),
            work: Mat::zeros(n, 1),
            alphas: Vec::with_capacity(k),
            betas: Vec::with_capacity(k),
            x_k: Mat::zeros(n, 1),
            b_norm: 0.0,
            k_cap: k,
        }
    }

    /// Read-only view of the current output vector `x_k`.
    #[must_use]
    pub fn x_k(&self) -> MatRef<'_, f64> {
        self.x_k.as_ref()
    }

    /// Number of rows every internal buffer is sized for.
    pub(crate) fn n(&self) -> usize {
        self.v_prev.nrows()
    }

    /// Capacity of the basis and of the scalar vectors.
    pub(crate) fn k_cap(&self) -> usize {
        self.k_cap
    }

    /// `||b||` cached by [`crate::lanczos::algorithms::lanczos::lanczos_one_pass_into`]
    /// before the recurrence starts, for reuse in the final
    /// `x_k = ||b|| V_k g` scaling without a second O(n) sweep.
    pub(crate) fn b_norm(&self) -> f64 {
        self.b_norm
    }

    /// Setter used by the one-pass driver to record `||b||`.
    pub(crate) fn set_b_norm(&mut self, b_norm: f64) {
        self.b_norm = b_norm;
    }

    /// Disjoint mutable borrows over every buffer that the one-pass driver
    /// touches, so the caller can hand them to
    /// [`crate::lanczos::algorithms::LanczosIteration::new_borrowed`] without
    /// fighting the borrow checker.
    #[expect(
        clippy::type_complexity,
        reason = "destructuring helper; naming each slot would be less clear"
    )]
    pub(crate) fn parts_mut(
        &mut self,
    ) -> (
        &mut Mat<f64>,
        &mut Mat<f64>,
        &mut Mat<f64>,
        &mut Mat<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Mat<f64>,
    ) {
        (
            &mut self.v_k,
            &mut self.v_prev,
            &mut self.v_curr,
            &mut self.work,
            &mut self.alphas,
            &mut self.betas,
            &mut self.x_k,
        )
    }
}

/// Working buffers for a two-pass Lanczos run at a fixed `(n, k_cap)`.
///
/// Holds the three rolling vectors plus the output `x_k`. Two-pass drops
/// the basis matrix: only `O(n)` storage.
#[derive(Debug)]
pub struct TwoPassWorkspace {
    v_prev: Mat<f64>,
    v_curr: Mat<f64>,
    work: Mat<f64>,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    x_k: Mat<f64>,
    b_norm: f64,
    k_cap: usize,
}

impl TwoPassWorkspace {
    /// Allocates every buffer for a problem of size `n` and Krylov
    /// dimension at most `k`.
    #[must_use]
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            v_prev: Mat::zeros(n, 1),
            v_curr: Mat::zeros(n, 1),
            work: Mat::zeros(n, 1),
            alphas: Vec::with_capacity(k),
            betas: Vec::with_capacity(k),
            x_k: Mat::zeros(n, 1),
            b_norm: 0.0,
            k_cap: k,
        }
    }

    /// Read-only view of the current output vector `x_k`.
    #[must_use]
    pub fn x_k(&self) -> MatRef<'_, f64> {
        self.x_k.as_ref()
    }

    /// Number of rows every internal buffer is sized for.
    pub(crate) fn n(&self) -> usize {
        self.v_prev.nrows()
    }

    /// Capacity of the scalar vectors.
    pub(crate) fn k_cap(&self) -> usize {
        self.k_cap
    }

    /// `||b||` cached by [`lanczos_pass_one_into`] for use by
    /// [`lanczos_pass_two_into`] and the high-level driver.
    pub(crate) fn b_norm(&self) -> f64 {
        self.b_norm
    }

    /// Read-only view over the diagonal scalars of `T_k`.
    pub(crate) fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Read-only view over the off-diagonal scalars of `T_k`.
    pub(crate) fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// Setter used by the pass-one driver to record `||b||`.
    pub(crate) fn set_b_norm(&mut self, b_norm: f64) {
        self.b_norm = b_norm;
    }

    /// Disjoint mutable borrows over the rolling buffers, the scalar vectors,
    /// and the output, for use in [`lanczos_pass_one_into`] and
    /// [`lanczos_pass_two_into`].
    #[expect(
        clippy::type_complexity,
        reason = "destructuring helper; naming each slot would be less clear"
    )]
    pub(crate) fn parts_mut(
        &mut self,
    ) -> (
        &mut Mat<f64>,
        &mut Mat<f64>,
        &mut Mat<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Mat<f64>,
    ) {
        (
            &mut self.v_prev,
            &mut self.v_curr,
            &mut self.work,
            &mut self.alphas,
            &mut self.betas,
            &mut self.x_k,
        )
    }
}

/// One-pass Lanczos for `f(A) b`: stores the full basis `V_k` (`O(nk)` memory)
/// and invokes `f_tk` on the resulting tridiagonal coefficients. The result
/// `x_k = ||b|| V_k f(T_k) e_1` is written into `ws.x_k`.
///
/// # Errors
/// Returns [`LanczosError`] if `b` has the wrong number of rows, if `k`
/// exceeds `ws.k_cap`, on zero input, on breakdown mishandling, or if `f_tk`
/// fails or returns a vector of the wrong length.
#[expect(
    clippy::too_many_arguments,
    reason = "direct translation of the existing allocating API; grouping would fight the LinOp/closure generics"
)]
pub fn lanczos_into<O, F>(
    ws: &mut LanczosWorkspace,
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    reorthog: Reorthogonalization,
    stack: &mut MemStack,
    mut f_tk: F,
) -> Result<(), LanczosError>
where
    O: LinOp<f64>,
    F: FnMut(&[f64], &[f64]) -> Result<Mat<f64>, anyhow::Error>,
{
    if b.nrows() != ws.n() {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "b.nrows".to_string(),
            expected: ws.n(),
            actual: b.nrows(),
        }
        .into());
    }
    if k > ws.k_cap() {
        return Err(LanczosErrorKind::CapacityExceeded {
            param_name: "k".to_string(),
            cap: ws.k_cap(),
            requested: k,
        }
        .into());
    }

    let steps_taken = lanczos_one_pass_into(ws, operator, b, k, par, reorthog, stack)?;

    if steps_taken == 0 {
        zip!(ws.x_k.as_mut()).for_each(|unzip!(x_i)| *x_i = 0.0);
        return Ok(());
    }

    // Lengths of the valid scalar slices that drive `f_tk`.
    // `ws.betas` may carry up to `steps_taken - 1` off-diagonals.
    let n_alpha = steps_taken;
    let n_beta = ws.betas.len().min(steps_taken.saturating_sub(1));
    let alphas_slice = &ws.alphas[..n_alpha];
    let betas_slice = &ws.betas[..n_beta];

    let g = f_tk(alphas_slice, betas_slice)
        .map_err(|e| LanczosError::from(LanczosErrorKind::SolverError(e.to_string())))?;

    if g.nrows() != steps_taken || g.ncols() != 1 {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "g".to_string(),
            expected: steps_taken,
            actual: g.nrows(),
        }
        .into());
    }

    // x_k = ||b|| * V_k * g, column-by-column accumulation.
    // Uses zip! instead of the convenience matmul, which internally
    // allocates O(n*k) packing buffers via the global allocator and
    // would violate the zero-n-allocation hot-path policy.
    let b_norm = ws.b_norm();
    let v_k_slice = ws.v_k.as_ref().get(.., 0..steps_taken);
    zip!(ws.x_k.as_mut()).for_each(|unzip!(x_i)| *x_i = 0.0);
    for j in 0..steps_taken {
        let coeff = b_norm * g[(j, 0)];
        let v_col = v_k_slice.subcols(j, 1);
        zip!(ws.x_k.as_mut(), v_col).for_each(|unzip!(x_i, v_ij)| {
            *x_i += coeff * *v_ij;
        });
    }

    Ok(())
}

/// Two-pass Lanczos for `f(A) b`: the first pass produces only the
/// tridiagonal scalars, the second pass replays the recurrence to
/// accumulate `x_k = V_k y_k` with `O(n)` working memory. The result is
/// written into `ws.x_k`.
///
/// # Errors
/// Returns [`LanczosError`] if `b` has the wrong number of rows, if `k`
/// exceeds `ws.k_cap`, on zero input, on breakdown mishandling, or if `f_tk`
/// fails or returns a vector of the wrong length.
pub fn lanczos_two_pass_into<O, F>(
    ws: &mut TwoPassWorkspace,
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    stack: &mut MemStack,
    mut f_tk: F,
) -> Result<(), LanczosError>
where
    O: LinOp<f64>,
    F: FnMut(&[f64], &[f64]) -> Result<Mat<f64>, anyhow::Error>,
{
    if b.nrows() != ws.n() {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "b.nrows".to_string(),
            expected: ws.n(),
            actual: b.nrows(),
        }
        .into());
    }
    if k > ws.k_cap() {
        return Err(LanczosErrorKind::CapacityExceeded {
            param_name: "k".to_string(),
            cap: ws.k_cap(),
            requested: k,
        }
        .into());
    }

    let steps_taken = lanczos_pass_one_into(ws, operator, b, k, par, stack)?;

    if steps_taken == 0 {
        zip!(ws.x_k.as_mut()).for_each(|unzip!(x_i)| *x_i = 0.0);
        return Ok(());
    }

    let n_alpha = steps_taken;
    let n_beta = ws.betas.len().min(steps_taken.saturating_sub(1));
    let alphas_slice = &ws.alphas[..n_alpha];
    let betas_slice = &ws.betas[..n_beta];

    let mut g = f_tk(alphas_slice, betas_slice)
        .map_err(|e| LanczosError::from(LanczosErrorKind::SolverError(e.to_string())))?;

    if g.nrows() != steps_taken || g.ncols() != 1 {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "g".to_string(),
            expected: steps_taken,
            actual: g.nrows(),
        }
        .into());
    }

    // Fold ||b|| into y_k in place so pass two accumulates x_k directly.
    let b_norm = ws.b_norm();
    zip!(g.as_mut()).for_each(|unzip!(y_i)| {
        *y_i *= b_norm;
    });

    lanczos_pass_two_into(ws, operator, b, g.as_ref(), steps_taken, par, stack)
}

/// Allocating wrapper over [`lanczos_into`]: builds a fresh workspace and
/// returns the owned result. Suitable for tests and one-off calls; hot
/// loops must use [`lanczos_into`] with a reused workspace.
///
/// # Errors
/// See [`lanczos_into`].
pub fn lanczos<O, F>(
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    reorthog: Reorthogonalization,
    stack: &mut MemStack,
    f_tk: F,
) -> Result<Mat<f64>, LanczosError>
where
    O: LinOp<f64>,
    F: FnMut(&[f64], &[f64]) -> Result<Mat<f64>, anyhow::Error>,
{
    let mut ws = LanczosWorkspace::new(b.nrows(), k);
    lanczos_into(&mut ws, operator, b, k, par, reorthog, stack, f_tk)?;
    Ok(ws.x_k.clone())
}

/// Allocating wrapper over [`lanczos_two_pass_into`]: builds a fresh
/// workspace and returns the owned result. Suitable for tests and
/// one-off calls; hot loops must use [`lanczos_two_pass_into`] with a
/// reused workspace.
///
/// # Errors
/// See [`lanczos_two_pass_into`].
pub fn lanczos_two_pass<O, F>(
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    stack: &mut MemStack,
    f_tk: F,
) -> Result<Mat<f64>, LanczosError>
where
    O: LinOp<f64>,
    F: FnMut(&[f64], &[f64]) -> Result<Mat<f64>, anyhow::Error>,
{
    let mut ws = TwoPassWorkspace::new(b.nrows(), k);
    lanczos_two_pass_into(&mut ws, operator, b, k, par, stack, f_tk)?;
    Ok(ws.x_k.clone())
}
