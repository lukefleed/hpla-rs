//! Core Lanczos primitives shared by the one-pass and two-pass solvers:
//! the stateful recurrence iterator, the breakdown tolerance, and the
//! tridiagonal decomposition type. Invoked only from
//! [`crate::lanczos::solvers`] and the Lanczos module internals.

pub mod lanczos;
pub mod lanczos_two_pass;

use crate::lanczos::error::{LanczosError, LanczosErrorKind};
use faer::{
    Par,
    dyn_stack::{MemStack, StackReq},
    matrix_free::LinOp,
    prelude::*,
};

/// Reorthogonalization strategy for the one-pass Lanczos basis. The two-pass
/// variant drops the basis vectors and cannot reorthogonalize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Reorthogonalization {
    /// Bare three-term recurrence.
    #[default]
    None,
    /// Full reorthogonalization with a DGKS refinement pass.
    Full,
}

/// Scalar output of a Lanczos run: the tridiagonal `T_k` and `||b||`.
/// Consumed by [`crate::lanczos::adaptive_krylov_dim`] as its return value.
#[derive(Debug, Clone)]
pub struct LanczosDecomposition {
    /// Diagonal of `T_k`.
    pub alphas: Vec<f64>,
    /// Off-diagonal of `T_k`, length `steps_taken - 1`.
    pub betas: Vec<f64>,
    /// Number of steps actually completed (may be less than the request on breakdown).
    pub steps_taken: usize,
    /// `||b||_2`, cached for the final scaling `x_k = ||b|| V_k f(T_k) e_1`.
    pub b_norm: f64,
}

/// Breakdown threshold scaled off machine epsilon.
pub(crate) fn breakdown_tolerance() -> f64 {
    f64::EPSILON * 1000.0
}

/// Scratch requirement for one Lanczos step, forwarded from the operator.
pub fn lanczos_scratch(operator: &impl LinOp<f64>, par: Par) -> StackReq {
    operator.apply_scratch(1, par)
}

/// Computes the unnormalized Lanczos residual
/// `w = A v_curr - alpha v_curr - beta_prev v_prev` and returns `alpha`, the
/// Rayleigh quotient `<v_curr, A v_curr>` taken before the `alpha v_curr`
/// subtraction. The norm `beta = ||w||` and the breakdown check are left to
/// the caller, so an optional reorthogonalization pass over `w` (including
/// over `v_curr` itself; in exact arithmetic the component is already zero)
/// feeds one coherent `beta` into both the stored tridiagonal and the
/// iterator's `beta_prev`.
pub(crate) fn lanczos_recurrence_step<O: LinOp<f64>>(
    operator: &O,
    mut w: MatMut<'_, f64>,
    v_curr: MatRef<'_, f64>,
    v_prev: MatRef<'_, f64>,
    beta_prev: f64,
    par: Par,
    stack: &mut MemStack,
) -> f64 {
    operator.apply(w.rb_mut(), v_curr, par, stack);

    // Fused: w -= beta_prev * v_prev AND alpha = <v_curr, w>, one memory pass.
    let mut alpha = 0.0_f64;
    zip!(w.rb_mut(), v_prev, v_curr).for_each(|unzip!(w_i, v_prev_i, v_curr_i)| {
        *w_i -= beta_prev * *v_prev_i;
        alpha += *v_curr_i * *w_i;
    });

    zip!(w.rb_mut(), v_curr).for_each(|unzip!(w_i, v_curr_i)| {
        *w_i -= alpha * *v_curr_i;
    });

    alpha
}

/// Scalar output of one Lanczos step.
pub(crate) struct LanczosStep {
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
}

/// Drives the three-term recurrence one step at a time over three rolling
/// vectors `v_prev`, `v_curr`, `work` borrowed from a caller-owned workspace.
/// The borrow lifetime `'ws` is disjoint from the operator lifetime `'a`.
pub(crate) struct LanczosIteration<'a, 'ws, O: LinOp<f64>> {
    operator: &'a O,
    v_prev: &'ws mut Mat<f64>,
    v_curr: &'ws mut Mat<f64>,
    work: &'ws mut Mat<f64>,
    beta_prev: f64,
    tolerance: f64,
    par: Par,
    k: usize,
    max_k: usize,
}

impl<'a, 'ws, O: LinOp<f64>> LanczosIteration<'a, 'ws, O> {
    /// Writes `v_1 = b / ||b||` into the provided `v_curr` buffer, zeros the
    /// provided `v_prev` and `work` buffers, and seeds the iterator state.
    /// The caller is responsible for precomputing `b_norm = ||b||_2` so this
    /// constructor does not pay a second O(n) sweep; the high-level driver
    /// caches it in the workspace for downstream consumers.
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is a distinct rolling buffer or scalar; grouping would require an extra struct for no benefit"
    )]
    pub(crate) fn new_borrowed(
        operator: &'a O,
        v_prev: &'ws mut Mat<f64>,
        v_curr: &'ws mut Mat<f64>,
        work: &'ws mut Mat<f64>,
        b: MatRef<'_, f64>,
        b_norm: f64,
        max_k: usize,
        par: Par,
    ) -> Result<Self, LanczosError> {
        debug_assert_eq!(
            v_prev.nrows(),
            b.nrows(),
            "v_prev.nrows() must match b.nrows()"
        );
        debug_assert_eq!(
            v_curr.nrows(),
            b.nrows(),
            "v_curr.nrows() must match b.nrows()"
        );
        debug_assert_eq!(
            work.nrows(),
            b.nrows(),
            "work.nrows() must match b.nrows()"
        );

        let zero_threshold = breakdown_tolerance();
        if b_norm <= zero_threshold {
            return Err(LanczosErrorKind::ZeroInputVector.into());
        }

        let inv_norm = b_norm.recip();

        zip!(v_prev.as_mut()).for_each(|unzip!(x)| *x = 0.0);
        zip!(work.as_mut()).for_each(|unzip!(x)| *x = 0.0);
        zip!(v_curr.as_mut(), b).for_each(|unzip!(v, b_i)| *v = *b_i * inv_norm);

        Ok(Self {
            operator,
            v_prev,
            v_curr,
            work,
            beta_prev: 0.0,
            tolerance: breakdown_tolerance(),
            par,
            k: 0,
            max_k,
        })
    }

    /// Read-only view of `v_curr` for callers that need to copy the current
    /// basis vector out before the next step swaps the rolling buffers.
    pub(crate) fn v_curr(&self) -> MatRef<'_, f64> {
        self.v_curr.as_ref()
    }

    /// One step with a no-op reortho hook.
    pub(crate) fn next_step(&mut self, stack: &mut MemStack) -> Option<LanczosStep> {
        self.next_step_with(stack, |_| {})
    }

    /// One step, letting `reortho` correct the unnormalized residual `w`
    /// before `beta = ||w||` is derived. The returned `beta` and the stored
    /// `beta_prev` both reflect the corrected `w`, keeping the tridiagonal
    /// consistent with the next recurrence step.
    pub(crate) fn next_step_with<F>(
        &mut self,
        stack: &mut MemStack,
        mut reortho: F,
    ) -> Option<LanczosStep>
    where
        F: FnMut(MatMut<'_, f64>),
    {
        if self.k >= self.max_k {
            return None;
        }

        let alpha = lanczos_recurrence_step(
            self.operator,
            self.work.as_mut(),
            self.v_curr.as_ref(),
            self.v_prev.as_ref(),
            self.beta_prev,
            self.par,
            stack,
        );

        reortho(self.work.as_mut());

        let beta = self.work.as_ref().norm_l2();
        self.k += 1;

        if beta <= self.tolerance {
            // Breakdown: the (possibly reorthogonalized) residual lies in
            // span(V_j) to working precision. Leave v_prev / v_curr / work
            // untouched and return zero beta so the caller terminates.
            return Some(LanczosStep {
                alpha,
                beta: 0.0,
            });
        }

        let inv_beta = beta.recip();
        zip!(self.work.as_mut()).for_each(|unzip!(w_i)| {
            *w_i *= inv_beta;
        });

        core::mem::swap(self.v_prev, self.v_curr);
        core::mem::swap(self.v_curr, self.work);
        self.beta_prev = beta;

        Some(LanczosStep { alpha, beta })
    }
}
