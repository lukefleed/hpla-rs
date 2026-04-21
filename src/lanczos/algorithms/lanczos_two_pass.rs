//! Two-pass symmetric Lanczos for `f(A) b` with `O(n)` memory. The first
//! pass ([`lanczos_pass_one_into`]) produces the scalars of `T_k` without
//! storing the basis; the second pass ([`lanczos_pass_two_into`]) replays
//! the recurrence from those scalars to accumulate `x_k = V_k y_k` on the
//! fly. Both passes reuse the buffers owned by
//! [`crate::lanczos::solvers::TwoPassWorkspace`].

use super::{LanczosError, LanczosErrorKind, LanczosIteration, breakdown_tolerance};
use crate::lanczos::solvers::TwoPassWorkspace;
use faer::{
    Accum, Par, dyn_stack::MemStack, linalg::matmul::matmul, matrix_free::LinOp, prelude::*,
};

/// Runs up to `k` Lanczos steps and writes the scalar tridiagonal into
/// `ws.alphas`/`ws.betas`. Caches `||b||` in `ws` for the second pass.
/// Returns the number of steps actually taken.
///
/// The caller has already validated `b.nrows() == ws.n()` and
/// `k <= ws.k_cap()` in the high-level driver.
///
/// # Errors
/// Returns [`LanczosError`] if `b` is the zero vector.
pub(crate) fn lanczos_pass_one_into<O: LinOp<f64>>(
    ws: &mut TwoPassWorkspace,
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    stack: &mut MemStack,
) -> Result<usize, LanczosError> {
    let b_norm = b.norm_l2();
    ws.set_b_norm(b_norm);

    let tolerance = breakdown_tolerance();
    // Pass one never touches `x_k`; the second pass writes to it.
    let (v_prev, v_curr, work, alphas, betas, _) = ws.parts_mut();

    alphas.clear();
    betas.clear();

    if k == 0 {
        return Ok(0);
    }

    let mut lanczos_iter =
        LanczosIteration::new_borrowed(operator, v_prev, v_curr, work, b, b_norm, k, par)?;

    let mut steps_taken = 0usize;
    for i in 0..k {
        if let Some(step) = lanczos_iter.next_step(stack) {
            alphas.push(step.alpha);
            steps_taken += 1;

            if step.beta <= tolerance {
                break;
            }

            if i < k - 1 {
                betas.push(step.beta);
            }
        } else {
            break;
        }
    }

    Ok(steps_taken)
}

/// Replays the Lanczos three-term recurrence to recover
/// `w = A v_j - alpha_j v_j - beta_{j-1} v_{j-1}` with only the three
/// rolling vectors in flight.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct rolling buffer or scalar; grouping would require an extra struct for no benefit"
)]
fn reconstruct_step<O: LinOp<f64>>(
    operator: &O,
    mut w: MatMut<'_, f64>,
    v_curr: MatRef<'_, f64>,
    v_prev: MatRef<'_, f64>,
    alpha_j: f64,
    beta_prev: f64,
    par: Par,
    stack: &mut MemStack,
) {
    operator.apply(w.rb_mut(), v_curr, par, stack);

    // Single-store fused axpy: w -= beta_prev*v_prev + alpha_j*v_curr.
    zip!(w.rb_mut(), v_prev, v_curr).for_each(|unzip!(w_i, v_prev_i, v_curr_i)| {
        *w_i -= beta_prev * *v_prev_i + alpha_j * *v_curr_i;
    });
}

/// Replays the recurrence from the scalars stashed in `ws` to compute
/// `x_k = V_k y_k`, writing into `ws.x_k`. `y_k` is the projected
/// coefficient vector (already scaled by `||b||`), with `y_k.nrows()`
/// equal to `steps_taken`.
///
/// # Errors
/// Returns [`LanczosError`] on a `y_k` length mismatch or a zero `b_norm`
/// cached from the first pass.
pub(crate) fn lanczos_pass_two_into<O: LinOp<f64>>(
    ws: &mut TwoPassWorkspace,
    operator: &O,
    b: MatRef<'_, f64>,
    y_k: MatRef<'_, f64>,
    steps_taken: usize,
    par: Par,
    stack: &mut MemStack,
) -> Result<(), LanczosError> {
    if steps_taken != y_k.nrows() {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k".to_string(),
            expected: steps_taken,
            actual: y_k.nrows(),
        }
        .into());
    }

    let zero_threshold = breakdown_tolerance();
    let b_norm = ws.b_norm();
    if b_norm <= zero_threshold {
        return Err(LanczosErrorKind::ZeroInputVector.into());
    }

    let (v_prev, v_curr, work, alphas, betas, x_k) = ws.parts_mut();

    if steps_taken == 0 {
        zip!(x_k.as_mut()).for_each(|unzip!(x_i)| *x_i = 0.0);
        return Ok(());
    }

    // Seed: v_1 = b / ||b||, v_0 = 0.
    let inv_norm = b_norm.recip();
    zip!(v_prev.as_mut()).for_each(|unzip!(v_i)| *v_i = 0.0);
    zip!(v_curr.as_mut(), b).for_each(|unzip!(v_i, b_i)| *v_i = *b_i * inv_norm);

    // x_k = y_0 * v_1.
    let y0 = y_k[(0, 0)];
    zip!(x_k.as_mut(), v_curr.as_ref()).for_each(|unzip!(x_i, v_i)| *x_i = y0 * *v_i);

    for j in 0..steps_taken.saturating_sub(1) {
        let alpha_j = alphas[j];
        let beta_j = betas[j];
        let beta_prev = if j == 0 { 0.0 } else { betas[j - 1] };

        reconstruct_step(
            operator,
            work.as_mut(),
            v_curr.as_ref(),
            v_prev.as_ref(),
            alpha_j,
            beta_prev,
            par,
            stack,
        );

        let inv_beta = beta_j.recip();
        zip!(work.as_mut()).for_each(|unzip!(w_i)| {
            *w_i *= inv_beta;
        });

        // x_k += coeff * work  (SIMD via matvec_colmajor fast-path)
        let coeff = y_k[(j + 1, 0)];
        let one_val = 1.0_f64;
        let one_mat = unsafe { MatRef::<f64>::from_raw_parts(&one_val, 1, 1, 1, 1) };
        matmul(x_k.as_mut(), Accum::Add, work.as_ref(), one_mat, coeff, par);

        core::mem::swap(v_prev, v_curr);
        core::mem::swap(v_curr, work);
    }

    Ok(())
}
