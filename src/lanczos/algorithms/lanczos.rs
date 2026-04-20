//! One-pass symmetric Lanczos recurrence that materializes the basis `V_k`
//! in an `n x k` dense matrix owned by the caller-supplied workspace.
//! Called by [`crate::lanczos::solvers::lanczos_into`].

use super::{LanczosError, LanczosIteration, Reorthogonalization, breakdown_tolerance};
use crate::lanczos::solvers::LanczosWorkspace;
use faer::{
    Conj, Par, dyn_stack::MemStack, linalg::matmul::dot::inner_prod, matrix_free::LinOp, prelude::*,
};

/// Runs up to `k` steps of symmetric Lanczos with the basis stored in full
/// inside `ws.v_k`. Writes the scalar tridiagonal into `ws.alphas`/`ws.betas`
/// and returns the number of steps actually taken (may be less than `k` on
/// breakdown).
///
/// The caller has already validated `b.nrows() == ws.n()` and
/// `k <= ws.k_cap()` in [`crate::lanczos::solvers::lanczos_into`].
///
/// # Errors
/// Returns [`LanczosError`] if `b` is the zero vector.
pub(crate) fn lanczos_one_pass_into<O: LinOp<f64>>(
    ws: &mut LanczosWorkspace,
    operator: &O,
    b: MatRef<'_, f64>,
    k: usize,
    par: Par,
    reorthog: Reorthogonalization,
    stack: &mut MemStack,
) -> Result<usize, LanczosError> {
    let tolerance = breakdown_tolerance();
    // Cache `||b||` in the workspace so `lanczos_into` can reuse it for the
    // final `x_k = ||b|| V_k g` scaling without a second O(n) sweep.
    let b_norm = b.norm_l2();
    ws.set_b_norm(b_norm);

    // The one-pass driver never touches `x_k`; the high-level driver writes
    // to it after this function returns.
    let (v_k, v_prev, v_curr, work, alphas, betas, _) = ws.parts_mut();

    alphas.clear();
    betas.clear();

    if k == 0 {
        return Ok(0);
    }

    let mut lanczos_iter =
        LanczosIteration::new_borrowed(operator, v_prev, v_curr, work, b, b_norm, k, par)?;

    // v_1 = b / ||b||, stash into the first basis column.
    v_k.col_mut(0).copy_from(lanczos_iter.v_curr().col(0));

    let mut steps_taken = 0usize;

    for i in 0..k {
        // DGKS Full reortho is applied to the unnormalized residual `w` held
        // inside the iterator, so `beta = ||w_post||` feeds one coherent
        // coupling element into both the stored tridiagonal and the
        // iterator's `beta_prev`. Splitting the correction across a
        // post-normalization sweep would break this invariant.
        let step_opt = match reorthog {
            Reorthogonalization::None => lanczos_iter.next_step(stack),
            Reorthogonalization::Full => {
                // Split borrow: the closure only needs an immutable view of
                // the basis columns written so far. Rebinding through an
                // immutable reference taken before the mutable iter call is
                // unsound (iter holds exclusive borrows of v_prev/v_curr/work
                // which are disjoint from v_k), so we form the slice each
                // iteration from `v_k` directly.
                let prev_basis = v_k.as_ref().get(.., 0..(i + 1));
                lanczos_iter.next_step_with(stack, |mut w| {
                    for _ in 0..2 {
                        for col_idx in 0..prev_basis.ncols() {
                            let v_col = prev_basis.col(col_idx);
                            let h = inner_prod(
                                v_col.transpose(),
                                Conj::No,
                                w.as_ref().col(0),
                                Conj::No,
                            );
                            zip!(w.as_mut(), v_col.as_mat()).for_each(|unzip!(w_i, u_i)| {
                                *w_i -= h * *u_i;
                            });
                        }
                    }
                })
            }
        };

        let Some(step) = step_opt else { break };

        alphas.push(step.alpha);
        steps_taken += 1;

        if step.beta <= tolerance {
            break;
        }

        if i < k - 1 {
            betas.push(step.beta);
            v_k.col_mut(i + 1).copy_from(lanczos_iter.v_curr().col(0));
        }
    }

    Ok(steps_taken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lanczos::algorithms::LanczosDecomposition;
    use faer::Accum;
    use faer::dyn_stack::MemBuffer;
    use faer::sparse::{SparseColMat, Triplet, linalg::matmul::sparse_dense_matmul};

    /// Test-only bundle that mirrors the fields the DGKS stress and
    /// recurrence tests need to inspect after a run. Built by `run_lanczos`
    /// from the workspace so the assertions can own their data.
    struct LanczosTestResult {
        v_k: Mat<f64>,
        decomposition: LanczosDecomposition,
    }

    fn diagonal(diag: &[f64]) -> SparseColMat<u32, f64> {
        let n = diag.len();
        let triplets: Vec<_> = diag
            .iter()
            .enumerate()
            .map(|(i, &d)| Triplet::new(i as u32, i as u32, d))
            .collect();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    fn laplacian_1d(n: usize) -> SparseColMat<u32, f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i as u32, i as u32, 2.0));
            if i + 1 < n {
                triplets.push(Triplet::new(i as u32, (i + 1) as u32, -1.0));
                triplets.push(Triplet::new((i + 1) as u32, i as u32, -1.0));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    fn sinusoidal_rhs(n: usize) -> faer::Mat<f64> {
        faer::Mat::<f64>::from_fn(n, 1, |i, _| ((i + 1) as f64).sin())
    }

    fn run_lanczos(
        a: &SparseColMat<u32, f64>,
        b: MatRef<'_, f64>,
        k: usize,
        reorthog: Reorthogonalization,
    ) -> LanczosTestResult {
        let scratch = a.as_ref().apply_scratch(1, Par::Seq);
        let mut mem = MemBuffer::new(scratch);
        let stack = MemStack::new(&mut mem);
        let mut ws = LanczosWorkspace::new(a.nrows(), k);
        let steps_taken =
            lanczos_one_pass_into(&mut ws, &a.as_ref(), b, k, Par::Seq, reorthog, stack)
                .expect("Lanczos failed");

        // Clone only the valid prefix of V_k, alphas, and betas so the
        // assertions can own their data and ws goes out of scope cleanly.
        let b_norm = b.norm_l2();
        let parts = ws.parts_mut();
        let v_k_trimmed = parts.0.as_ref().get(.., 0..steps_taken).to_owned();
        let alphas = parts.4.clone();
        let betas = parts.5.clone();
        LanczosTestResult {
            v_k: v_k_trimmed,
            decomposition: LanczosDecomposition {
                alphas,
                betas,
                steps_taken,
                b_norm,
            },
        }
    }

    /// Frobenius norm of `V_k^T V_k - I`, measuring loss of orthogonality.
    fn orthonormality_error(v_k: &faer::Mat<f64>) -> f64 {
        let k = v_k.ncols();
        let vtv = v_k.transpose() * v_k;
        let mut err_sq = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                let d = vtv[(i, j)] - expected;
                err_sq += d * d;
            }
        }
        err_sq.sqrt()
    }

    /// Worst-case L2 residual of `A v_j - beta_{j-1} v_{j-1} - alpha_j v_j - beta_j v_{j+1}`
    /// over all interior steps `0 <= j < k-1`.
    fn recurrence_error(
        a: &SparseColMat<u32, f64>,
        v_k: &faer::Mat<f64>,
        alphas: &[f64],
        betas: &[f64],
    ) -> f64 {
        let n = v_k.nrows();
        let k = v_k.ncols();
        let mut worst = 0.0_f64;
        for j in 0..k - 1 {
            let mut av_j = faer::Mat::<f64>::zeros(n, 1);
            sparse_dense_matmul(
                av_j.as_mut(),
                Accum::Replace,
                a.as_ref(),
                v_k.as_ref().get(.., j..j + 1),
                1.0,
                Par::Seq,
            );
            let mut err_sq = 0.0_f64;
            for row in 0..n {
                let mut rhs = alphas[j] * v_k[(row, j)] + betas[j] * v_k[(row, j + 1)];
                if j > 0 {
                    rhs += betas[j - 1] * v_k[(row, j - 1)];
                }
                let d = av_j[(row, 0)] - rhs;
                err_sq += d * d;
            }
            worst = worst.max(err_sq.sqrt());
        }
        worst
    }

    /// Diagonal SPD matrix with eigenvalues 1..n is the canonical stress case
    /// for Lanczos. `Reorthogonalization::Full` must bring `||V^T V - I||_F`
    /// back to the 1e-14 regime and be many orders of magnitude better than
    /// the unreorthogonalized run.
    #[test]
    fn full_reortho_restores_orthogonality_under_stress() {
        let n = 200;
        let k = 150;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let a = diagonal(&diag);
        let b = sinusoidal_rhs(n);

        let none = run_lanczos(&a, b.as_ref(), k, Reorthogonalization::None);
        let full = run_lanczos(&a, b.as_ref(), k, Reorthogonalization::Full);

        let none_err = orthonormality_error(&none.v_k);
        let full_err = orthonormality_error(&full.v_k);
        eprintln!("diag(1..{n}) k={k}: ||V^T V - I||_F  None={none_err:.3e}  Full={full_err:.3e}");

        assert!(
            full_err < 1e-11,
            "Full reortho: ||V^T V - I||_F = {full_err:.3e} > 1e-11"
        );
        assert!(
            full_err * 1e4 < none_err,
            "Full reortho ({full_err:.3e}) not significantly better than None ({none_err:.3e})"
        );
    }

    /// With Full reortho the three-term recurrence must hold at every
    /// interior step: catches any drift between the coupling beta stored
    /// in `T_k` and the basis vector written to `V_k`.
    #[test]
    fn full_reortho_three_term_recurrence() {
        let n = 60;
        let k = 40;
        let a = laplacian_1d(n);
        let b = sinusoidal_rhs(n);
        let result = run_lanczos(&a, b.as_ref(), k, Reorthogonalization::Full);

        let err = recurrence_error(
            &a,
            &result.v_k,
            &result.decomposition.alphas,
            &result.decomposition.betas,
        );
        assert!(err < 1e-12, "recurrence error = {err:.3e}");
    }

    /// Without reortho the recurrence still holds at every step because
    /// `betas` comes directly from the iterator's internal beta.
    #[test]
    fn no_reortho_three_term_recurrence() {
        let n = 60;
        let k = 40;
        let a = laplacian_1d(n);
        let b = sinusoidal_rhs(n);
        let result = run_lanczos(&a, b.as_ref(), k, Reorthogonalization::None);

        let err = recurrence_error(
            &a,
            &result.v_k,
            &result.decomposition.alphas,
            &result.decomposition.betas,
        );
        assert!(err < 1e-12, "recurrence error = {err:.3e}");
    }

    /// Zero-allocation regression for the one-pass hot path. Once the
    /// workspace is built, a single `lanczos_into` call must not perform
    /// any kernel-owned heap allocation, including the projected solve for
    /// `exp(-T_k)e_1`. In particular, no allocation may scale with `n`:
    /// the bytes counted between `n = 1_000` and `n = 10_000` (at the same
    /// `k`) must match to within a small fudge.
    ///
    /// Under `cargo test --lib` the counting allocator sees every other
    /// test thread's allocations, so the absolute cap is generous. The
    /// invariance check (`delta_large - delta_small`) is the real signal
    /// because concurrent noise affects both measurements symmetrically.
    #[test]
    #[ignore = "global counting allocator sees concurrent test noise; run isolated: cargo test hot_path_is_allocation_free -- --ignored"]
    fn hot_path_is_allocation_free_regardless_of_n() {
        use crate::lanczos::alloc_counter;
        use crate::lanczos::solvers::{LanczosWorkspace, lanczos_into};
        use crate::lanczos::{ProjectedTridiagonalWorkspace, Reorthogonalization};

        fn measure_one_call(n: usize, k: usize) -> u64 {
            let diag_vec: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let a = diagonal(&diag_vec);
            let b = sinusoidal_rhs(n);
            let mut ws = LanczosWorkspace::new(n, k);
            let mut projected = ProjectedTridiagonalWorkspace::new(k, Par::Seq);
            let scratch = a.as_ref().apply_scratch(1, Par::Seq);
            let mut mem = MemBuffer::new(scratch);

            // Warmup call: exercises every lazy-initialized scratch path
            // once (faer projected eigensolver inside
            // `ProjectedTridiagonalWorkspace`, etc.) so the
            // measured second call sees only steady-state allocations.
            {
                let stack = MemStack::new(&mut mem);
                lanczos_into(
                    &mut ws,
                    &a.as_ref(),
                    b.as_ref(),
                    k,
                    Par::Seq,
                    Reorthogonalization::None,
                    stack,
                    |alphas, betas, out| projected.exp_neg_tk(alphas, betas, out),
                )
                .expect("warmup lanczos_into failed");
            }

            alloc_counter::reset();
            {
                let stack = MemStack::new(&mut mem);
                lanczos_into(
                    &mut ws,
                    &a.as_ref(),
                    b.as_ref(),
                    k,
                    Par::Seq,
                    Reorthogonalization::None,
                    stack,
                    |alphas, betas, out| projected.exp_neg_tk(alphas, betas, out),
                )
                .expect("measured lanczos_into failed");
            }
            let (bytes, _count) = alloc_counter::snapshot();
            bytes
        }

        let k = 20;
        let delta_small = measure_one_call(1_000, k);
        let delta_large = measure_one_call(10_000, k);
        eprintln!(
            "hot_path_is_allocation_free: n=1000 delta={delta_small} bytes, \
             n=10000 delta={delta_large} bytes"
        );

        // With a dedicated projected-solve workspace, the measured call must
        // not perform any kernel-owned heap allocation. Keep a 1 KiB cap to
        // absorb incidental noise from the test harness when run isolated.
        let absolute_cap = 1024;
        assert!(
            delta_small < absolute_cap,
            "n=1000 hot path allocated {delta_small} bytes, expected < {absolute_cap}"
        );
        assert!(
            delta_large < absolute_cap,
            "n=10000 hot path allocated {delta_large} bytes, expected < {absolute_cap}"
        );

        // Invariance: any O(n) allocation in the hot path would push the
        // growth above ~72 KB (one `Mat::zeros(n)` for n=10000 minus the
        // same for n=1000). 64 KB is tight enough to detect that while
        // tolerating residual noise from concurrent tests.
        let growth = delta_large.saturating_sub(delta_small);
        let growth_cap = 64 * 1024;
        assert!(
            growth < growth_cap,
            "hot path allocation grew by {growth} bytes from n=1000 to \
             n=10000, expected < {growth_cap}; the kernel appears to \
             allocate O(n)"
        );
    }
}
