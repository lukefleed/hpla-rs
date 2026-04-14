//! Cross-backend equivalence tests.
//!
//! Each test iterates over the matrices in `matrices/` and checks that every
//! backend produces the same output as the faer reference, up to a relative
//! L2 tolerance. The tests use the public FFI surface and are gated by
//! `#[cfg(test)]` at the crate root.

use super::*;

/// Computes the relative L2 error between two vectors.
///
/// Returns `||actual - reference||_2 / ||reference||_2`.
/// If the reference norm is zero, returns the absolute norm of the
/// difference instead.
fn relative_l2_error(actual: &[f64], reference: &[f64]) -> f64 {
    let diff_norm: f64 = actual
        .iter()
        .zip(reference)
        .map(|(a, r)| (a - r).powi(2))
        .sum::<f64>()
        .sqrt();
    let ref_norm: f64 = reference.iter().map(|r| r.powi(2)).sum::<f64>().sqrt();
    if ref_norm == 0.0 {
        diff_norm
    } else {
        diff_norm / ref_norm
    }
}

/// Checks all backends produce the same y = A*x (relative L2 error).
/// Runs on every .mtx in matrices/.
#[test]
fn test_backend_numerical_equivalence() -> anyhow::Result<()> {
    let mut matrices: Vec<_> = std::fs::read_dir("matrices")?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "mtx"))
        .map(|entry| entry.path())
        .collect();
    matrices.sort();

    anyhow::ensure!(!matrices.is_empty(), "no .mtx files found in matrices/");

    let tol = 1e-4;

    for path in &matrices {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("\n=== {name} ===");

        let raw = load_mtx_raw(path).map_err(|e| anyhow::anyhow!("load_mtx_raw({name}): {e}"))?;

        // --- Faer reference (CSC) ---
        let a_faer = SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
            .map_err(|e| anyhow::anyhow!("faer SparseColMat({name}): {e:?}"))?;
        let x_faer: Col<f64> = Col::from_fn(raw.ncols, |_| 1.0);
        let mut y_faer: Col<f64> = Col::zeros(raw.nrows);
        spmv_faer(&a_faer, &x_faer, &mut y_faer);

        let faer_ref: Vec<f64> = (0..raw.nrows).map(|i| y_faer[i]).collect();
        let faer_norm: f64 = faer_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            faer_norm > 0.0,
            "{name}: faer y is all zeros — matrix may be empty"
        );

        // --- Faer CSR ---
        {
            let a_faer_csr =
                SparseRowMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
                    .map_err(|e| anyhow::anyhow!("faer SparseRowMat({name}): {e:?}"))?;
            let mut y_csr: Col<f64> = Col::zeros(raw.nrows);
            spmv_faer_csr(&a_faer_csr, &x_faer, &mut y_csr);
            let csr_result: Vec<f64> = (0..raw.nrows).map(|i| y_csr[i]).collect();
            let err = relative_l2_error(&csr_result, &faer_ref);
            eprintln!("  faer/csr:         rel L2 = {err:.2e}");
            assert!(err < tol, "{name}: faer/csr diverged: rel L2 = {err:.2e}");
        }

        // --- PETSc (CSR, inodes disabled) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::petsc::libpetsc_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                    1, // disable inodes
                );
                crate::petsc::libpetsc_spmv_execute(ctx);
                crate::petsc::libpetsc_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::petsc::libpetsc_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  petsc/csr_raw:    rel L2 = {err:.2e}");
            assert!(
                err < tol,
                "{name}: petsc/csr_raw diverged: rel L2 = {err:.2e}"
            );
        }

        // --- PETSc (CSR, inodes enabled) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::petsc::libpetsc_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                    0, // enable inodes
                );
                crate::petsc::libpetsc_spmv_execute(ctx);
                crate::petsc::libpetsc_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::petsc::libpetsc_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  petsc/csr_inodes: rel L2 = {err:.2e}");
            assert!(
                err < tol,
                "{name}: petsc/csr_inodes diverged: rel L2 = {err:.2e}"
            );
        }

        // --- Eigen (CSC) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::eigen::libeigen_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.col_ptr.as_ptr(),
                    raw.row_idx.as_ptr(),
                    raw.csc_values.as_ptr(),
                );
                crate::eigen::libeigen_spmv_execute(ctx);
                crate::eigen::libeigen_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::eigen::libeigen_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  eigen/csc_map:    rel L2 = {err:.2e}");
            assert!(
                err < tol,
                "{name}: eigen/csc_map diverged: rel L2 = {err:.2e}"
            );
        }

        // --- Eigen (CSR, cross-format control) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::eigen::libeigen_csr_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                );
                crate::eigen::libeigen_csr_spmv_execute(ctx);
                crate::eigen::libeigen_csr_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::eigen::libeigen_csr_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  eigen/csr_map:    rel L2 = {err:.2e}");
            assert!(
                err < tol,
                "{name}: eigen/csr_map diverged: rel L2 = {err:.2e}"
            );
        }

        // --- MKL (CSR, Inspection-Execution) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::mkl::libmkl_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                );
                crate::mkl::libmkl_spmv_execute(ctx);
                crate::mkl::libmkl_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::mkl::libmkl_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  mkl/csr_ie:       rel L2 = {err:.2e}");
            assert!(err < tol, "{name}: mkl/csr_ie diverged: rel L2 = {err:.2e}");
        }

        // --- MKL (CSC, Inspection-Execution, cross-format control) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::mkl::libmkl_csc_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.col_ptr.as_ptr(),
                    raw.row_idx.as_ptr(),
                    raw.csc_values.as_ptr(),
                );
                crate::mkl::libmkl_csc_spmv_execute(ctx);
                crate::mkl::libmkl_csc_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::mkl::libmkl_csc_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  mkl/csc_ie:       rel L2 = {err:.2e}");
            assert!(err < tol, "{name}: mkl/csc_ie diverged: rel L2 = {err:.2e}");
        }

        // --- PSBLAS (CSR) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::psblas::libpsblas_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                );
                crate::psblas::libpsblas_spmv_execute(ctx);
                crate::psblas::libpsblas_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::psblas::libpsblas_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  psblas/csr:       rel L2 = {err:.2e}");
            assert!(err < tol, "{name}: psblas/csr diverged: rel L2 = {err:.2e}");
        }

        // --- PSBLAS (CSC) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = crate::psblas::libpsblas_csc_spmv_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.col_ptr.as_ptr(),
                    raw.row_idx.as_ptr(),
                    raw.csc_values.as_ptr(),
                );
                crate::psblas::libpsblas_spmv_execute(ctx);
                crate::psblas::libpsblas_spmv_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                crate::psblas::libpsblas_spmv_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  psblas/csc:       rel L2 = {err:.2e}");
            assert!(err < tol, "{name}: psblas/csc diverged: rel L2 = {err:.2e}");
        }
    }

    eprintln!(
        "\nAll backends match Faer reference across {} matrices.",
        matrices.len()
    );
    Ok(())
}

/// Checks that the Eigen two-pass Lanczos produces the same exp(-A)b
/// as the Faer reference. Runs on every symmetric .mtx in matrices/.
#[test]
fn test_lanczos_two_pass_backend_equivalence() -> anyhow::Result<()> {
    use faer::Par;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::matrix_free::LinOp;

    use crate::eigen::{
        libeigen_csc_lanczos_two_pass_execute, libeigen_csc_lanczos_two_pass_get_y,
        libeigen_csc_lanczos_two_pass_setup, libeigen_csc_lanczos_two_pass_teardown,
        libeigen_lanczos_two_pass_execute, libeigen_lanczos_two_pass_get_y,
        libeigen_lanczos_two_pass_setup, libeigen_lanczos_two_pass_teardown,
    };
    use crate::lanczos::{
        KRYLOV_HARD_LIMIT, KRYLOV_MARGIN, SAAD_TOL, SPECTRAL_PROBE_STEPS, adaptive_krylov_dim,
        deterministic_rhs, estimate_spectral_radius, exp_neg_tk, lanczos_two_pass,
    };
    // Temporarily disabled while ffi/lanczos/psblas_lanczos_two_pass.f90 is WIP.
    // use crate::psblas::{
    //     libpsblas_lanczos_two_pass_execute, libpsblas_lanczos_two_pass_get_y,
    //     libpsblas_lanczos_two_pass_setup, libpsblas_lanczos_two_pass_teardown,
    // };

    let tol = 1e-8;
    let mut checked = 0;

    for name in crate::LANCZOS_SUITE {
        let path = std::path::PathBuf::from(format!("matrices/{name}.mtx"));
        if !path.exists() {
            eprintln!(
                "  {name}: skipped (matrices/{name}.mtx not present; \
                 run `bash download_matrices.sh`)"
            );
            continue;
        }
        checked += 1;
        eprintln!("\n=== {name} (Lanczos) ===");

        let mut raw =
            load_mtx_raw(&path).map_err(|e| anyhow::anyhow!("load_mtx_raw({name}): {e}"))?;

        let b_vec = deterministic_rhs(raw.nrows);

        // Absorb tau = -1/rho into the matrix per Saad 1992.
        // Probe rho via a short Lanczos run on the unscaled matrix,
        // then divide the RawMatrix values so exp(-raw)v is numerically
        // well-posed and the Saad estimator is meaningful.
        let scale = {
            let a_tmp = SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
                .map_err(|e| anyhow::anyhow!("faer SparseColMat({name}): {e:?}"))?;
            let b_tmp = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);
            let scratch_req = a_tmp.as_ref().apply_scratch(1, Par::Seq);
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(
                &a_tmp.as_ref(),
                b_tmp.as_ref(),
                SPECTRAL_PROBE_STEPS,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: spectral radius probe failed: {e}"))?
        };
        crate::scale_values(&mut raw, scale);
        eprintln!("  rho(A) = {scale:.3e}");

        let a_faer = SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
            .map_err(|e| anyhow::anyhow!("faer SparseColMat({name}): {e:?}"))?;

        let b_mat = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);

        let scratch_req = a_faer.as_ref().apply_scratch(1, Par::Seq);

        // Adaptive Krylov dimension: probe the spectral radius via a short
        // Lanczos run, then cap max_k at (ceil(rho) + margin) or the hard
        // limit, whichever is smaller. Mirrors the bench exactly.
        let spectral_radius = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                SPECTRAL_PROBE_STEPS,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: estimate_spectral_radius failed: {e}"))?
        };
        let max_k = ((spectral_radius.ceil() as usize) + KRYLOV_MARGIN).min(KRYLOV_HARD_LIMIT);

        let krylov_dim = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            let (m, _) = adaptive_krylov_dim(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                max_k,
                SAAD_TOL,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: adaptive_krylov_dim failed: {e}"))?;
            m.max(1)
        };
        eprintln!("  rho~{spectral_radius:.1}, max_k={max_k}, krylov_dim = {krylov_dim}");

        // --- Faer reference ---
        let faer_ref: Vec<f64> = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            let result = lanczos_two_pass(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                krylov_dim,
                Par::Seq,
                stack,
                exp_neg_tk,
            )
            .map_err(|e| anyhow::anyhow!("{name}: faer lanczos_two_pass failed: {e}"))?;
            (0..raw.nrows).map(|i| result[(i, 0)]).collect()
        };

        let faer_norm: f64 = faer_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(faer_norm > 0.0, "{name}: faer Lanczos result is all zeros");

        // --- Eigen ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = libeigen_lanczos_two_pass_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.row_ptr.as_ptr(),
                    raw.col_idx.as_ptr(),
                    raw.values.as_ptr(),
                    b_vec.as_ptr(),
                    krylov_dim as i32,
                );
                if ctx.is_null() {
                    eprintln!("  eigen/two_pass:   skipped (stub returned null)");
                } else {
                    libeigen_lanczos_two_pass_execute(ctx);
                    libeigen_lanczos_two_pass_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                    libeigen_lanczos_two_pass_teardown(ctx);

                    let err = relative_l2_error(&y_buf, &faer_ref);
                    eprintln!("  eigen/two_pass:   rel L2 = {err:.2e}");
                    assert!(
                        err < tol,
                        "{name}: eigen/two_pass diverged: rel L2 = {err:.2e}"
                    );
                }
            }
        }

        // --- Eigen CSC (cross-format control) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
            unsafe {
                let ctx = libeigen_csc_lanczos_two_pass_setup(
                    raw.nrows as i32,
                    raw.ncols as i32,
                    raw.nnz as i32,
                    raw.col_ptr.as_ptr(),
                    raw.row_idx.as_ptr(),
                    raw.csc_values.as_ptr(),
                    b_vec.as_ptr(),
                    krylov_dim as i32,
                );
                if ctx.is_null() {
                    eprintln!("  eigen_csc/two_pass: skipped (stub returned null)");
                } else {
                    libeigen_csc_lanczos_two_pass_execute(ctx);
                    libeigen_csc_lanczos_two_pass_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                    libeigen_csc_lanczos_two_pass_teardown(ctx);

                    let err = relative_l2_error(&y_buf, &faer_ref);
                    eprintln!("  eigen_csc/two_pass: rel L2 = {err:.2e}");
                    assert!(
                        err < tol,
                        "{name}: eigen_csc/two_pass diverged: rel L2 = {err:.2e}"
                    );
                }
            }
        }

        // PSBLAS two-pass temporarily disabled while
        // ffi/lanczos/psblas_lanczos_two_pass.f90 is WIP.
        // {
        //     let mut y_buf = vec![0.0f64; raw.nrows];
        //     unsafe {
        //         let ctx = libpsblas_lanczos_two_pass_setup(
        //             raw.nrows as i32,
        //             raw.ncols as i32,
        //             raw.nnz as i32,
        //             raw.row_ptr.as_ptr(),
        //             raw.col_idx.as_ptr(),
        //             raw.values.as_ptr(),
        //             b_vec.as_ptr(),
        //             krylov_dim as i32,
        //         );
        //         if ctx.is_null() {
        //             eprintln!("  psblas/two_pass:  skipped (stub returned null)");
        //             continue;
        //         }
        //         libpsblas_lanczos_two_pass_execute(ctx);
        //         libpsblas_lanczos_two_pass_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
        //         libpsblas_lanczos_two_pass_teardown(ctx);
        //     }
        //     let err = relative_l2_error(&y_buf, &faer_ref);
        //     eprintln!("  psblas/two_pass:  rel L2 = {err:.2e}");
        //     assert!(
        //         err < tol,
        //         "{name}: psblas/two_pass diverged: rel L2 = {err:.2e}"
        //     );
        // }
    }

    anyhow::ensure!(
        checked > 0,
        "no Lanczos matrices available; run download_matrices.sh"
    );
    eprintln!("\nTwo-pass Lanczos Eigen backend matches Faer reference across {checked} matrices.");
    // Temporarily disabled while ffi/lanczos/psblas_lanczos_two_pass.f90 is WIP.
    // eprintln!("\nTwo-pass Lanczos backends match Faer reference across {checked} matrices.");
    Ok(())
}

/// Checks that the PSBLAS one-pass Lanczos produces the same exp(-A)b
/// as the Faer one-pass reference. Runs on every matrix in
/// [`crate::LANCZOS_SUITE`] that is present on disk. Skips gracefully if
/// the PSBLAS stub returns a null context.
#[test]
fn test_lanczos_backend_equivalence() -> anyhow::Result<()> {
    use faer::Par;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::matrix_free::LinOp;

    use crate::eigen::{
        libeigen_csc_lanczos_execute, libeigen_csc_lanczos_get_y, libeigen_csc_lanczos_setup,
        libeigen_csc_lanczos_teardown, libeigen_lanczos_execute, libeigen_lanczos_get_y,
        libeigen_lanczos_setup, libeigen_lanczos_teardown,
    };
    use crate::lanczos::{
        KRYLOV_HARD_LIMIT, KRYLOV_MARGIN, Reorthogonalization, SAAD_TOL, SPECTRAL_PROBE_STEPS,
        adaptive_krylov_dim, deterministic_rhs, estimate_spectral_radius, exp_neg_tk, lanczos,
    };
    use crate::psblas::{
        libpsblas_lanczos_execute, libpsblas_lanczos_get_y, libpsblas_lanczos_setup,
        libpsblas_lanczos_teardown,
    };

    let tol = 1e-8;
    let mut checked = 0;

    for name in crate::LANCZOS_SUITE {
        let path = std::path::PathBuf::from(format!("matrices/{name}.mtx"));
        if !path.exists() {
            eprintln!(
                "  {name}: skipped (matrices/{name}.mtx not present; \
                 run `bash download_matrices.sh`)"
            );
            continue;
        }
        checked += 1;
        eprintln!("\n=== {name} (one-pass Lanczos) ===");

        let mut raw =
            load_mtx_raw(&path).map_err(|e| anyhow::anyhow!("load_mtx_raw({name}): {e}"))?;

        let b_vec = deterministic_rhs(raw.nrows);

        // Absorb tau = -1/rho into the matrix per Saad 1992.
        // Probe rho via a short Lanczos run on the unscaled matrix,
        // then divide the RawMatrix values so exp(-raw)v is numerically
        // well-posed and the Saad estimator is meaningful.
        let scale = {
            let a_tmp = SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
                .map_err(|e| anyhow::anyhow!("faer SparseColMat({name}): {e:?}"))?;
            let b_tmp = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);
            let scratch_req = a_tmp.as_ref().apply_scratch(1, Par::Seq);
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(
                &a_tmp.as_ref(),
                b_tmp.as_ref(),
                SPECTRAL_PROBE_STEPS,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: spectral radius probe failed: {e}"))?
        };
        crate::scale_values(&mut raw, scale);
        eprintln!("  rho(A) = {scale:.3e}");

        let a_faer = SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
            .map_err(|e| anyhow::anyhow!("faer SparseColMat({name}): {e:?}"))?;

        let b_mat = faer::Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);

        let scratch_req = a_faer.as_ref().apply_scratch(1, Par::Seq);

        // Adaptive Krylov dimension: probe the spectral radius via a short
        // Lanczos run, then cap max_k at (ceil(rho) + margin) or the hard
        // limit, whichever is smaller. Mirrors the bench exactly so both
        // variants run on the same number of Lanczos iterations per matrix.
        let spectral_radius = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            estimate_spectral_radius(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                SPECTRAL_PROBE_STEPS,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: estimate_spectral_radius failed: {e}"))?
        };
        let max_k = ((spectral_radius.ceil() as usize) + KRYLOV_MARGIN).min(KRYLOV_HARD_LIMIT);

        let krylov_dim = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            let (m, _) = adaptive_krylov_dim(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                max_k,
                SAAD_TOL,
                Par::Seq,
                stack,
            )
            .map_err(|e| anyhow::anyhow!("{name}: adaptive_krylov_dim failed: {e}"))?;
            m.max(1)
        };
        eprintln!("  rho~{spectral_radius:.1}, max_k={max_k}, krylov_dim = {krylov_dim}");

        // --- Faer reference (one-pass exp(-A)b) ---
        let faer_ref: Vec<f64> = {
            let mut mem = MemBuffer::new(scratch_req);
            let stack = MemStack::new(&mut mem);
            let result = lanczos(
                &a_faer.as_ref(),
                b_mat.as_ref(),
                krylov_dim,
                Par::Seq,
                Reorthogonalization::None,
                stack,
                exp_neg_tk,
            )
            .map_err(|e| anyhow::anyhow!("{name}: faer lanczos failed: {e}"))?;
            (0..raw.nrows).map(|i| result[(i, 0)]).collect()
        };

        let faer_norm: f64 = faer_ref.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(faer_norm > 0.0, "{name}: faer Lanczos result is all zeros");

        // --- Eigen CSR ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
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
                if ctx.is_null() {
                    eprintln!("  eigen/one_pass:    skipped (stub returned null)");
                } else {
                    libeigen_lanczos_execute(ctx);
                    libeigen_lanczos_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                    libeigen_lanczos_teardown(ctx);

                    let err = relative_l2_error(&y_buf, &faer_ref);
                    eprintln!("  eigen/one_pass:    rel L2 = {err:.2e}");
                    assert!(
                        err < tol,
                        "{name}: eigen/one_pass diverged: rel L2 = {err:.2e}"
                    );
                }
            }
        }

        // --- Eigen CSC (cross-format control) ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
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
                if ctx.is_null() {
                    eprintln!("  eigen_csc/one_pass: skipped (stub returned null)");
                } else {
                    libeigen_csc_lanczos_execute(ctx);
                    libeigen_csc_lanczos_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                    libeigen_csc_lanczos_teardown(ctx);

                    let err = relative_l2_error(&y_buf, &faer_ref);
                    eprintln!("  eigen_csc/one_pass: rel L2 = {err:.2e}");
                    assert!(
                        err < tol,
                        "{name}: eigen_csc/one_pass diverged: rel L2 = {err:.2e}"
                    );
                }
            }
        }

        // --- PSBLAS ---
        {
            let mut y_buf = vec![0.0f64; raw.nrows];
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
                if ctx.is_null() {
                    eprintln!("  psblas/one_pass:  skipped (stub returned null)");
                    continue;
                }
                libpsblas_lanczos_execute(ctx);
                libpsblas_lanczos_get_y(ctx, y_buf.as_mut_ptr(), raw.nrows as i32);
                libpsblas_lanczos_teardown(ctx);
            }
            let err = relative_l2_error(&y_buf, &faer_ref);
            eprintln!("  psblas/one_pass:  rel L2 = {err:.2e}");
            assert!(
                err < tol,
                "{name}: psblas/one_pass diverged: rel L2 = {err:.2e}"
            );
        }
    }

    anyhow::ensure!(
        checked > 0,
        "no Lanczos matrices available; run download_matrices.sh"
    );
    eprintln!("\nOne-pass Lanczos backends match Faer reference across {checked} matrices.");
    Ok(())
}
