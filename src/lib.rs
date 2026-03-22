//! Core library module handling matrix I/O and Faer SpMV kernels.
//!
//! Provides the data structures required for Matrix Market file parsing
//! and the foundational `spmv_faer` benchmark kernel.

pub mod eigen;
pub mod mkl;
pub mod petsc;
pub mod psblas;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::PathBuf;

use faer::col::Col;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Accum, Par};
use matrix_market_rs::MtxData;

pub struct RawMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub values: Vec<f64>,
    pub col_ptr: Vec<i32>,
    pub row_idx: Vec<i32>,
    pub csc_values: Vec<f64>,
    pub triplets: Vec<Triplet<u32, u32, f64>>, // For Faer CSC
}

// Helper to determine symmetry reading the raw header
fn detect_symmetry(path: &PathBuf) -> (bool, bool) {
    if let Ok(file) = File::open(path) {
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        if reader.read_line(&mut line).is_ok() {
            let lower = line.to_lowercase();
            if lower.starts_with("%%matrixmarket") {
                let is_skew = lower.contains("skew-symmetric");
                let is_sym =
                    !is_skew && (lower.contains("symmetric") || lower.contains("hermitian"));
                return (is_sym, is_skew);
            }
        }
    }
    (false, false)
}

pub fn load_mtx_raw(path: &PathBuf) -> Result<RawMatrix, String> {
    let (is_symmetric, is_skew) = detect_symmetry(path);
    let data = MtxData::<f64>::from_file(path).map_err(|e| format!("{}", e))?;

    let MtxData::Sparse([nrows, ncols], coords, values, _) = data else {
        return Err("Only sparse matrices supported".into());
    };

    if nrows > u32::MAX as usize || ncols > u32::MAX as usize {
        return Err("Matrix dimensions exceed u32 index limits".into());
    }

    let capacity = if is_symmetric || is_skew {
        coords.len() * 2
    } else {
        coords.len()
    };

    let mut triplets = Vec::with_capacity(capacity);
    let mut row_counts = vec![0; nrows];

    for ([r, c], &v) in coords.iter().zip(values.iter()) {
        let row = *r;
        let col = *c;

        triplets.push(Triplet::new(row as u32, col as u32, v));
        row_counts[row] += 1;

        if (is_symmetric || is_skew) && row != col {
            let val = if is_skew { -v } else { v };
            triplets.push(Triplet::new(col as u32, row as u32, val));
            row_counts[col] += 1;
        }
    }

    let nnz = triplets.len();

    triplets.sort_unstable_by(|a, b| {
        if a.row != b.row {
            a.row.cmp(&b.row)
        } else {
            a.col.cmp(&b.col)
        }
    });

    let mut row_ptr = vec![0i32; nrows + 1];
    let mut col_idx = vec![0i32; nnz];
    let mut csr_values = vec![0.0f64; nnz];

    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i] as i32;
    }

    for (i, t) in triplets.iter().enumerate() {
        col_idx[i] = t.col as i32;
        csr_values[i] = t.val;
    }

    // Create equivalent CSC arrays for Eigen mapping
    let mut csc_triplets = triplets.clone();
    csc_triplets.sort_unstable_by(|a, b| {
        if a.col != b.col {
            a.col.cmp(&b.col)
        } else {
            a.row.cmp(&b.row)
        }
    });

    let mut col_counts = vec![0; ncols];
    for t in &csc_triplets {
        col_counts[t.col as usize] += 1;
    }

    let mut col_ptr = vec![0i32; ncols + 1];
    let mut row_idx = vec![0i32; nnz];
    let mut csc_values = vec![0.0f64; nnz];

    for i in 0..ncols {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i] as i32;
    }

    for (i, t) in csc_triplets.into_iter().enumerate() {
        row_idx[i] = t.row as i32;
        csc_values[i] = t.val;
    }

    Ok(RawMatrix {
        nrows,
        ncols,
        nnz,
        row_ptr,
        col_idx,
        values: csr_values,
        col_ptr,
        row_idx,
        csc_values,
        triplets,
    })
}

#[inline(always)]
pub fn spmv_faer(a: &SparseColMat<u32, f64>, x: &Col<f64>, y: &mut Col<f64>) {
    sparse_dense_matmul(
        y.as_mat_mut(),
        Accum::Add,
        a.as_ref(),
        x.as_mat(),
        1.0,
        Par::Seq,
    );
}

#[cfg(test)]
mod tests {
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
    /// Runs on every .mtx in matrices/. PSBLAS has wider tolerance (1e-6)
    /// because it copies data internally.
    #[test]
    fn test_backend_numerical_equivalence() -> anyhow::Result<()> {
        let mut matrices: Vec<_> = std::fs::read_dir("matrices")?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "mtx")
            })
            .map(|entry| entry.path())
            .collect();
        matrices.sort();

        anyhow::ensure!(!matrices.is_empty(), "no .mtx files found in matrices/");

        // Tolerance for zero-copy backends (PETSc, Eigen, MKL): floating-point
        // reordering under -ffast-math can introduce small differences.
        let tol_strict = 1e-10;
        // PSBLAS assembles its own internal CSR from the input data, so
        // accumulated rounding may be slightly larger.
        let tol_psblas = 1e-6;

        for path in &matrices {
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            eprintln!("\n=== {name} ===");

            let raw = load_mtx_raw(path)
                .map_err(|e| anyhow::anyhow!("load_mtx_raw({name}): {e}"))?;

            // --- Faer reference (CSC) ---
            let a_faer =
                SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
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
                    crate::petsc::libpetsc_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::petsc::libpetsc_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  petsc/csr_raw:    rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
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
                    crate::petsc::libpetsc_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::petsc::libpetsc_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  petsc/csr_inodes: rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
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
                    crate::eigen::libeigen_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::eigen::libeigen_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  eigen/csc_map:    rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
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
                    crate::eigen::libeigen_csr_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::eigen::libeigen_csr_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  eigen/csr_map:    rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
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
                    crate::mkl::libmkl_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::mkl::libmkl_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  mkl/csr_ie:       rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
                    "{name}: mkl/csr_ie diverged: rel L2 = {err:.2e}"
                );
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
                    crate::mkl::libmkl_csc_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::mkl::libmkl_csc_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  mkl/csc_ie:       rel L2 = {err:.2e}");
                assert!(
                    err < tol_strict,
                    "{name}: mkl/csc_ie diverged: rel L2 = {err:.2e}"
                );
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
                    crate::psblas::libpsblas_spmv_get_y(
                        ctx,
                        y_buf.as_mut_ptr(),
                        raw.nrows as i32,
                    );
                    crate::psblas::libpsblas_spmv_teardown(ctx);
                }
                let err = relative_l2_error(&y_buf, &faer_ref);
                eprintln!("  psblas/csr:       rel L2 = {err:.2e}");
                assert!(
                    err < tol_psblas,
                    "{name}: psblas/csr diverged: rel L2 = {err:.2e}"
                );
            }
        }

        eprintln!("\nAll backends match Faer reference across {} matrices.", matrices.len());
        Ok(())
    }
}
