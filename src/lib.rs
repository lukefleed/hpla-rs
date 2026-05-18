//! Crate root for the hpla-rs benchmarking suite.
//!
//! Hosts the Matrix Market loader (`load_mtx_raw`, `RawMatrix`, `Symmetry`),
//! the faer SpMV kernels (`spmv_faer`, `spmv_faer_csr`), the `lanczos` module,
//! and the FFI shims for `eigen`, `mkl`, `petsc`, `psblas`.

pub mod eigen;
pub mod lanczos;
pub mod mkl;
pub mod petsc;
pub mod psblas;

#[cfg(test)]
mod tests;

// Counting global allocator, test-only, powering the zero-allocation
// regression test in `lanczos::algorithms::lanczos::tests`.
#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: crate::lanczos::alloc_counter::CountingAllocator =
    crate::lanczos::alloc_counter::CountingAllocator;

/// Names of the matrices used by the Lanczos benchmarks and by their
/// equivalence tests. Each entry is the stem of a `.mtx` file expected
/// to live under `matrices/` at bench time (the layout produced by
/// `download_matrices.sh`).
///
/// The suite is curated so that every matrix has small or zero mean
/// diagonal, which is the precondition for the Saad a posteriori error
/// estimator on `exp(-A)v` to be meaningful at every Krylov dimension.
/// See `docs/superpowers/specs/` and `README.md` for the rationale.
pub const LANCZOS_SUITE: &[&str] = &[
    "kron_g500-logn18",
    "coPapersDBLP",
    "thermal2",
    "as-Skitter",
    "roadNet-CA",
    "delaunay_n22",
    "caidaRouterLevel",
    "citationCiteseer",
    "coAuthorsCiteseer",
    "coPapersCiteseer",
    "preferentialAttachment",
    "smallworld",
    "rgg_n_2_20_s0",
    "belgium_osm",
    "auto",
];

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use faer::col::Col;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use faer::{Accum, Par};
use matrix_market_rs::MtxData;

/// Divides every numeric value in `raw` by `scale`, in place. Touches the
/// CSR `values`, the CSC `csc_values`, and the faer `triplets` together so
/// the three representations remain consistent. `O(nnz)`.
///
/// This realises the scalar `tau` of Saad 1992, formula (4): after calling
/// `scale_values(&mut raw, s)` the Lanczos kernel running on `raw` computes
/// `exp(-raw/s) v`, which is `exp(tau A) v` with `tau = -1/s`, the Krylov
/// subspace being invariant under scaling of `A`.
pub fn scale_values(raw: &mut RawMatrix, scale: f64) {
    let inv = 1.0 / scale;
    for v in raw.values.iter_mut() {
        *v *= inv;
    }
    for v in raw.csc_values.iter_mut() {
        *v *= inv;
    }
    for t in raw.triplets.iter_mut() {
        t.val *= inv;
    }
}

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

/// Symmetry classification extracted from a Matrix Market header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    General,
    Symmetric,
    Skew,
}

// Helper to determine symmetry reading the raw header
pub fn detect_symmetry(path: &Path) -> Symmetry {
    if let Ok(file) = File::open(path) {
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        if reader.read_line(&mut line).is_ok() {
            let lower = line.to_lowercase();
            if lower.starts_with("%%matrixmarket") {
                if lower.contains("skew-symmetric") {
                    return Symmetry::Skew;
                }
                if lower.contains("symmetric") || lower.contains("hermitian") {
                    return Symmetry::Symmetric;
                }
                return Symmetry::General;
            }
        }
    }
    Symmetry::General
}

pub fn load_mtx_raw(path: &Path) -> Result<RawMatrix, String> {
    let sym = detect_symmetry(path);
    let is_symmetric = matches!(sym, Symmetry::Symmetric);
    let is_skew = matches!(sym, Symmetry::Skew);
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

#[inline(always)]
pub fn spmv_faer_csr(a: &SparseRowMat<u32, f64>, x: &Col<f64>, y: &mut Col<f64>) {
    // faer 0.24 SparseDenseMatMul trait accepts SparseRowMatRef directly.
    // Internally reinterprets CSR as transposed CSC (zero-cost pointer rename)
    // and dispatches to dense_sparse_csc_matmul, which with M=1 (transposed
    // column vector) produces the row-oriented dot-product-per-row pattern.
    sparse_dense_matmul(
        y.as_mat_mut(),
        Accum::Add,
        a.as_ref(),
        x.as_mat(),
        1.0,
        Par::Seq,
    );
}
