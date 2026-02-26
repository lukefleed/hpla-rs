pub mod petsc;

use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
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
    pub triplets: Vec<Triplet<u32, f64>>, // For Faer CSC
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

    let capacity = if is_symmetric || is_skew {
        coords.len() * 2
    } else {
        coords.len()
    };
    
    let mut triplets = Vec::with_capacity(capacity);
    let mut row_counts = vec![0; nrows];

    for ([r, c], &v) in coords.iter().zip(values.iter()) {
        let row = *r as usize;
        let col = *c as usize;

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
        if a.row() != b.row() {
            a.row().cmp(&b.row())
        } else {
            a.col().cmp(&b.col())
        }
    });

    let mut row_ptr = vec![0i32; nrows + 1];
    let mut col_idx = vec![0i32; nnz];
    let mut csr_values = vec![0.0f64; nnz];

    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i] as i32;
    }

    for (i, t) in triplets.iter().enumerate() {
        col_idx[i] = t.col() as i32;
        csr_values[i] = t.val();
    }

    Ok(RawMatrix {
        nrows,
        ncols,
        nnz,
        row_ptr,
        col_idx,
        values: csr_values,
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
