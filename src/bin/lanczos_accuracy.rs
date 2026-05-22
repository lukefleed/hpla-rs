//! Export cross-backend Lanczos accuracy checks to CSV.

use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseRowMat};
use faer::{Mat, Par};
use hpla_rs::eigen::{
    libeigen_csc_lanczos_execute, libeigen_csc_lanczos_get_y, libeigen_csc_lanczos_setup,
    libeigen_csc_lanczos_teardown, libeigen_csc_lanczos_two_pass_execute,
    libeigen_csc_lanczos_two_pass_get_y, libeigen_csc_lanczos_two_pass_setup,
    libeigen_csc_lanczos_two_pass_teardown, libeigen_lanczos_execute, libeigen_lanczos_get_y,
    libeigen_lanczos_setup, libeigen_lanczos_teardown, libeigen_lanczos_two_pass_execute,
    libeigen_lanczos_two_pass_get_y, libeigen_lanczos_two_pass_setup,
    libeigen_lanczos_two_pass_teardown,
};
use hpla_rs::lanczos::{
    KRYLOV_HARD_LIMIT, KRYLOV_MARGIN, ProjectedTridiagonalWorkspace, Reorthogonalization, SAAD_TOL,
    SPECTRAL_PROBE_STEPS, adaptive_krylov_dim_with_estimate, deterministic_rhs,
    estimate_spectral_radius, lanczos, lanczos_two_pass,
};
use hpla_rs::petsc::{
    libpetsc_lanczos_execute, libpetsc_lanczos_get_y, libpetsc_lanczos_setup,
    libpetsc_lanczos_teardown, libpetsc_lanczos_two_pass_execute, libpetsc_lanczos_two_pass_get_y,
    libpetsc_lanczos_two_pass_setup, libpetsc_lanczos_two_pass_teardown,
};
use hpla_rs::psblas::{
    libpsblas_csc_lanczos_execute, libpsblas_csc_lanczos_get_y, libpsblas_csc_lanczos_setup,
    libpsblas_csc_lanczos_teardown, libpsblas_csc_lanczos_two_pass_execute,
    libpsblas_csc_lanczos_two_pass_get_y, libpsblas_csc_lanczos_two_pass_setup,
    libpsblas_csc_lanczos_two_pass_teardown, libpsblas_csr_lanczos_execute,
    libpsblas_csr_lanczos_get_y, libpsblas_csr_lanczos_setup, libpsblas_csr_lanczos_teardown,
    libpsblas_csr_lanczos_two_pass_execute, libpsblas_csr_lanczos_two_pass_get_y,
    libpsblas_csr_lanczos_two_pass_setup, libpsblas_csr_lanczos_two_pass_teardown,
};
use hpla_rs::{LANCZOS_SUITE, RawMatrix, load_mtx_raw, scale_values};
use indicatif::{ProgressBar, ProgressStyle};

const ONE_PASS: &str = "lanczos_one_pass";
const TWO_PASS: &str = "lanczos_two_pass";
const REL_L2_TOL: f64 = 1e-8;

#[derive(Debug, Parser)]
#[command(
    name = "lanczos_accuracy",
    about = "Export cross-backend Lanczos accuracy checks to CSV."
)]
struct Cli {
    /// CSV path to write.
    #[arg(
        short,
        long,
        value_name = "PATH",
        default_value = "python/data/lanczos_accuracy.csv"
    )]
    output: PathBuf,
}

#[derive(Debug)]
struct AccuracyRow {
    kernel: &'static str,
    matrix: &'static str,
    backend: &'static str,
    format: &'static str,
    m: Option<usize>,
    saad_tol: Option<f64>,
    saad_estimate: Option<f64>,
    rel_l2_vs_faer: Option<f64>,
    norm_y: Option<f64>,
    status: &'static str,
}

#[derive(Debug)]
struct FfiDims {
    nrows: i32,
    ncols: i32,
    nnz: i32,
    len: i32,
}

struct MatrixCase {
    name: &'static str,
    raw: RawMatrix,
    a_csc: SparseColMat<u32, f64>,
    a_csr: SparseRowMat<u32, f64>,
    b_vec: Vec<f64>,
    b_mat: Mat<f64>,
    m: usize,
    saad_estimate: f64,
}

#[derive(Clone, Copy)]
struct CaseInfo {
    matrix: &'static str,
    m: usize,
    saad_estimate: f64,
}

struct ResultContext<'a> {
    info: CaseInfo,
    kernel: &'static str,
    reference: &'a [f64],
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    export_accuracy(&cli.output)
}

fn export_accuracy(output: &Path) -> Result<()> {
    let mut rows = Vec::new();
    let mut had_failure = false;
    let progress = progress_bar()?;

    for &name in LANCZOS_SUITE {
        set_progress_stage(&progress, name, "checking matrix file");
        match prepare_matrix(name, &progress) {
            Ok(Some(case)) => {
                had_failure |= record_case(&case, &mut rows, &progress);
            }
            Ok(None) => {
                progress_log(&progress, format!("{name}: missing matrix"));
                rows.push(status_row(ONE_PASS, name, "missing_matrix"));
                rows.push(status_row(TWO_PASS, name, "missing_matrix"));
            }
            Err(error) => {
                progress_log(&progress, format!("{name}: {error:#}"));
                rows.push(status_row(ONE_PASS, name, "failed"));
                rows.push(status_row(TWO_PASS, name, "failed"));
                had_failure = true;
            }
        }
        progress.inc(1);
    }
    progress.finish_and_clear();

    write_rows(output, &rows)?;
    eprintln!("wrote {}", output.display());
    print_summary(&rows);

    if had_failure {
        bail!("accuracy export completed with failures");
    }

    Ok(())
}

fn progress_bar() -> Result<ProgressBar> {
    let progress = ProgressBar::new(LANCZOS_SUITE.len() as u64);
    let style = ProgressStyle::with_template("[{bar:32.cyan/blue}] {pos:>2}/{len} {msg}")
        .context("failed to configure progress bar")?
        .progress_chars("=>-");
    progress.set_style(style);
    Ok(progress)
}

fn progress_log(progress: &ProgressBar, message: impl AsRef<str>) {
    progress.suspend(|| eprintln!("{}", message.as_ref()));
}

fn set_progress_stage(progress: &ProgressBar, matrix: &str, stage: impl AsRef<str>) {
    progress.set_message(format!("{matrix} | {}", stage.as_ref()));
}

fn prepare_matrix(name: &'static str, progress: &ProgressBar) -> Result<Option<MatrixCase>> {
    let path = PathBuf::from(format!("matrices/{name}.mtx"));
    if !path.exists() {
        return Ok(None);
    }

    set_progress_stage(progress, name, "loading Matrix Market file");
    let mut raw =
        load_mtx_raw(&path).map_err(|error| anyhow!("{name}: failed to load matrix: {error}"))?;
    let b_vec = deterministic_rhs(raw.nrows);

    set_progress_stage(
        progress,
        name,
        format!(
            "estimating rho(A), n={}, nnz={}, probe_steps={}",
            raw.nrows, raw.nnz, SPECTRAL_PROBE_STEPS
        ),
    );
    let scale = estimate_unscaled_radius(&raw, &b_vec, name)?;
    ensure!(
        scale.is_finite() && scale > 0.0,
        "{name}: invalid spectral radius estimate {scale}"
    );

    set_progress_stage(progress, name, format!("scaling matrix by rho={scale:.3e}"));
    scale_values(&mut raw, scale);

    set_progress_stage(progress, name, "building faer CSC/CSR matrices");
    let a_csc = sparse_col(&raw, name)?;
    let a_csr = sparse_row(&raw, name)?;
    let b_mat = Mat::from_fn(raw.nrows, 1, |i, _| b_vec[i]);

    let scratch_req = a_csc.as_ref().apply_scratch(1, Par::Seq);
    set_progress_stage(progress, name, "estimating rho(A/rho)");
    let spectral_radius = {
        let mut mem = MemBuffer::new(scratch_req);
        let stack = MemStack::new(&mut mem);
        estimate_spectral_radius(
            &a_csc.as_ref(),
            b_mat.as_ref(),
            SPECTRAL_PROBE_STEPS,
            Par::Seq,
            stack,
        )
        .with_context(|| format!("{name}: scaled spectral-radius probe failed"))?
    };
    let max_k = ((spectral_radius.ceil() as usize) + KRYLOV_MARGIN).min(KRYLOV_HARD_LIMIT);

    set_progress_stage(
        progress,
        name,
        format!("selecting Krylov dimension, max_k={max_k}, tol={SAAD_TOL:.1e}"),
    );
    let (m, _, saad_estimate) = {
        let mut mem = MemBuffer::new(scratch_req);
        let stack = MemStack::new(&mut mem);
        adaptive_krylov_dim_with_estimate(
            &a_csc.as_ref(),
            b_mat.as_ref(),
            max_k,
            SAAD_TOL,
            Par::Seq,
            stack,
        )
        .with_context(|| format!("{name}: krylov-dimension probe failed"))?
    };
    let m = m.max(1);
    set_progress_stage(
        progress,
        name,
        format!("ready, m={m}, saad_estimate={saad_estimate:.3e}"),
    );

    Ok(Some(MatrixCase {
        name,
        raw,
        a_csc,
        a_csr,
        b_vec,
        b_mat,
        m,
        saad_estimate,
    }))
}

fn estimate_unscaled_radius(raw: &RawMatrix, b: &[f64], name: &str) -> Result<f64> {
    let a_tmp = sparse_col(raw, name)?;
    let b_tmp = Mat::from_fn(raw.nrows, 1, |i, _| b[i]);
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
    .with_context(|| format!("{name}: unscaled spectral-radius probe failed"))
}

fn sparse_col(raw: &RawMatrix, name: &str) -> Result<SparseColMat<u32, f64>> {
    SparseColMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
        .map_err(|error| anyhow!("{name}: failed to build faer csc matrix: {error:?}"))
}

fn sparse_row(raw: &RawMatrix, name: &str) -> Result<SparseRowMat<u32, f64>> {
    SparseRowMat::try_new_from_triplets(raw.nrows, raw.ncols, &raw.triplets)
        .map_err(|error| anyhow!("{name}: failed to build faer csr matrix: {error:?}"))
}

fn record_case(case: &MatrixCase, rows: &mut Vec<AccuracyRow>, progress: &ProgressBar) -> bool {
    let mut had_failure = false;
    let info = case_info(case);

    set_progress_backend(progress, info, ONE_PASS, "faer", "csc");
    let one_pass_csc = match run_faer_one_pass(&case.a_csc.as_ref(), case.b_mat.as_ref(), case.m) {
        Ok(result) => result,
        Err(error) => {
            progress_log(
                progress,
                format!("{} {ONE_PASS} faer/csc: {error:#}", case.name),
            );
            rows.push(status_row(ONE_PASS, case.name, "failed"));
            had_failure = true;
            Vec::new()
        }
    };

    if !one_pass_csc.is_empty() {
        let ctx = ResultContext {
            info,
            kernel: ONE_PASS,
            reference: &one_pass_csc,
        };
        had_failure |= push_result_row(rows, &ctx, "faer", "csc", one_pass_csc.clone());
        had_failure |= record_backend(rows, &ctx, progress, "faer", "csr", || {
            run_faer_one_pass(&case.a_csr.as_ref(), case.b_mat.as_ref(), case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "eigen", "csr", || {
            run_eigen_one_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "eigen", "csc", || {
            run_eigen_one_pass_csc(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "petsc", "csr", || {
            run_petsc_one_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "psblas", "csr", || {
            run_psblas_one_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "psblas", "csc", || {
            run_psblas_one_pass_csc(&case.raw, &case.b_vec, case.m)
        });
    }

    set_progress_backend(progress, info, TWO_PASS, "faer", "csc");
    let two_pass_csc = match run_faer_two_pass(&case.a_csc.as_ref(), case.b_mat.as_ref(), case.m) {
        Ok(result) => result,
        Err(error) => {
            progress_log(
                progress,
                format!("{} {TWO_PASS} faer/csc: {error:#}", case.name),
            );
            rows.push(status_row(TWO_PASS, case.name, "failed"));
            had_failure = true;
            Vec::new()
        }
    };

    if !two_pass_csc.is_empty() {
        let ctx = ResultContext {
            info,
            kernel: TWO_PASS,
            reference: &two_pass_csc,
        };
        had_failure |= push_result_row(rows, &ctx, "faer", "csc", two_pass_csc.clone());
        had_failure |= record_backend(rows, &ctx, progress, "faer", "csr", || {
            run_faer_two_pass(&case.a_csr.as_ref(), case.b_mat.as_ref(), case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "eigen", "csr", || {
            run_eigen_two_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "eigen", "csc", || {
            run_eigen_two_pass_csc(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "petsc", "csr", || {
            run_petsc_two_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "psblas", "csr", || {
            run_psblas_two_pass_csr(&case.raw, &case.b_vec, case.m)
        });
        had_failure |= record_backend(rows, &ctx, progress, "psblas", "csc", || {
            run_psblas_two_pass_csc(&case.raw, &case.b_vec, case.m)
        });
    }

    had_failure
}

fn set_progress_backend(
    progress: &ProgressBar,
    info: CaseInfo,
    kernel: &str,
    backend: &str,
    format: &str,
) {
    let kernel = match kernel {
        ONE_PASS => "one-pass",
        TWO_PASS => "two-pass",
        _ => kernel,
    };
    progress.set_message(format!(
        "{} | {kernel} | {backend}/{format} | m={}, saad={:.2e}",
        info.matrix, info.m, info.saad_estimate
    ));
}

fn case_info(case: &MatrixCase) -> CaseInfo {
    CaseInfo {
        matrix: case.name,
        m: case.m,
        saad_estimate: case.saad_estimate,
    }
}

fn record_backend(
    rows: &mut Vec<AccuracyRow>,
    ctx: &ResultContext<'_>,
    progress: &ProgressBar,
    backend: &'static str,
    format: &'static str,
    run: impl FnOnce() -> Result<Vec<f64>>,
) -> bool {
    set_progress_backend(progress, ctx.info, ctx.kernel, backend, format);
    match run() {
        Ok(result) => push_result_row(rows, ctx, backend, format, result),
        Err(error) => {
            progress_log(
                progress,
                format!(
                    "{} {} {backend}/{format}: {error:#}",
                    ctx.info.matrix, ctx.kernel
                ),
            );
            rows.push(result_row(ctx, backend, format, None, None, "failed"));
            true
        }
    }
}

fn push_result_row(
    rows: &mut Vec<AccuracyRow>,
    ctx: &ResultContext<'_>,
    backend: &'static str,
    format: &'static str,
    y: Vec<f64>,
) -> bool {
    let Ok(rel) = relative_l2_error(&y, ctx.reference) else {
        rows.push(result_row(ctx, backend, format, None, None, "failed"));
        return true;
    };
    let norm = norm_l2(&y);
    let status = if !rel.is_finite() || !norm.is_finite() {
        "failed"
    } else if rel <= REL_L2_TOL {
        "ok"
    } else {
        "diverged"
    };
    rows.push(result_row(
        ctx,
        backend,
        format,
        Some(rel),
        Some(norm),
        status,
    ));
    status != "ok"
}

fn result_row(
    ctx: &ResultContext<'_>,
    backend: &'static str,
    format: &'static str,
    rel_l2_vs_faer: Option<f64>,
    norm_y: Option<f64>,
    status: &'static str,
) -> AccuracyRow {
    AccuracyRow {
        kernel: ctx.kernel,
        matrix: ctx.info.matrix,
        backend,
        format,
        m: Some(ctx.info.m),
        saad_tol: Some(SAAD_TOL),
        saad_estimate: Some(ctx.info.saad_estimate),
        rel_l2_vs_faer,
        norm_y,
        status,
    }
}

fn status_row(kernel: &'static str, matrix: &'static str, status: &'static str) -> AccuracyRow {
    AccuracyRow {
        kernel,
        matrix,
        backend: "",
        format: "",
        m: None,
        saad_tol: Some(SAAD_TOL),
        saad_estimate: None,
        rel_l2_vs_faer: None,
        norm_y: None,
        status,
    }
}

fn run_faer_one_pass(operator: &impl LinOp<f64>, b: MatRef<'_, f64>, m: usize) -> Result<Vec<f64>> {
    let scratch_req = operator.apply_scratch(1, Par::Seq);
    let mut mem = MemBuffer::new(scratch_req);
    let stack = MemStack::new(&mut mem);
    let mut projected = ProjectedTridiagonalWorkspace::new(m, Par::Seq);
    let result = lanczos(
        operator,
        b,
        m,
        Par::Seq,
        Reorthogonalization::None,
        stack,
        |alphas, betas, out| projected.exp_neg_tk(alphas, betas, out),
    )?;
    Ok(mat_to_vec(result.as_ref()))
}

fn run_faer_two_pass(operator: &impl LinOp<f64>, b: MatRef<'_, f64>, m: usize) -> Result<Vec<f64>> {
    let scratch_req = operator.apply_scratch(1, Par::Seq);
    let mut mem = MemBuffer::new(scratch_req);
    let stack = MemStack::new(&mut mem);
    let mut projected = ProjectedTridiagonalWorkspace::new(m, Par::Seq);
    let result = lanczos_two_pass(operator, b, m, Par::Seq, stack, |alphas, betas, out| {
        projected.exp_neg_tk(alphas, betas, out)
    })?;
    Ok(mat_to_vec(result.as_ref()))
}

fn run_eigen_one_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libeigen_lanczos_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "eigen/csr one-pass")?;
        libeigen_lanczos_execute(ctx);
        libeigen_lanczos_get_y(ctx, out.as_mut_ptr(), dims.len);
        libeigen_lanczos_teardown(ctx);
    }

    Ok(out)
}

fn run_eigen_one_pass_csc(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSC buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libeigen_csc_lanczos_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.col_ptr.as_ptr(),
            raw.row_idx.as_ptr(),
            raw.csc_values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "eigen/csc one-pass")?;
        libeigen_csc_lanczos_execute(ctx);
        libeigen_csc_lanczos_get_y(ctx, out.as_mut_ptr(), dims.len);
        libeigen_csc_lanczos_teardown(ctx);
    }

    Ok(out)
}

fn run_eigen_two_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libeigen_lanczos_two_pass_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "eigen/csr two-pass")?;
        libeigen_lanczos_two_pass_execute(ctx);
        libeigen_lanczos_two_pass_get_y(ctx, out.as_mut_ptr(), dims.len);
        libeigen_lanczos_two_pass_teardown(ctx);
    }

    Ok(out)
}

fn run_eigen_two_pass_csc(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSC buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libeigen_csc_lanczos_two_pass_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.col_ptr.as_ptr(),
            raw.row_idx.as_ptr(),
            raw.csc_values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "eigen/csc two-pass")?;
        libeigen_csc_lanczos_two_pass_execute(ctx);
        libeigen_csc_lanczos_two_pass_get_y(ctx, out.as_mut_ptr(), dims.len);
        libeigen_csc_lanczos_two_pass_teardown(ctx);
    }

    Ok(out)
}

fn run_petsc_one_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpetsc_lanczos_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "petsc/csr one-pass")?;
        libpetsc_lanczos_execute(ctx);
        libpetsc_lanczos_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpetsc_lanczos_teardown(ctx);
    }

    Ok(out)
}

fn run_petsc_two_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpetsc_lanczos_two_pass_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "petsc/csr two-pass")?;
        libpetsc_lanczos_two_pass_execute(ctx);
        libpetsc_lanczos_two_pass_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpetsc_lanczos_two_pass_teardown(ctx);
    }

    Ok(out)
}

fn run_psblas_one_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpsblas_csr_lanczos_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "psblas/csr one-pass")?;
        libpsblas_csr_lanczos_execute(ctx);
        libpsblas_csr_lanczos_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpsblas_csr_lanczos_teardown(ctx);
    }

    Ok(out)
}

fn run_psblas_one_pass_csc(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSC buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpsblas_csc_lanczos_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.col_ptr.as_ptr(),
            raw.row_idx.as_ptr(),
            raw.csc_values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "psblas/csc one-pass")?;
        libpsblas_csc_lanczos_execute(ctx);
        libpsblas_csc_lanczos_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpsblas_csc_lanczos_teardown(ctx);
    }

    Ok(out)
}

fn run_psblas_two_pass_csr(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSR buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpsblas_csr_lanczos_two_pass_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.row_ptr.as_ptr(),
            raw.col_idx.as_ptr(),
            raw.values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "psblas/csr two-pass")?;
        libpsblas_csr_lanczos_two_pass_execute(ctx);
        libpsblas_csr_lanczos_two_pass_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpsblas_csr_lanczos_two_pass_teardown(ctx);
    }

    Ok(out)
}

fn run_psblas_two_pass_csc(raw: &RawMatrix, b: &[f64], m: usize) -> Result<Vec<f64>> {
    let dims = ffi_dims(raw)?;
    let mut out = vec![0.0; raw.nrows];
    let m = checked_i32(m, "krylov dimension")?;

    // The Rust-owned CSC buffers and `b` outlive setup, execute, and get_y.
    unsafe {
        let ctx = libpsblas_csc_lanczos_two_pass_setup(
            dims.nrows,
            dims.ncols,
            dims.nnz,
            raw.col_ptr.as_ptr(),
            raw.row_idx.as_ptr(),
            raw.csc_values.as_ptr(),
            b.as_ptr(),
            m,
        );
        ensure_context(ctx, "psblas/csc two-pass")?;
        libpsblas_csc_lanczos_two_pass_execute(ctx);
        libpsblas_csc_lanczos_two_pass_get_y(ctx, out.as_mut_ptr(), dims.len);
        libpsblas_csc_lanczos_two_pass_teardown(ctx);
    }

    Ok(out)
}

fn ffi_dims(raw: &RawMatrix) -> Result<FfiDims> {
    Ok(FfiDims {
        nrows: checked_i32(raw.nrows, "row count")?,
        ncols: checked_i32(raw.ncols, "column count")?,
        nnz: checked_i32(raw.nnz, "nonzero count")?,
        len: checked_i32(raw.nrows, "output length")?,
    })
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).with_context(|| format!("{name} does not fit in i32"))
}

fn ensure_context<T>(ctx: *mut T, label: &str) -> Result<()> {
    ensure!(!ctx.is_null(), "{label} setup returned null");
    Ok(())
}

fn mat_to_vec(mat: MatRef<'_, f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| mat[(i, 0)]).collect()
}

fn norm_l2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn relative_l2_error(actual: &[f64], reference: &[f64]) -> Result<f64> {
    ensure!(
        actual.len() == reference.len(),
        "length mismatch: actual={}, reference={}",
        actual.len(),
        reference.len()
    );
    let diff_norm = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (actual - reference).powi(2))
        .sum::<f64>()
        .sqrt();
    let ref_norm = norm_l2(reference);
    if ref_norm == 0.0 {
        Ok(diff_norm)
    } else {
        Ok(diff_norm / ref_norm)
    }
}

fn write_rows(output: &Path, rows: &[AccuracyRow]) -> Result<()> {
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file =
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record([
        "kernel",
        "matrix",
        "backend",
        "format",
        "m",
        "saad_tol",
        "saad_estimate",
        "rel_l2_vs_faer",
        "norm_y",
        "status",
    ])?;

    for row in rows {
        writer.write_record([
            row.kernel.to_owned(),
            row.matrix.to_owned(),
            row.backend.to_owned(),
            row.format.to_owned(),
            format_usize(row.m),
            format_f64(row.saad_tol),
            format_f64(row.saad_estimate),
            format_f64(row.rel_l2_vs_faer),
            format_f64(row.norm_y),
            row.status.to_owned(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn print_summary(rows: &[AccuracyRow]) {
    let ok = rows.iter().filter(|row| row.status == "ok").count();
    let diverged = rows.iter().filter(|row| row.status == "diverged").count();
    let failed = rows.iter().filter(|row| row.status == "failed").count();
    let missing = rows
        .iter()
        .filter(|row| row.status == "missing_matrix")
        .count();

    eprintln!("summary: ok={ok}, diverged={diverged}, failed={failed}, missing_matrix={missing}");
}

fn format_usize(value: Option<usize>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn format_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.17e}"))
}
