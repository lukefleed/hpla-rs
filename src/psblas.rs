//! Low-level FFI bindings to the C++ PSBLAS wrapper.
//!
//! Exposes external C functions for CSR/CSC SpMV and two-pass Lanczos
//! via Fortran PSBLAS C bindings.

use std::os::raw::c_double;

/// Opaque struct representing the C++ side PSBLAS benchmark context.
#[repr(C)]
pub struct PsblasBenchContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libpsblas_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
    ) -> *mut PsblasBenchContext;

    pub fn libpsblas_spmv_execute(ctx: *mut PsblasBenchContext);
    pub fn libpsblas_spmv_get_y(ctx: *mut PsblasBenchContext, out: *mut c_double, len: i32);

    pub fn libpsblas_spmv_teardown(ctx: *mut PsblasBenchContext);

    // CSC variant — same context/execute/get_y/teardown, different assembly format
    pub fn libpsblas_csc_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
    ) -> *mut PsblasBenchContext;
}

/// Opaque struct for the PSBLAS two-pass Lanczos benchmark context.
#[repr(C)]
pub struct PsblasLanczosBenchContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    /// Assembles the sparse matrix and starting vector into PSBLAS structures.
    /// `krylov_dim` is the number of Lanczos iterations (determined by the Rust
    /// side via the Saad 1992 a posteriori error estimate).
    pub fn libpsblas_lanczos_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PsblasLanczosBenchContext;

    /// Runs the full two-pass Lanczos computing exp(-A)b. Must be idempotent
    /// (Criterion calls this many times per benchmark).
    pub fn libpsblas_lanczos_execute(ctx: *mut PsblasLanczosBenchContext);

    /// Copies the result vector into the caller-owned buffer `out`.
    pub fn libpsblas_lanczos_get_y(
        ctx: *mut PsblasLanczosBenchContext,
        out: *mut c_double,
        len: i32,
    );

    /// Frees all PSBLAS objects. Must use psb_c_exit_ctxt (not psb_c_exit)
    /// to avoid calling MPI_Finalize.
    pub fn libpsblas_lanczos_teardown(ctx: *mut PsblasLanczosBenchContext);
}
