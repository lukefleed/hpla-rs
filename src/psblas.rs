//! Low-level FFI bindings to the C++/Fortran PSBLAS wrappers.
//!
//! Exposes external symbols for CSR/CSC SpMV (`ffi/spmv/psblas.cpp`),
//! one-pass symmetric Lanczos (`ffi/lanczos/psblas_lanczos.f90`) and
//! two-pass Lanczos for `f(A)b` (`ffi/lanczos/psblas_lanczos_two_pass.f90`).

use std::os::raw::c_double;

/// Opaque struct representing the C++ side PSBLAS benchmark context.
#[repr(C)]
pub struct PsblasSpmv {
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
    ) -> *mut PsblasSpmv;

    pub fn libpsblas_spmv_execute(ctx: *mut PsblasSpmv);
    pub fn libpsblas_spmv_get_y(ctx: *mut PsblasSpmv, out: *mut c_double, len: i32);

    pub fn libpsblas_spmv_teardown(ctx: *mut PsblasSpmv);

    // CSC variant — same context/execute/get_y/teardown, different assembly format
    pub fn libpsblas_csc_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
    ) -> *mut PsblasSpmv;
}

/// Opaque context for the PSBLAS one-pass Lanczos kernel (`f(A)b`).
#[repr(C)]
pub struct PsblasLanczos {
    _private: [u8; 0],
}

unsafe extern "C" {
    /// Assembles the sparse matrix and starting vector into PSBLAS structures.
    /// `krylov_dim` is the number of Lanczos iterations (determined by the
    /// Rust side via the Saad 1992 a posteriori error estimate).
    pub fn libpsblas_lanczos_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PsblasLanczos;

    /// Runs the one-pass Lanczos computing `exp(-A)b`: builds `V_m` in
    /// memory, solves `g = exp(-T_m)*e_1`, and accumulates
    /// `y = ||b|| * V_m * g`. Must be idempotent (Criterion calls this
    /// many times per benchmark).
    pub fn libpsblas_lanczos_execute(ctx: *mut PsblasLanczos);

    /// Copies the result vector into the caller-owned buffer `out`.
    pub fn libpsblas_lanczos_get_y(ctx: *mut PsblasLanczos, out: *mut c_double, len: i32);

    /// Frees all PSBLAS objects. Must use `psb_c_exit_ctxt` (not
    /// `psb_c_exit`) to avoid calling `MPI_Finalize`.
    pub fn libpsblas_lanczos_teardown(ctx: *mut PsblasLanczos);
}

/// Opaque context for the PSBLAS two-pass Lanczos kernel (`f(A)b`).
#[repr(C)]
pub struct PsblasLanczosTwoPass {
    _private: [u8; 0],
}

unsafe extern "C" {
    /// Assembles the sparse matrix and starting vector into PSBLAS structures.
    /// `krylov_dim` is the number of Lanczos iterations (determined by the
    /// Rust side via the Saad 1992 a posteriori error estimate).
    pub fn libpsblas_lanczos_two_pass_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PsblasLanczosTwoPass;

    /// Runs the full two-pass Lanczos computing `exp(-A)b`. Must be idempotent
    /// (Criterion calls this many times per benchmark).
    pub fn libpsblas_lanczos_two_pass_execute(ctx: *mut PsblasLanczosTwoPass);

    /// Copies the result vector into the caller-owned buffer `out`.
    pub fn libpsblas_lanczos_two_pass_get_y(
        ctx: *mut PsblasLanczosTwoPass,
        out: *mut c_double,
        len: i32,
    );

    /// Frees all PSBLAS objects. Must use `psb_c_exit_ctxt` (not
    /// `psb_c_exit`) to avoid calling `MPI_Finalize`.
    pub fn libpsblas_lanczos_two_pass_teardown(ctx: *mut PsblasLanczosTwoPass);

    /// CSC variant of the two-pass setup. Inserts from CSC arrays and
    /// assembles internally as CSC. The execute/get_y/teardown symbols
    /// are shared with the CSR variant (same opaque context).
    pub fn libpsblas_csc_lanczos_two_pass_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PsblasLanczosTwoPass;
}
