//! Low-level FFI bindings to the C PETSc wrapper.
//!
//! Exposes external C functions for CSR SpMV via PETSc.

use std::os::raw::{c_double, c_int};

/// Opaque struct representing the C-side internal context (Mat, Vecs).
#[repr(C)]
pub struct PetscSpmv {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libpetsc_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        disable_inode: c_int,
    ) -> *mut PetscSpmv;

    pub fn libpetsc_spmv_execute(ctx: *mut PetscSpmv);
    pub fn libpetsc_spmv_get_y(ctx: *mut PetscSpmv, out: *mut c_double, len: i32);
    pub fn libpetsc_spmv_teardown(ctx: *mut PetscSpmv);
}

/// Opaque context for the PETSc one-pass Lanczos kernel (`exp(-A)b`).
#[repr(C)]
pub struct PetscLanczos {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libpetsc_lanczos_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PetscLanczos;

    pub fn libpetsc_lanczos_execute(ctx: *mut PetscLanczos);
    pub fn libpetsc_lanczos_get_y(ctx: *mut PetscLanczos, out: *mut c_double, len: i32);
    pub fn libpetsc_lanczos_teardown(ctx: *mut PetscLanczos);
}

/// Opaque context for the PETSc two-pass Lanczos kernel (`exp(-A)b`).
#[repr(C)]
pub struct PetscLanczosTwoPass {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libpetsc_lanczos_two_pass_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut PetscLanczosTwoPass;

    pub fn libpetsc_lanczos_two_pass_execute(ctx: *mut PetscLanczosTwoPass);
    pub fn libpetsc_lanczos_two_pass_get_y(
        ctx: *mut PetscLanczosTwoPass,
        out: *mut c_double,
        len: i32,
    );
    pub fn libpetsc_lanczos_two_pass_teardown(ctx: *mut PetscLanczosTwoPass);
}
