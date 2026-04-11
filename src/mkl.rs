//! Low-level FFI bindings to the C Intel MKL wrapper.
//!
//! Exposes external C functions for CSR and CSC Inspection-Execution SpMV.

use std::os::raw::c_double;

/// Opaque struct representing the C-side CSR benchmark context.
#[repr(C)]
pub struct MklCsrSpmv {
    _private: [u8; 0],
}

/// Opaque struct representing the C-side CSC benchmark context.
#[repr(C)]
pub struct MklCscSpmv {
    _private: [u8; 0],
}

unsafe extern "C" {
    // CSR Inspection-Execution API
    pub fn libmkl_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
    ) -> *mut MklCsrSpmv;

    pub fn libmkl_spmv_execute(ctx: *mut MklCsrSpmv);
    pub fn libmkl_spmv_get_y(ctx: *mut MklCsrSpmv, out: *mut c_double, len: i32);
    pub fn libmkl_spmv_teardown(ctx: *mut MklCsrSpmv);

    // CSC Inspection-Execution API
    pub fn libmkl_csc_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
    ) -> *mut MklCscSpmv;

    pub fn libmkl_csc_spmv_execute(ctx: *mut MklCscSpmv);
    pub fn libmkl_csc_spmv_get_y(ctx: *mut MklCscSpmv, out: *mut c_double, len: i32);
    pub fn libmkl_csc_spmv_teardown(ctx: *mut MklCscSpmv);
}
