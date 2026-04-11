//! Low-level FFI bindings to the C++ Eigen wrapper.
//!
//! Exposes external C functions for CSC and CSR SpMV via Eigen::Map.

use std::os::raw::c_double;

/// Opaque struct representing the C++ side internal context.
#[repr(C)]
pub struct EigenCscSpmv {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
    ) -> *mut EigenCscSpmv;

    pub fn libeigen_spmv_execute(ctx: *mut EigenCscSpmv);
    pub fn libeigen_spmv_get_y(ctx: *mut EigenCscSpmv, out: *mut c_double, len: i32);
    pub fn libeigen_spmv_teardown(ctx: *mut EigenCscSpmv);
}

/// Opaque struct representing the C++ side CSR internal context.
#[repr(C)]
pub struct EigenCsrSpmv {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_csr_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
    ) -> *mut EigenCsrSpmv;

    pub fn libeigen_csr_spmv_execute(ctx: *mut EigenCsrSpmv);
    pub fn libeigen_csr_spmv_get_y(ctx: *mut EigenCsrSpmv, out: *mut c_double, len: i32);
    pub fn libeigen_csr_spmv_teardown(ctx: *mut EigenCsrSpmv);
}
