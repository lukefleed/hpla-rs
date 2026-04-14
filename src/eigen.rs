//! Low-level FFI bindings to the C++ Eigen wrapper.
//!
//! Exposes external C functions for CSC and CSR SpMV via Eigen::Map,
//! and CSR/CSC two-pass Lanczos for `exp(-A)b`.

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

/// Opaque struct representing the C++ side two-pass Lanczos context.
#[repr(C)]
pub struct EigenLanczosTwoPass {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_lanczos_two_pass_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut EigenLanczosTwoPass;

    pub fn libeigen_lanczos_two_pass_execute(ctx: *mut EigenLanczosTwoPass);
    pub fn libeigen_lanczos_two_pass_get_y(
        ctx: *mut EigenLanczosTwoPass,
        out: *mut c_double,
        len: i32,
    );
    pub fn libeigen_lanczos_two_pass_teardown(ctx: *mut EigenLanczosTwoPass);
}

/// Opaque struct for the Eigen CSC two-pass Lanczos context (cross-format control).
#[repr(C)]
pub struct EigenCscLanczosTwoPass {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_csc_lanczos_two_pass_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut EigenCscLanczosTwoPass;

    pub fn libeigen_csc_lanczos_two_pass_execute(ctx: *mut EigenCscLanczosTwoPass);
    pub fn libeigen_csc_lanczos_two_pass_get_y(
        ctx: *mut EigenCscLanczosTwoPass,
        out: *mut c_double,
        len: i32,
    );
    pub fn libeigen_csc_lanczos_two_pass_teardown(ctx: *mut EigenCscLanczosTwoPass);
}

/// Opaque struct for the Eigen CSR one-pass Lanczos context.
#[repr(C)]
pub struct EigenLanczos {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_lanczos_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut EigenLanczos;

    pub fn libeigen_lanczos_execute(ctx: *mut EigenLanczos);
    pub fn libeigen_lanczos_get_y(ctx: *mut EigenLanczos, out: *mut c_double, len: i32);
    pub fn libeigen_lanczos_teardown(ctx: *mut EigenLanczos);
}

/// Opaque struct for the Eigen CSC one-pass Lanczos context (cross-format control).
#[repr(C)]
pub struct EigenCscLanczos {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libeigen_csc_lanczos_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
        b: *const c_double,
        krylov_dim: i32,
    ) -> *mut EigenCscLanczos;

    pub fn libeigen_csc_lanczos_execute(ctx: *mut EigenCscLanczos);
    pub fn libeigen_csc_lanczos_get_y(ctx: *mut EigenCscLanczos, out: *mut c_double, len: i32);
    pub fn libeigen_csc_lanczos_teardown(ctx: *mut EigenCscLanczos);
}
