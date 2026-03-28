//! Low-level FFI bindings to the C++ PSBLAS wrapper.
//!
//! Exposes external C functions for CSR SpMV via Fortran PSBLAS C bindings.

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
