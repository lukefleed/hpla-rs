//! Low-level FFI bindings to the C++ Eigen wrapper.
//!
//! Exposes external C functions that bypass standard library overhead
//! by operating directly on raw pointers mapped via `Eigen::Map`.

use std::os::raw::c_double;

/// Opaque struct representing the C++ side internal context.
#[repr(C)]
pub struct EigenBenchContext {
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
    ) -> *mut EigenBenchContext;

    pub fn libeigen_spmv_execute(ctx: *mut EigenBenchContext);
    pub fn libeigen_spmv_teardown(ctx: *mut EigenBenchContext);
}
