//! Low-level FFI bindings to the C PETSc wrapper.
//!
//! Exposes external C functions that bypass standard library overhead
//! by operating directly on raw pointers.

use std::os::raw::{c_double, c_int};

/// Opaque struct representing the C-side internal context (Mat, Vecs).
#[repr(C)]
pub struct PetscBenchContext {
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
    ) -> *mut PetscBenchContext;

    pub fn libpetsc_spmv_execute(ctx: *mut PetscBenchContext);
    pub fn libpetsc_spmv_teardown(ctx: *mut PetscBenchContext);
}
