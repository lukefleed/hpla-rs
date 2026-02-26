use std::ffi::c_double;

#[repr(C)]
pub struct MklBenchContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libmkl_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const c_double,
    ) -> *mut MklBenchContext;

    pub fn libmkl_spmv_execute(ctx: *mut MklBenchContext);
    pub fn libmkl_spmv_teardown(ctx: *mut MklBenchContext);
}
