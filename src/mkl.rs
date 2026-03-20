use std::ffi::c_double;

#[repr(C)]
pub struct MklBenchContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MklCscBenchContext {
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
    ) -> *mut MklBenchContext;

    pub fn libmkl_spmv_execute(ctx: *mut MklBenchContext);
    pub fn libmkl_spmv_teardown(ctx: *mut MklBenchContext);

    // CSC Inspection-Execution API
    pub fn libmkl_csc_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        col_ptr: *const i32,
        row_idx: *const i32,
        values: *const c_double,
    ) -> *mut MklCscBenchContext;

    pub fn libmkl_csc_spmv_execute(ctx: *mut MklCscBenchContext);
    pub fn libmkl_csc_spmv_teardown(ctx: *mut MklCscBenchContext);
}
