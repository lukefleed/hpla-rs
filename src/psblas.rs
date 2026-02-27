#[repr(C)]
pub struct PsblasContextOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn libpsblas_spmv_setup(
        nrows: i32,
        ncols: i32,
        nnz: i32,
        row_ptr: *const i32,
        col_idx: *const i32,
        values: *const f64,
    ) -> *mut PsblasContextOpaque;

    pub fn libpsblas_spmv_execute(ctx: *mut PsblasContextOpaque);

    pub fn libpsblas_spmv_teardown(ctx: *mut PsblasContextOpaque);
}
