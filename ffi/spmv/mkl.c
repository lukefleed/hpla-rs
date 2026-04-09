// MKL Sparse BLAS FFI wrapper for SpMV benchmarking (Inspection-Execution API).
// Matrix: create_csr/csc stores pointers to caller's arrays (zero-copy at creation);
// mkl_sparse_optimize() may build optimized internal representations.
// The actual SpMV kernel is in libmkl_core.so (Intel-compiled).
// Vectors: mkl_malloc(64) for 64-byte alignment.

#include <mkl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(sizeof(MKL_INT) == sizeof(int32_t),
               "MKL must use LP64 linking (32-bit MKL_INT). Link with mkl_intel_lp64, not ilp64.");

typedef struct {
    sparse_matrix_t A;
    struct matrix_descr descr;
    int32_t nrows;
    int32_t ncols;
    int32_t nnz;
    double* x;
    double* y;
} MKLBenchContext;

MKLBenchContext* libmkl_spmv_setup(
    int32_t nrows,
    int32_t ncols,
    int32_t nnz,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const double *values
) {
    MKLBenchContext* ctx = (MKLBenchContext*)malloc(sizeof(MKLBenchContext));
    if (!ctx) return NULL;

    ctx->nrows = nrows;
    ctx->ncols = ncols;
    ctx->nnz = nnz;

    // We allocate x and y vectors internally since the Rust bench just passes ptrs anyway (or we can just reuse Rust vectors, but this avoids lifetimes)
    ctx->x = (double*)mkl_malloc(ncols * sizeof(double), 64);
    ctx->y = (double*)mkl_malloc(nrows * sizeof(double), 64);

    for (int i=0; i<ncols; ++i) ctx->x[i] = 1.0;
    for (int i=0; i<nrows; ++i) ctx->y[i] = 0.0;

    // Zero-copy CSR handle over caller's arrays
    sparse_status_t status = mkl_sparse_d_create_csr(
        &ctx->A,
        SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        (MKL_INT*)row_ptr,
        (MKL_INT*)(row_ptr + 1),
        (MKL_INT*)col_idx,
        (double*)values
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        mkl_free(ctx->x);
        mkl_free(ctx->y);
        free(ctx);
        return NULL;
    }

    ctx->descr = (struct matrix_descr){ .type = SPARSE_MATRIX_TYPE_GENERAL };
    // mkl_sparse_set_mv_hint enables format-specific IE optimizations but
    // effects are matrix-dependent: can help or hurt depending on structure.
    // mkl_sparse_set_mv_hint(ctx->A, SPARSE_OPERATION_NON_TRANSPOSE, ctx->descr, 1000);
    status = mkl_sparse_optimize(ctx->A);
    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[MKL] Warning: mkl_sparse_optimize (CSR) returned %d\n", status);
    }

    return ctx;
}

void libmkl_spmv_execute(MKLBenchContext* ctx) {
    // y = alpha * A * x + beta * y
    // alpha = 1.0, beta = 1.0  =>  y += A*x (Accum::Add equivalent)
    mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE, 
        1.0, 
        ctx->A, 
        ctx->descr, 
        ctx->x, 
        1.0, 
        ctx->y
    );
}

void libmkl_spmv_get_y(MKLBenchContext* ctx, double* out, int32_t len) {
    int32_t n = len < ctx->nrows ? len : ctx->nrows;
    for (int32_t i = 0; i < n; i++) out[i] = ctx->y[i];
}

void libmkl_spmv_teardown(MKLBenchContext* ctx) {
    if (ctx) {
        if (ctx->A) {
            mkl_sparse_destroy(ctx->A);
        }
        if (ctx->x) mkl_free(ctx->x);
        if (ctx->y) mkl_free(ctx->y);
        free(ctx);
    }
}

/* ------------------------------------------------------------------ */
/*  CSC variant — Inspection-Execution API with CSC input format      */
/* ------------------------------------------------------------------ */

typedef struct {
    sparse_matrix_t A;
    struct matrix_descr descr;
    int32_t nrows;
    int32_t ncols;
    int32_t nnz;
    double* x;
    double* y;
} MKLCscBenchContext;

MKLCscBenchContext* libmkl_csc_spmv_setup(
    int32_t nrows,
    int32_t ncols,
    int32_t nnz,
    const int32_t *col_ptr,
    const int32_t *row_idx,
    const double *values
) {
    MKLCscBenchContext* ctx = (MKLCscBenchContext*)malloc(sizeof(MKLCscBenchContext));
    if (!ctx) return NULL;

    ctx->nrows = nrows;
    ctx->ncols = ncols;
    ctx->nnz = nnz;

    ctx->x = (double*)mkl_malloc(ncols * sizeof(double), 64);
    ctx->y = (double*)mkl_malloc(nrows * sizeof(double), 64);

    for (int i=0; i<ncols; ++i) ctx->x[i] = 1.0;
    for (int i=0; i<nrows; ++i) ctx->y[i] = 0.0;

    // Zero-copy CSC handle over caller's arrays
    sparse_status_t status = mkl_sparse_d_create_csc(
        &ctx->A,
        SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        (MKL_INT*)col_ptr,
        (MKL_INT*)(col_ptr + 1),
        (MKL_INT*)row_idx,
        (double*)values
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        mkl_free(ctx->x);
        mkl_free(ctx->y);
        free(ctx);
        return NULL;
    }

    ctx->descr = (struct matrix_descr){ .type = SPARSE_MATRIX_TYPE_GENERAL };
    // mkl_sparse_set_mv_hint(ctx->A, SPARSE_OPERATION_NON_TRANSPOSE, ctx->descr, 1000);
    status = mkl_sparse_optimize(ctx->A);
    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[MKL] Warning: mkl_sparse_optimize (CSC) returned %d\n", status);
    }

    return ctx;
}

void libmkl_csc_spmv_execute(MKLCscBenchContext* ctx) {
    // y = alpha * A * x + beta * y
    // alpha = 1.0, beta = 1.0  =>  y += A*x
    mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        ctx->A,
        ctx->descr,
        ctx->x,
        1.0,
        ctx->y
    );
}

void libmkl_csc_spmv_get_y(MKLCscBenchContext* ctx, double* out, int32_t len) {
    int32_t n = len < ctx->nrows ? len : ctx->nrows;
    for (int32_t i = 0; i < n; i++) out[i] = ctx->y[i];
}

void libmkl_csc_spmv_teardown(MKLCscBenchContext* ctx) {
    if (ctx) {
        if (ctx->A) {
            mkl_sparse_destroy(ctx->A);
        }
        if (ctx->x) mkl_free(ctx->x);
        if (ctx->y) mkl_free(ctx->y);
        free(ctx);
    }
}
