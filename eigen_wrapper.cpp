// Eigen FFI wrapper for SpMV benchmarking.
// Zero-copy via Eigen::Map over CSC/CSR arrays.
// Header-only: the SpMV kernel is compiled with our flags (-O3 -ffast-math
// -flto), unlike PETSc/MKL whose kernels are in pre-compiled libraries.

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdint.h>
#include <stdlib.h>

extern "C" {

// Benchmark context holding Eigen objects
typedef struct {
    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>>* A;
    Eigen::VectorXd* x;
    Eigen::VectorXd* y;
} EigenBenchContext;

EigenBenchContext* libeigen_spmv_setup(
    int32_t nrows,
    int32_t ncols,
    int32_t nnz,
    const int32_t* col_ptr,
    const int32_t* row_idx,
    const double* values
) {
    EigenBenchContext* ctx = new EigenBenchContext;
    
    // Map the raw CSC arrays into a const Eigen SparseMatrix (zero-copy)
    ctx->A = new Eigen::Map<const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>>(
        nrows, ncols, nnz, col_ptr, row_idx, values
    );

    ctx->x = new Eigen::VectorXd(ncols);
    ctx->x->setConstant(1.0);

    ctx->y = new Eigen::VectorXd(nrows);
    ctx->y->setZero();

    return ctx;
}

void libeigen_spmv_execute(EigenBenchContext* ctx) {
    // y += A * x
    *(ctx->y) += (*(ctx->A)) * (*(ctx->x));
}

void libeigen_spmv_get_y(EigenBenchContext* ctx, double* out, int32_t len) {
    int32_t n = len < (int32_t)ctx->y->size() ? len : (int32_t)ctx->y->size();
    for (int32_t i = 0; i < n; i++) out[i] = (*(ctx->y))[i];
}

void libeigen_spmv_teardown(EigenBenchContext* ctx) {
    delete ctx->A;
    delete ctx->x;
    delete ctx->y;
    delete ctx;
}

// Benchmark context holding Eigen objects (CSR / RowMajor)
typedef struct {
    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>>* A;
    Eigen::VectorXd* x;
    Eigen::VectorXd* y;
} EigenCsrBenchContext;

EigenCsrBenchContext* libeigen_csr_spmv_setup(
    int32_t nrows,
    int32_t ncols,
    int32_t nnz,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values
) {
    EigenCsrBenchContext* ctx = new EigenCsrBenchContext;

    // Map the raw CSR arrays into a const Eigen SparseMatrix (zero-copy)
    ctx->A = new Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>>(
        nrows, ncols, nnz, row_ptr, col_idx, values
    );

    ctx->x = new Eigen::VectorXd(ncols);
    ctx->x->setConstant(1.0);

    ctx->y = new Eigen::VectorXd(nrows);
    ctx->y->setZero();

    return ctx;
}

void libeigen_csr_spmv_execute(EigenCsrBenchContext* ctx) {
    // y += A * x
    *(ctx->y) += (*(ctx->A)) * (*(ctx->x));
}

void libeigen_csr_spmv_get_y(EigenCsrBenchContext* ctx, double* out, int32_t len) {
    int32_t n = len < (int32_t)ctx->y->size() ? len : (int32_t)ctx->y->size();
    for (int32_t i = 0; i < n; i++) out[i] = (*(ctx->y))[i];
}

void libeigen_csr_spmv_teardown(EigenCsrBenchContext* ctx) {
    delete ctx->A;
    delete ctx->x;
    delete ctx->y;
    delete ctx;
}

} // extern "C"
