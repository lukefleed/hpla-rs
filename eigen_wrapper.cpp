// Minimal C++ wrapper to interface Eigen with Rust FFI.
// 
// Operates on pre-allocated raw memory buffers constructed by Rust
// to guarantee a zero-copy architecture (CSC) for fair comparisons.

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
    Eigen::VectorXd* y_init;
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
    ctx->y_init = new Eigen::VectorXd(nrows);
    
    for (int32_t i = 0; i < nrows; ++i) {
        (*(ctx->y_init))(i) = (double)i * 1e-9;
    }

    return ctx;
}

void libeigen_spmv_execute(EigenBenchContext* ctx) {
    // y += A * x
    *(ctx->y) += (*(ctx->A)) * (*(ctx->x));
}

void libeigen_spmv_teardown(EigenBenchContext* ctx) {
    delete ctx->A;
    delete ctx->x;
    delete ctx->y;
    delete ctx->y_init;
    delete ctx;
}

} // extern "C"
