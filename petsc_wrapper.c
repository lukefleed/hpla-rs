// Minimal C wrapper to interface PETSc with Rust FFI.
// 
// Operates on pre-allocated raw memory buffers constructed by Rust
// to guarantee a zero-copy architecture for fair comparisons.

#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
#include <stdint.h>
#include <stdio.h>

// Forward declare the benchmark context holding PETSc objects
typedef struct {
    Mat A;
    Vec x;
    Vec y;
    int32_t nrows;
    int32_t ncols;
    int32_t nnz;
} PetscBenchContext;

// Initialize PETSc and create the matrix from raw CSR pointers
// Note: We expect 32-bit indices (PetscInt typically 32-bit depending on config, we enforce u32 in rust)
PetscBenchContext* libpetsc_spmv_setup(
    int32_t nrows, 
    int32_t ncols, 
    int32_t nnz,
    const int32_t *row_ptr, 
    const int32_t *col_idx, 
    const double *values,
    int disable_inode)
{
    PetscBool initialized;
    PetscInitialized(&initialized);
    if (!initialized) {
        PetscInitializeNoArguments();
    }

    PetscBenchContext* ctx = (PetscBenchContext*)malloc(sizeof(PetscBenchContext));
    ctx->nrows = nrows;
    ctx->ncols = ncols;
    ctx->nnz = nnz;

    // Create Vectors
    VecCreateSeq(PETSC_COMM_SELF, ncols, &ctx->x);
    VecCreateSeq(PETSC_COMM_SELF, nrows, &ctx->y);
    // Initialize vectors (x=1.0, y=0.0)
    VecSet(ctx->x, 1.0);
    VecSet(ctx->y, 0.0);

    // Create Matrix from raw CSR
    /*
        Petsc's MatCreateSeqAIJWithArrays uses the provided memory directly!
        This is zero-copy and identical to Rust.
    */
    MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, ncols, (PetscInt*)row_ptr, (PetscInt*)col_idx, (PetscScalar*)values, &ctx->A);

    if (disable_inode) {
        MatSetOption(ctx->A, MAT_USE_INODES, PETSC_FALSE);
    }

    return ctx;
}

void libpetsc_spmv_execute(PetscBenchContext* ctx) {
    // y = A * x + y
    MatMultAdd(ctx->A, ctx->x, ctx->y, ctx->y);
}

void libpetsc_spmv_teardown(PetscBenchContext* ctx) {
    MatDestroy(&ctx->A);
    VecDestroy(&ctx->x);
    VecDestroy(&ctx->y);
    free(ctx);
    // PetscFinalize can be called at the very end of the rust program
}
