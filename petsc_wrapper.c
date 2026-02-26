// Minimal C wrapper to interface PETSc with Rust FFI.
// 
// Operates on pre-allocated raw memory buffers constructed by Rust
// to guarantee a zero-copy architecture for apples-to-apples comparisons.

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
    Vec y_init;
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
    VecDuplicate(ctx->y, &ctx->y_init);

    // Initialize vectors (same logic as before: x=1.0, y_init=i*1e-9)
    VecSet(ctx->x, 1.0);
    PetscScalar *yarr;
    VecGetArray(ctx->y_init, &yarr);
    for (int32_t i = 0; i < nrows; i++) {
        yarr[i] = (PetscScalar)i * 1e-9;
    }
    VecRestoreArray(ctx->y_init, &yarr);

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
    // y = y_init
    VecCopy(ctx->y_init, ctx->y);
    // y = A*x + y
    MatMultAdd(ctx->A, ctx->x, ctx->y, ctx->y);
}

void libpetsc_spmv_teardown(PetscBenchContext* ctx) {
    MatDestroy(&ctx->A);
    VecDestroy(&ctx->x);
    VecDestroy(&ctx->y);
    VecDestroy(&ctx->y_init);
    free(ctx);
    // PetscFinalize can be called at the very end of the rust program
}
