// PETSc FFI wrapper for SpMV benchmarking.
// Matrix: zero-copy via MatCreateSeqAIJWithArrays. Vectors: PETSc-allocated.
// The actual SpMV kernel (MatMult_SeqAIJ) lives in libpetsc.so, compiled
// by spack, not by our build.rs flags.

#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
#include <stdint.h>
#include <stdio.h>

_Static_assert(sizeof(PetscInt) == sizeof(int32_t),
               "PETSc must be built with 32-bit indices (--with-64-bit-indices=0)");
_Static_assert(sizeof(PetscScalar) == sizeof(double),
               "PETSc must be built with real scalars (not complex)");

typedef struct {
    Mat A;
    Vec x;
    Vec y;
    int32_t nrows;
    int32_t ncols;
    int32_t nnz;
} PetscBenchContext;

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

    PetscBenchContext* ctx = malloc(sizeof *ctx);
    if (!ctx) return NULL;
    ctx->nrows = nrows;
    ctx->ncols = ncols;
    ctx->nnz = nnz;

    VecCreateSeq(PETSC_COMM_SELF, ncols, &ctx->x);
    VecCreateSeq(PETSC_COMM_SELF, nrows, &ctx->y);
    VecSet(ctx->x, 1.0);
    VecSet(ctx->y, 0.0);

    // PETSc API does not declare these parameters const; the zero-copy
    // contract guarantees PETSc will not modify the caller's arrays.
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

void libpetsc_spmv_get_y(PetscBenchContext* ctx, double *restrict out, int32_t len) {
    const PetscScalar *y_array;
    VecGetArrayRead(ctx->y, &y_array);
    const int32_t n = len < ctx->nrows ? len : ctx->nrows;
    for (int32_t i = 0; i < n; i++) out[i] = y_array[i];
    VecRestoreArrayRead(ctx->y, &y_array);
}

void libpetsc_spmv_teardown(PetscBenchContext* ctx) {
    if (!ctx) return;
    MatDestroy(&ctx->A);
    VecDestroy(&ctx->x);
    VecDestroy(&ctx->y);
    free(ctx);
}
