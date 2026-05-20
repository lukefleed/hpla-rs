// PETSc FFI wrapper for two-pass Lanczos computing exp(-A)b.
// Zero-copy via MatCreateSeqAIJWithArrays over CSR arrays. All O(n)
// buffers are pre-allocated in setup; execute reuses them without any
// n-scaled allocation. Memory footprint is O(n), not O(nk).
//
// Rolling buffer rotation uses VecResetArray/VecPlaceArray (O(1) pointer
// swap) instead of VecSwap (O(n) BLASswap). The three Vec shells are
// created once via VecCreateSeqWithArray and repointed each step.

#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
#include <petscblaslapack.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

_Static_assert(sizeof(PetscInt) == sizeof(int32_t),
               "PETSc must be built with 32-bit indices (--with-64-bit-indices=0)");
_Static_assert(sizeof(PetscScalar) == sizeof(double),
               "PETSc must be built with real scalars (not complex)");

static const double breakdown_tol = DBL_EPSILON * 1000.0;

typedef struct {
    Mat     A;
    Vec     v_prev;
    Vec     v_curr;
    Vec     work;
    Vec     x;

    double *raw_prev;
    double *raw_curr;
    double *raw_work;

    double *b_copy;
    double *alphas;
    double *betas;

    // Pre-allocated LAPACK workspace (avoids malloc in execute).
    double *eig_d;
    double *eig_e;
    double *eig_z;
    double *eig_work;
    double *eig_weights;
    double *eig_g;

    int32_t n;
    int32_t krylov_dim;
    int32_t steps_taken;
} PetscLanczosTwoPassCtx;

PetscLanczosTwoPassCtx* libpetsc_lanczos_two_pass_setup(
    int32_t nrows,
    int32_t ncols,
    int32_t nnz,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const double  *values,
    const double  *b,
    int32_t        krylov_dim)
{
    PetscBool initialized;
    PetscInitialized(&initialized);
    if (!initialized) {
        PetscInitializeNoArguments();
    }

    (void)nnz; // PETSc derives nnz from row_ptr[nrows].
    if (nrows <= 0 || krylov_dim <= 0) return NULL;

    PetscLanczosTwoPassCtx *ctx = malloc(sizeof *ctx);
    if (!ctx) return NULL;

    ctx->n = nrows;
    ctx->krylov_dim = krylov_dim;
    ctx->steps_taken = 0;

    ctx->raw_prev = calloc(nrows, sizeof(double));
    ctx->raw_curr = calloc(nrows, sizeof(double));
    ctx->raw_work = calloc(nrows, sizeof(double));
    ctx->b_copy   = malloc(nrows * sizeof(double));
    ctx->alphas   = malloc(krylov_dim * sizeof(double));
    ctx->betas    = malloc(krylov_dim * sizeof(double));

    // Pre-allocate LAPACK workspace sized to krylov_dim.
    const int32_t lwork_sz = (krylov_dim > 1) ? (2 * krylov_dim - 2) : 1;
    ctx->eig_d       = malloc(krylov_dim * sizeof(double));
    ctx->eig_e       = malloc(krylov_dim * sizeof(double));
    ctx->eig_z       = malloc((size_t)krylov_dim * krylov_dim * sizeof(double));
    ctx->eig_work    = malloc(lwork_sz * sizeof(double));
    ctx->eig_weights = malloc(krylov_dim * sizeof(double));
    ctx->eig_g       = malloc(krylov_dim * sizeof(double));

    if (!ctx->raw_prev || !ctx->raw_curr || !ctx->raw_work ||
        !ctx->b_copy || !ctx->alphas || !ctx->betas ||
        !ctx->eig_d || !ctx->eig_e || !ctx->eig_z || !ctx->eig_work ||
        !ctx->eig_weights || !ctx->eig_g) {
        free(ctx->raw_prev); free(ctx->raw_curr); free(ctx->raw_work);
        free(ctx->b_copy); free(ctx->alphas); free(ctx->betas);
        free(ctx->eig_d); free(ctx->eig_e); free(ctx->eig_z);
        free(ctx->eig_work); free(ctx->eig_weights); free(ctx->eig_g);
        free(ctx);
        return NULL;
    }

    memcpy(ctx->b_copy, b, nrows * sizeof(double));

    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, nrows, ctx->raw_prev, &ctx->v_prev);
    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, nrows, ctx->raw_curr, &ctx->v_curr);
    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, nrows, ctx->raw_work, &ctx->work);
    VecCreateSeq(PETSC_COMM_SELF, nrows, &ctx->x);

    // PETSc API does not declare these parameters const; the zero-copy
    // contract guarantees PETSc will not modify the caller's arrays.
    MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nrows, ncols,
                              (PetscInt *)row_ptr, (PetscInt *)col_idx,
                              (PetscScalar *)values, &ctx->A);

    return ctx;
}

void libpetsc_lanczos_two_pass_disable_inodes(PetscLanczosTwoPassCtx *ctx) {
    if (!ctx) return;
    MatSetOption(ctx->A, MAT_USE_INODES, PETSC_FALSE);
}

void libpetsc_lanczos_two_pass_execute(PetscLanczosTwoPassCtx *ctx) {
    if (!ctx || ctx->n <= 0 || ctx->krylov_dim <= 0) {
        if (ctx) { VecSet(ctx->x, 0.0); ctx->steps_taken = 0; }
        return;
    }

    const int32_t n = ctx->n;
    const int32_t k = ctx->krylov_dim;

    // Reset rolling buffers. raw_curr is not zeroed because it is
    // immediately overwritten with b / ||b|| below.
    memset(ctx->raw_prev, 0, n * sizeof(double));
    memset(ctx->raw_work, 0, n * sizeof(double));
    VecResetArray(ctx->v_prev);
    VecResetArray(ctx->v_curr);
    VecResetArray(ctx->work);
    VecPlaceArray(ctx->v_prev, ctx->raw_prev);
    VecPlaceArray(ctx->v_curr, ctx->raw_curr);
    VecPlaceArray(ctx->work,   ctx->raw_work);
    VecSet(ctx->x, 0.0);
    memset(ctx->alphas, 0, k * sizeof(double));
    memset(ctx->betas,  0, k * sizeof(double));
    ctx->steps_taken = 0;

    // b_norm = ||b||.
    double b_norm_sq = 0.0;
    for (int32_t i = 0; i < n; i++) {
        b_norm_sq += ctx->b_copy[i] * ctx->b_copy[i];
    }
    const double b_norm = sqrt(b_norm_sq);
    if (b_norm <= breakdown_tol) return;

    const double inv_b_norm = 1.0 / b_norm;

    // v_curr = b / ||b||.
    for (int32_t i = 0; i < n; i++) {
        ctx->raw_curr[i] = ctx->b_copy[i] * inv_b_norm;
    }

    // ---- Pass 1: build tridiagonal T_k ----
    double beta_prev = 0.0;

    for (int32_t i = 0; i < k; i++) {
        MatMult(ctx->A, ctx->v_curr, ctx->work);

        if (i > 0) {
            VecAXPY(ctx->work, -beta_prev, ctx->v_prev);
        }

        PetscScalar alpha_s;
        VecDot(ctx->v_curr, ctx->work, &alpha_s);
        const double alpha = (double)alpha_s;
        ctx->alphas[i] = alpha;

        VecAXPY(ctx->work, -alpha, ctx->v_curr);

        PetscReal beta_r;
        VecNorm(ctx->work, NORM_2, &beta_r);
        const double beta = (double)beta_r;

        if (beta <= breakdown_tol) {
            ctx->steps_taken = i + 1;
            break;
        }

        if (i < k - 1) {
            ctx->betas[i] = beta;
        }

        VecScale(ctx->work, 1.0 / beta);

        double *tmp   = ctx->raw_prev;
        ctx->raw_prev = ctx->raw_curr;
        ctx->raw_curr = ctx->raw_work;
        ctx->raw_work = tmp;
        VecResetArray(ctx->v_prev);
        VecResetArray(ctx->v_curr);
        VecResetArray(ctx->work);
        VecPlaceArray(ctx->v_prev, ctx->raw_prev);
        VecPlaceArray(ctx->v_curr, ctx->raw_curr);
        VecPlaceArray(ctx->work,   ctx->raw_work);

        beta_prev = beta;
        ctx->steps_taken = i + 1;
    }

    // ---- exp(-T_m) * e_1 via eigendecomposition ----
    // All workspace buffers are pre-allocated in setup (sized to krylov_dim).
    const int32_t m = ctx->steps_taken;
    if (m <= 0) return;

    double *d       = ctx->eig_d;
    double *e       = ctx->eig_e;
    double *z       = ctx->eig_z;
    double *weights = ctx->eig_weights;
    double *g       = ctx->eig_g;

    for (int32_t j = 0; j < m; j++)     d[j] = -ctx->alphas[j];
    for (int32_t j = 0; j < m - 1; j++) e[j] = -ctx->betas[j];

    PetscBLASInt bm = (PetscBLASInt)m;
    PetscBLASInt info = 0;
    LAPACKREALstev_("V", &bm, d, e, z, &bm, ctx->eig_work, &info);

    if (info != 0) {
        VecSet(ctx->x, 0.0);
        return;
    }

    // weights[j] = exp(lambda_j) * Q[0, j].
    for (int32_t j = 0; j < m; j++) {
        weights[j] = exp(d[j]) * z[0 + (size_t)j * m];
    }

    // g = b_norm * Q * weights via BLAS dgemv.
    {
        PetscBLASInt one_i = 1;
        double zero_d = 0.0;
        BLASgemv_("N", &bm, &bm, &b_norm, z, &bm,
                  weights, &one_i, &zero_d, g, &one_i);
    }

    // ---- Pass 2: reconstruct x = V_m * g without V_m ----
    memset(ctx->raw_prev, 0, n * sizeof(double));
    for (int32_t i = 0; i < n; i++) {
        ctx->raw_curr[i] = ctx->b_copy[i] * inv_b_norm;
    }
    memset(ctx->raw_work, 0, n * sizeof(double));
    VecResetArray(ctx->v_prev);
    VecResetArray(ctx->v_curr);
    VecResetArray(ctx->work);
    VecPlaceArray(ctx->v_prev, ctx->raw_prev);
    VecPlaceArray(ctx->v_curr, ctx->raw_curr);
    VecPlaceArray(ctx->work,   ctx->raw_work);

    // x = g[0] * v_curr.
    VecSet(ctx->x, 0.0);
    VecAXPY(ctx->x, g[0], ctx->v_curr);

    for (int32_t j = 0; j < m - 1; j++) {
        MatMult(ctx->A, ctx->v_curr, ctx->work);
        VecAXPY(ctx->work, -ctx->alphas[j], ctx->v_curr);
        if (j > 0) {
            VecAXPY(ctx->work, -ctx->betas[j - 1], ctx->v_prev);
        }

        const double beta_j = ctx->betas[j];
        if (beta_j <= breakdown_tol) break;
        VecScale(ctx->work, 1.0 / beta_j);

        VecAXPY(ctx->x, g[j + 1], ctx->work);

        double *tmp   = ctx->raw_prev;
        ctx->raw_prev = ctx->raw_curr;
        ctx->raw_curr = ctx->raw_work;
        ctx->raw_work = tmp;
        VecResetArray(ctx->v_prev);
        VecResetArray(ctx->v_curr);
        VecResetArray(ctx->work);
        VecPlaceArray(ctx->v_prev, ctx->raw_prev);
        VecPlaceArray(ctx->v_curr, ctx->raw_curr);
        VecPlaceArray(ctx->work,   ctx->raw_work);
    }
}

void libpetsc_lanczos_two_pass_get_y(PetscLanczosTwoPassCtx *ctx,
                                     double *restrict out, int32_t len) {
    if (!ctx || !out || len <= 0) return;
    const PetscScalar *x_array;
    VecGetArrayRead(ctx->x, &x_array);
    const int32_t count = (len < ctx->n) ? len : ctx->n;
    for (int32_t i = 0; i < count; i++) out[i] = x_array[i];
    VecRestoreArrayRead(ctx->x, &x_array);
}

void libpetsc_lanczos_two_pass_teardown(PetscLanczosTwoPassCtx *ctx) {
    if (!ctx) return;
    MatDestroy(&ctx->A);
    VecDestroy(&ctx->v_prev);
    VecDestroy(&ctx->v_curr);
    VecDestroy(&ctx->work);
    VecDestroy(&ctx->x);
    free(ctx->raw_prev);
    free(ctx->raw_curr);
    free(ctx->raw_work);
    free(ctx->b_copy);
    free(ctx->alphas);
    free(ctx->betas);
    free(ctx->eig_d);
    free(ctx->eig_e);
    free(ctx->eig_z);
    free(ctx->eig_work);
    free(ctx->eig_weights);
    free(ctx->eig_g);
    free(ctx);
}
