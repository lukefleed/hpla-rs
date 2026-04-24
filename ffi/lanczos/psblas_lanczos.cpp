// PSBLAS FFI wrapper for SpMV benchmarking (Fortran via C bindings + MPI).
// NOT zero-copy: elements inserted via psb_c_dspins(), assembled into internal
// CSR or CSC via psb_c_dspasb_opt(). Copy happens during setup only.
// MPI initialized once; CPU affinity saved/restored around MPI_Init.
// psb_c_exit_ctxt frees the Fortran context without MPI_Finalize so Criterion
// can loop matrices.

// Include <complex> before psb_base_cbind.h: the PSBLAS header opens
// extern "C" { before #include <complex>, which is illegal in C++.
// Pre-including <complex> here makes the second inclusion a no-op via
// its include guard.
#include <complex>
#include "psb_base_cbind.h"
#include <mpi.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Opaque context struct we pass back to Rust
typedef struct {
  psb_c_ctxt *cctxt;
  psb_c_descriptor *cdh;
  psb_c_dspmat *ah;
  psb_c_dvector *xh;
  psb_c_dvector *yh;
  psb_l_t *vl;
} psblas_context_t;

extern "C" {

psblas_context_t *
libpsblas_spmv_setup(int32_t nrows, int32_t ncols, int32_t nnz,
                     const int32_t *row_ptr,  // CSR row pointers (0-based)
                     const int32_t *col_idx,  // CSR column indices (0-based)
                     const double *values     // CSR values
) {
  (void)nnz; // Suppress unused parameter warning

  psblas_context_t *ctx = (psblas_context_t *)malloc(sizeof(psblas_context_t));
  if (!ctx)
    return NULL;

  // 1. Initialize PSBLAS Context and MPI safely
  // Prevent OpenMP and OpenMPI from spawning background idle threads that steal
  // L3 cache from Faer and Eigen during the Criterion loop process lifecycle.
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("OMPI_MCA_mpi_yield_when_idle", "1", 1);

  // Preserve the caller's CPU affinity (taskset -c 0). OpenMPI resets it by
  // default.
  cpu_set_t cpuset;
  sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);

  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
  }

  // Restore the original affinity so the rest of the Criterion benchmarks
  // survive.
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

  ctx->cctxt = psb_c_new_ctxt();
  // We must convert the C MPI_COMM_WORLD to a Fortran integer handle
  // so PSBLAS can bind it natively without spinning up its own
  // sub-communicators.
  MPI_Fint f_comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  psb_c_init_from_fint(ctx->cctxt, f_comm);

  // Provide base 0 to match C/Rust conventions
  psb_c_set_index_base(0);

  // 2. Global descriptor for replicated setup (since we are strictly
  // single-threaded here)
  ctx->cdh = psb_c_new_descriptor();

  // Allocate local-to-global mapping array (identity map for single processor)
  ctx->vl = (psb_l_t *)malloc(nrows * sizeof(psb_l_t));
  for (int i = 0; i < nrows; i++) {
    ctx->vl[i] = i;
  }

  // Initialize the topology descriptor
  int info = psb_c_cdall_vl(nrows, ctx->vl, *(ctx->cctxt), ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: cdall failed with %d\n", info);
    exit(1);
  }

  info = psb_c_cdasb(ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: cdasb failed with %d\n", info);
    exit(1);
  }

  ctx->xh = psb_c_new_dvector();
  ctx->yh = psb_c_new_dvector();
  psb_c_dgeall(ctx->xh, ctx->cdh);
  psb_c_dgeall(ctx->yh, ctx->cdh);
  psb_c_dgeasb(ctx->xh, ctx->cdh);
  psb_c_dgeasb(ctx->yh, ctx->cdh);

  psb_c_dvect_set_scal(ctx->xh, 1.0);
  psb_c_dvect_set_scal(ctx->yh, 0.0);

  ctx->ah = psb_c_new_dspmat();
  psb_c_dspall(ctx->ah, ctx->cdh);

  // Insert elements row by row
  // PSBLAS C API requires explicit arrays of row_indices and col_indices per
  // insertion block. For max speed setup we can insert row by row.
  psb_l_t *temp_iw = (psb_l_t *)malloc(ncols * sizeof(psb_l_t));
  psb_l_t *temp_jw = (psb_l_t *)malloc(ncols * sizeof(psb_l_t));

  for (int i = 0; i < nrows; i++) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int nz_in_row = end - start;

    if (nz_in_row > 0) {
      for (int k = 0; k < nz_in_row; k++) {
        temp_iw[k] = (psb_l_t)i;
        temp_jw[k] = (psb_l_t)col_idx[start + k];
      }
      psb_c_dspins(nz_in_row, temp_iw, temp_jw, &values[start], ctx->ah,
                   ctx->cdh);
    }
  }
  free(temp_iw);
  free(temp_jw);

  // Assemble into internal CSR
  info = psb_c_dspasb_opt(ctx->ah, ctx->cdh, "CSR", 0, 0);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: dspasb_opt failed with %d\n", info);
    exit(1);
  }

  return ctx;
}

void libpsblas_spmv_execute(psblas_context_t *ctx, double tol, int max_iters) {
  // y = alpha * A * x + beta * y
  // alpha = 1.0, beta = 1.0  =>  y += A*x
  // int info = psb_c_dspmm(1.0, ctx->ah, ctx->xh, 1.0, ctx->yh, ctx->cdh);
  int info = psb_c_dexpmv_twopass(ctx->ah, ctx->cdh, ctx->yh, ctx->xh, tol, max_iters);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: dspmm failed with %d\n", info);
  }
}

void libpsblas_spmv_get_y(psblas_context_t *ctx, double *out, int32_t len) {
    // psb_c_dvect_f_get_pnt returns a raw pointer to the internal Fortran
    // vector storage, avoiding an extra allocation + copy.
    double *yptr = psb_c_dvect_f_get_pnt(ctx->yh);
    int32_t nrows = psb_c_dvect_get_nrows(ctx->yh);
    int32_t n = len < nrows ? len : nrows;
    for (int32_t i = 0; i < n; i++) out[i] = yptr[i];
}

void libpsblas_spmv_teardown(psblas_context_t *ctx) {
  if (!ctx)
    return;

  psb_c_dgefree(ctx->xh, ctx->cdh);
  psb_c_dgefree(ctx->yh, ctx->cdh);
  psb_c_dspfree(ctx->ah, ctx->cdh);
  psb_c_cdfree(ctx->cdh);

  free(ctx->xh);
  free(ctx->yh);
  free(ctx->ah);
  free(ctx->cdh);
  free(ctx->vl);

  psb_c_barrier(*(ctx->cctxt));
  // psb_c_exit_ctxt cleans up the Fortran-side context and frees the
  // duplicated MPI communicator without calling MPI_Finalize (unlike
  // psb_c_exit, which would make subsequent MPI_Init illegal).
  psb_c_exit_ctxt(*(ctx->cctxt));

  free(ctx->cctxt);
  free(ctx);
}

psblas_context_t *
libpsblas_csc_spmv_setup(int32_t nrows, int32_t ncols, int32_t nnz,
                         const int32_t *col_ptr,
                         const int32_t *row_idx,
                         const double *values) {
  (void)nnz;

  psblas_context_t *ctx = (psblas_context_t *)malloc(sizeof(psblas_context_t));
  if (!ctx)
    return NULL;

  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("OMPI_MCA_mpi_yield_when_idle", "1", 1);

  cpu_set_t cpuset;
  sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);

  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
  }

  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

  ctx->cctxt = psb_c_new_ctxt();
  MPI_Fint f_comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  psb_c_init_from_fint(ctx->cctxt, f_comm);
  psb_c_set_index_base(0);

  ctx->cdh = psb_c_new_descriptor();
  ctx->vl = (psb_l_t *)malloc(nrows * sizeof(psb_l_t));
  for (int i = 0; i < nrows; i++)
    ctx->vl[i] = i;

  int info = psb_c_cdall_vl(nrows, ctx->vl, *(ctx->cctxt), ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: cdall failed with %d\n", info);
    exit(1);
  }

  info = psb_c_cdasb(ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: cdasb failed with %d\n", info);
    exit(1);
  }

  ctx->xh = psb_c_new_dvector();
  ctx->yh = psb_c_new_dvector();
  psb_c_dgeall(ctx->xh, ctx->cdh);
  psb_c_dgeall(ctx->yh, ctx->cdh);
  psb_c_dgeasb(ctx->xh, ctx->cdh);
  psb_c_dgeasb(ctx->yh, ctx->cdh);

  psb_c_dvect_set_scal(ctx->xh, 1.0);
  psb_c_dvect_set_scal(ctx->yh, 0.0);

  ctx->ah = psb_c_new_dspmat();
  psb_c_dspall(ctx->ah, ctx->cdh);

  // Insert elements column by column from CSC arrays
  psb_l_t *temp_iw = (psb_l_t *)malloc(nrows * sizeof(psb_l_t));
  psb_l_t *temp_jw = (psb_l_t *)malloc(nrows * sizeof(psb_l_t));

  for (int j = 0; j < ncols; j++) {
    int start = col_ptr[j];
    int end = col_ptr[j + 1];
    int nz_in_col = end - start;

    if (nz_in_col > 0) {
      for (int k = 0; k < nz_in_col; k++) {
        temp_iw[k] = (psb_l_t)row_idx[start + k];
        temp_jw[k] = (psb_l_t)j;
      }
      psb_c_dspins(nz_in_col, temp_iw, temp_jw, &values[start], ctx->ah,
                   ctx->cdh);
    }
  }
  free(temp_iw);
  free(temp_jw);

  // Assemble into internal CSC
  info = psb_c_dspasb_opt(ctx->ah, ctx->cdh, "CSC", 0, 0);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: dspasb_opt (CSC) failed with %d\n", info);
    exit(1);
  }

  return ctx;
}

} // extern "C"
