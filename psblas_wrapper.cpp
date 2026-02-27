#include "psb_base_cbind.h"
#include <mpi.h>
#include <sched.h>
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
libpsblas_spmv_setup(int nrows, int ncols, int nnz,
                     const int *row_ptr,  // CSR row pointers (0-based)
                     const int *col_idx,  // CSR column indices (0-based)
                     const double *values // CSR values
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

  // Preserve the user's taskset -c 3 CPU affinity! OpenMPI hijacks it by
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
  // survive!
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

  // Assemble the descriptor
  info = psb_c_cdasb(ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: cdasb failed with %d\n", info);
    exit(1);
  }

  // 3. Setup vectors
  ctx->xh = psb_c_new_dvector();
  ctx->yh = psb_c_new_dvector();
  psb_c_dgeall(ctx->xh, ctx->cdh);
  psb_c_dgeall(ctx->yh, ctx->cdh);
  psb_c_dgeasb(ctx->xh, ctx->cdh);
  psb_c_dgeasb(ctx->yh, ctx->cdh);

  // Pre-fill x with 1.0 (as the benchmark expects)
  psb_c_dvect_set_scal(ctx->xh, 1.0);
  // Pre-fill y with 0.0 or the expected init
  psb_c_dvect_set_scal(ctx->yh, 0.0);

  // 4. Setup sparse matrix
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

  // Assemble sparse matrix (forces internal optimization and layout CSR)
  info = psb_c_dspasb_opt(ctx->ah, ctx->cdh, "CSR", 0, 0);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: dspasb_opt failed with %d\n", info);
    exit(1);
  }

  return ctx;
}

void libpsblas_spmv_execute(psblas_context_t *ctx) {
  // y = 1.0 * A * x + 1.0 * y
  // We use exactly 1.0 for alpha and 1.0 for beta as per benchmark standards
  int info = psb_c_dspmm(1.0, ctx->ah, ctx->xh, 1.0, ctx->yh, ctx->cdh);
  if (info != 0) {
    fprintf(stderr, "[PSBLAS] Fatal: dspmm failed with %d\n", info);
  }
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
  // DANGEROUS: Do NOT call psb_c_exit here!
  // psb_c_exit internally calls MPI_Finalize.
  // Criterion loops over multiple matrices, creating and destroying contexts.
  // Calling MPI_Init -> MPI_Finalize -> MPI_Init in the same process is illegal
  // in MPI and will crash. The OS will reap the MPI environment when the cargo
  // bench process exits.

  free(ctx->cctxt);
  free(ctx);
}

} // extern "C"
