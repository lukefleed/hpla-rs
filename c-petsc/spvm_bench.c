/*
 * SpMV Benchmark: PETSc
 *
 * Operation: y = A*x + y
 * Compile: make spmv_bench
 */

static char help[] = "SpMV benchmark using PETSc\n";

#include <petsc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <ctype.h>

/* ========================================================================== */
/* Matrix Market Parser                                                       */
/* ========================================================================== */

typedef enum { MM_GENERAL, MM_SYMMETRIC, MM_SKEW, MM_HERMITIAN } MMSymmetry;

typedef struct {
    PetscInt    nrows, ncols, nnz_file;
    MMSymmetry  symmetry;
    PetscBool   is_pattern;
} MMHeader;

static PetscErrorCode MMParseHeader(FILE *f, MMHeader *hdr)
{
    char line[1024], banner[64], object[64], format[64], field[64], symm[64];

    PetscFunctionBeginUser;
    if (!fgets(line, sizeof(line), f)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot read banner");

    if (sscanf(line, "%63s %63s %63s %63s %63s", banner, object, format, field, symm) != 5)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Invalid Matrix Market banner");

    /* Convert to lowercase */
    for (char *p = banner; *p; p++) *p = (char)tolower(*p);
    for (char *p = object; *p; p++) *p = (char)tolower(*p);
    for (char *p = format; *p; p++) *p = (char)tolower(*p);
    for (char *p = field;  *p; p++) *p = (char)tolower(*p);
    for (char *p = symm;   *p; p++) *p = (char)tolower(*p);

    if (strncmp(banner, "%%matrixmarket", 14) != 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Not a Matrix Market file");
    if (strcmp(object, "matrix") != 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Only 'matrix' object supported");
    if (strcmp(format, "coordinate") != 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Only 'coordinate' format supported");

    hdr->is_pattern = (strcmp(field, "pattern") == 0) ? PETSC_TRUE : PETSC_FALSE;

    if (strcmp(symm, "general") == 0)           hdr->symmetry = MM_GENERAL;
    else if (strcmp(symm, "symmetric") == 0)    hdr->symmetry = MM_SYMMETRIC;
    else if (strcmp(symm, "skew-symmetric") == 0) hdr->symmetry = MM_SKEW;
    else if (strcmp(symm, "hermitian") == 0)    hdr->symmetry = MM_HERMITIAN;
    else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Unknown symmetry type");

    /* Skip comments */
    do {
        if (!fgets(line, sizeof(line), f))
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Unexpected EOF");
    } while (line[0] == '%');

    /* Parse dimensions */
    if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,
               &hdr->nrows, &hdr->ncols, &hdr->nnz_file) != 3)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot parse dimensions");

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadMatrixMarket(const char *filename, Mat *A)
{
    FILE       *f;
    MMHeader   hdr;
    PetscInt   *nnz_per_row;
    PetscInt   i, row, col;
    PetscReal  val;
    char       line[1024];

    PetscFunctionBeginUser;
    f = fopen(filename, "r");
    if (!f) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", filename);

    PetscCall(MMParseHeader(f, &hdr));

    /* First pass: count nnz per row for preallocation */
    PetscCall(PetscCalloc1(hdr.nrows, &nnz_per_row));

    long data_start = ftell(f);
    for (i = 0; i < hdr.nnz_file; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        if (hdr.is_pattern) {
            if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT, &row, &col) != 2) continue;
        } else {
            if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %lf", &row, &col, &val) < 2) continue;
        }
        row--; col--;  /* 1-based to 0-based */
        nnz_per_row[row]++;
        if (hdr.symmetry != MM_GENERAL && row != col) {
            nnz_per_row[col]++;
        }
    }

    /* Create matrix with preallocation */
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, hdr.nrows, hdr.ncols, 0, nnz_per_row, A));
    PetscCall(MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

    /* Second pass: insert values */
    fseek(f, data_start, SEEK_SET);
    for (i = 0; i < hdr.nnz_file; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        if (hdr.is_pattern) {
            if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT, &row, &col) != 2) continue;
            val = 1.0;
        } else {
            if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %lf", &row, &col, &val) < 2) continue;
            if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %lf", &row, &col, &val) == 2) val = 1.0;
        }
        row--; col--;

        PetscCall(MatSetValue(*A, row, col, val, ADD_VALUES));
        if (hdr.symmetry == MM_SYMMETRIC && row != col) {
            PetscCall(MatSetValue(*A, col, row, val, ADD_VALUES));
        } else if (hdr.symmetry == MM_SKEW && row != col) {
            PetscCall(MatSetValue(*A, col, row, -val, ADD_VALUES));
        } else if (hdr.symmetry == MM_HERMITIAN && row != col) {
            PetscCall(MatSetValue(*A, col, row, val, ADD_VALUES)); /* Real case */
        }
    }

    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscFree(nnz_per_row));
    fclose(f);
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/* Timing & Statistics                                                        */
/* ========================================================================== */

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void ComputeStats(double *times, int n, double *median, double *mean, double *stddev, double *tmin, double *tmax)
{
    qsort(times, n, sizeof(double), cmp_double);
    *tmin = times[0];
    *tmax = times[n - 1];
    *median = (n % 2 == 0) ? (times[n/2 - 1] + times[n/2]) / 2.0 : times[n/2];

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += times[i];
    *mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; i++) var += (times[i] - *mean) * (times[i] - *mean);
    *stddev = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
}

/* ========================================================================== */
/* Benchmark                                                                  */
/* ========================================================================== */

typedef struct {
    char        name[256];
    PetscInt    rows, cols, nnz;
    double      median_s, mean_s, std_s, min_s, max_s;
    double      gflops, bw_gbs;
    int         iters;
} BenchResult;

static PetscErrorCode RunBenchmark(const char *filepath, int warmup, int min_iters, double min_time, BenchResult *result)
{
    Mat         A;
    Vec         x, y, y_init;
    PetscInt    nrows, ncols, nnz;
    double      *times;
    int         ntimes = 0, capacity;
    double      total_time = 0.0;
    PetscLogDouble t0, t1;

    PetscFunctionBeginUser;

    /* Extract filename */
    const char *fname = strrchr(filepath, '/');
    fname = fname ? fname + 1 : filepath;
    strncpy(result->name, fname, sizeof(result->name) - 1);
    char *dot = strrchr(result->name, '.');
    if (dot) *dot = '\0';

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Benchmarking: %s\n", result->name));

    /* Load matrix */
    PetscCall(LoadMatrixMarket(filepath, &A));
    PetscCall(MatGetSize(A, &nrows, &ncols));
    MatInfo info;
    PetscCall(MatGetInfo(A, MAT_LOCAL, &info));
    nnz = (PetscInt)info.nz_used;

    result->rows = nrows;
    result->cols = ncols;
    result->nnz  = nnz;

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %" PetscInt_FMT "x%" PetscInt_FMT ", nnz=%" PetscInt_FMT "\n", nrows, ncols, nnz));

    /* Create vectors */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, ncols, &x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nrows, &y));
    PetscCall(VecDuplicate(y, &y_init));

    /* Initialize: x = 1.0, y_init = i * 1e-9 */
    PetscCall(VecSet(x, 1.0));
    PetscScalar *yarr;
    PetscCall(VecGetArray(y_init, &yarr));
    for (PetscInt i = 0; i < nrows; i++) yarr[i] = (PetscScalar)i * 1e-9;
    PetscCall(VecRestoreArray(y_init, &yarr));

    /* Warm-up */
    for (int i = 0; i < warmup; i++) {
        PetscCall(VecCopy(y_init, y));
        PetscCall(MatMultAdd(A, x, y, y));  /* y = A*x + y */
    }

    /* Timed runs */
    capacity = min_iters * 2;
    PetscCall(PetscMalloc1(capacity, &times));

    while (ntimes < min_iters || total_time < min_time) {
        if (ntimes >= capacity) {
            capacity *= 2;
            PetscCall(PetscRealloc(capacity * sizeof(double), &times));
        }

        PetscCall(VecCopy(y_init, y));
        PetscCall(PetscTime(&t0));
        PetscCall(MatMultAdd(A, x, y, y));
        PetscCall(PetscTime(&t1));

        times[ntimes++] = t1 - t0;
        total_time += t1 - t0;
    }

    /* Statistics */
    ComputeStats(times, ntimes, &result->median_s, &result->mean_s, &result->std_s, &result->min_s, &result->max_s);
    result->gflops = (2.0 * nnz) / (result->median_s * 1e9);
    /* CSR: row_ptr + col_idx + values + x + y */
    double bytes = (double)((nrows + 1) * sizeof(PetscInt) + nnz * sizeof(PetscInt) + nnz * sizeof(PetscScalar)
                            + ncols * sizeof(PetscScalar) + nrows * sizeof(PetscScalar));
    result->bw_gbs = bytes / (result->median_s * 1e9);
    result->iters = ntimes;

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  median=%.3fms, %.2f GFLOP/s, %.1f GB/s\n",
                          result->median_s * 1e3, result->gflops, result->bw_gbs));

    /* Cleanup */
    PetscCall(PetscFree(times));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&y_init));
    PetscCall(MatDestroy(&A));

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteCSV(const char *filename, BenchResult *results, int n)
{
    FILE *f;

    PetscFunctionBeginUser;
    f = fopen(filename, "w");
    if (!f) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", filename);

    fprintf(f, "matrix,library,rows,cols,nnz,median_s,mean_s,std_s,min_s,max_s,gflops,bw_gbs,iters\n");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s,petsc,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.3f,%d\n",
                results[i].name, results[i].rows, results[i].cols, results[i].nnz,
                results[i].median_s, results[i].mean_s, results[i].std_s, results[i].min_s, results[i].max_s,
                results[i].gflops, results[i].bw_gbs, results[i].iters);
    }
    fclose(f);
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char **argv)
{
    char        matrix_dir[PETSC_MAX_PATH_LEN] = ".";
    char        output[PETSC_MAX_PATH_LEN] = "results_petsc.csv";
    PetscInt    warmup = 10, min_iters = 100;
    PetscReal   min_time = 1.0;
    DIR         *dir;
    struct dirent *entry;
    char        filepath[PETSC_MAX_PATH_LEN];
    BenchResult *results = NULL;
    int         nresults = 0, capacity = 16;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    /* Parse options */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-matrix_dir", matrix_dir, sizeof(matrix_dir), NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-output", output, sizeof(output), NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-warmup", &warmup, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_iters", &min_iters, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-min_time", &min_time, NULL));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "=== SpMV Benchmark: PETSc ===\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Operation: y = A*x + y (f64, sequential)\n\n"));

    /* Find .mtx files */
    dir = opendir(matrix_dir);
    if (!dir) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open directory %s", matrix_dir);

    PetscCall(PetscMalloc1(capacity, &results));

    while ((entry = readdir(dir)) != NULL) {
        const char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) continue;

        snprintf(filepath, sizeof(filepath), "%s/%s", matrix_dir, entry->d_name);

        if (nresults >= capacity) {
            capacity *= 2;
            PetscCall(PetscRealloc(capacity * sizeof(BenchResult), &results));
        }

        PetscErrorCode ierr = RunBenchmark(filepath, (int)warmup, (int)min_iters, min_time, &results[nresults]);
        if (ierr) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error loading %s, skipping\n", entry->d_name));
            continue;
        }
        nresults++;
    }
    closedir(dir);

    if (nresults == 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "No .mtx files found in %s\n", matrix_dir));
    } else {
        PetscCall(WriteCSV(output, results, nresults));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nResults written to %s\n", output));
    }

    PetscCall(PetscFree(results));
    PetscCall(PetscFinalize());
    return 0;
}
