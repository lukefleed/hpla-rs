# SpMV Benchmark - PETSc
#
# Usage:
#   make spmv_bench
#   ./spmv_bench -matrix_dir ../matrices -output results_petsc.csv

# PETSc configuration (adjust PETSC_DIR and PETSC_ARCH as needed)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compiler flags for maximum performance
CFLAGS_OPT = -O3 -march=icelake-server -mtune=icelake-server -ffast-math -funroll-loops

spmv_bench: spmv_bench.c
	${CLINKER} -o $@ $< ${PETSC_LIB} ${CFLAGS_OPT}

clean::
	rm -f spmv_bench

.PHONY: clean
