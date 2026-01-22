# SpMV Benchmark - PETSc
#
# Usage:
#   make spmv_bench
#   ./spmv_bench -matrix_dir ../matrices -output results_petsc.csv

# PETSc configuration (adjust PETSC_DIR and PETSC_ARCH as needed)
# Try to locate PETSc installed by Spack when PETSC_DIR is not set
PETSC_DIR ?= $(shell spack location -i petsc 2>/dev/null)
# Auto-detect PETSC_ARCH if not set
PETSC_ARCH ?= $(shell ls $(PETSC_DIR)/lib/petsc 2>/dev/null | grep -E 'arch' | head -n1)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compiler flags for maximum performance (use native CPU to match PETSc build)
CFLAGS ?= -O3 -march=native -mtune=native -ffast-math -funroll-loops

spmv_bench: spvm_bench.c
	${PETSC_CC} ${CFLAGS} ${PETSC_CC_INCLUDES} -o $@ $< ${PETSC_LIB}

clean::
	rm -f spmv_bench

.PHONY: clean
