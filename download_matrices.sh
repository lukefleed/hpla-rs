#!/usr/bin/env bash
# download_matrices.sh — Download SuiteSparse matrices used by the
# benchmark suite. Idempotent: skips matrices already in matrices/.
#
# Usage:
#   bash download_matrices.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRICES_DIR="${SCRIPT_DIR}/matrices"
TEMP_DIR="${SCRIPT_DIR}/.matrix_download_tmp"
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

mkdir -p "${MATRICES_DIR}"
mkdir -p "${TEMP_DIR}"

# SuiteSparse group/name pairs.
#
# The first block is used by the SpMV bench: a mix of FEM, circuit,
# graph, economic, and CFD matrices chosen for sparsity-pattern variety.
# The second block is the dedicated Lanczos suite: symmetric matrices
# with zero or small mean diagonal so that exp(-A)v is numerically
# well-posed for the Saad a posteriori error estimator. See
# src/lib.rs::LANCZOS_SUITE for the consumed list.
MATRICES=(
    # SpMV suite
    "SNAP/amazon0302"
    "Bourchtein/atmosmodd"
    "Williams/cant"
    "GHS_psdef/inline_1"
    "Rajat/rajat31"
    "Schmid/thermal2"
    "SNAP/web-Google"
    "GHS_psdef/audikw_1"
    "Janna/Queen_4147"
    "DNVS/shipsec1"
    "Williams/pdb1HYS"
    "Williams/consph"
    "Williams/mac_econ_fwd500"
    "Freescale/circuit5M"
    "SNAP/roadNet-CA"
    # Lanczos suite (zero or small mean diagonal, suitable for exp(-A)v)
    "DIMACS10/kron_g500-logn18"
    "DIMACS10/coPapersDBLP"
    "SNAP/as-Skitter"
    "DIMACS10/delaunay_n22"
)

for entry in "${MATRICES[@]}"; do
    name="${entry##*/}"
    mtx_file="${MATRICES_DIR}/${name}.mtx"

    if [[ -f "${mtx_file}" ]]; then
        echo "[skip] ${name}.mtx already present"
        continue
    fi

    echo "[download] ${name} from ${BASE_URL}/${entry}.tar.gz"
    curl -fSL -o "${TEMP_DIR}/${name}.tar.gz" "${BASE_URL}/${entry}.tar.gz"

    echo "[extract] ${name}.tar.gz"
    tar -xzf "${TEMP_DIR}/${name}.tar.gz" -C "${TEMP_DIR}"

    mv "${TEMP_DIR}/${name}/${name}.mtx" "${mtx_file}"
    echo "[ok]   ${mtx_file}"
done

rm -rf "${TEMP_DIR}"

echo ""
echo "Matrix suite:"
ls -lhS "${MATRICES_DIR}"/*.mtx | awk '{print $5, $NF}'
