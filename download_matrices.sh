#!/usr/bin/env bash
# download_matrices.sh — Download additional SuiteSparse matrices for
# SpMV benchmarking. Idempotent: skips matrices already in matrices/.
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

# Complete matrix suite: Group/Name pairs from SuiteSparse Collection.
# The script is idempotent — matrices already in matrices/ are skipped.
MATRICES=(
    "SNAP/amazon0302"            # 262K rows, 1.2M nnz — web graph, pattern, general
    "Bourchtein/atmosmodd"       # 1.3M rows, 8.8M nnz — atmospheric CFD, general
    "Williams/cant"              # 62K rows, 4M nnz — cantilever FEM, symmetric
    "GHS_psdef/inline_1"        # 504K rows, 36.8M nnz — structural FEM, symmetric
    "Rajat/rajat31"              # 4.7M rows, 20M nnz — circuit simulation, general
    "Schmid/thermal2"            # 1.2M rows, 8.6M nnz — thermal FEM, symmetric
    "SNAP/web-Google"            # 916K rows, 5.1M nnz — web graph, pattern, general
    "GHS_psdef/audikw_1"        # 943K rows, 77M nnz — large FEM, symmetric
    "Janna/Queen_4147"           # 4.1M rows, 329M nnz — very large, symmetric
    "DNVS/shipsec1"              # 140K rows, 7.8M nnz — structural, symmetric
    "Williams/pdb1HYS"           # 36K rows, 4.3M nnz — protein, symmetric
    "Williams/consph"            # 83K rows, 6M nnz — FEM sphere, symmetric
    "Williams/mac_econ_fwd500"   # 206K rows, 1.2M nnz — economic, general
    "Freescale/circuit5M"        # 5.6M rows, 59M nnz — very large circuit, general
    "SNAP/roadNet-CA"            # 1.9M rows, 5.5M nnz — road network, symmetric
)

for entry in "${MATRICES[@]}"; do
    name="${entry##*/}"  # Extract name after /
    mtx_file="${MATRICES_DIR}/${name}.mtx"

    if [[ -f "${mtx_file}" ]]; then
        echo "[skip] ${name}.mtx already exists"
        continue
    fi

    echo "[download] ${name} from ${BASE_URL}/${entry}.tar.gz"
    curl -fSL -o "${TEMP_DIR}/${name}.tar.gz" "${BASE_URL}/${entry}.tar.gz"

    echo "[extract] ${name}.tar.gz"
    tar -xzf "${TEMP_DIR}/${name}.tar.gz" -C "${TEMP_DIR}"

    # Move the .mtx file to matrices/
    mv "${TEMP_DIR}/${name}/${name}.mtx" "${mtx_file}"

    echo "[done] ${mtx_file}"
done

# Cleanup
rm -rf "${TEMP_DIR}"

echo ""
echo "Matrix suite now contains:"
ls -lhS "${MATRICES_DIR}"/*.mtx | awk '{print $5, $NF}'
