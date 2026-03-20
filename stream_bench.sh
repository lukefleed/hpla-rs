#!/usr/bin/env bash
# stream_bench.sh — Measure single-core STREAM Triad bandwidth and write
# the result together with the CPU model to python/hw_config.json.
#
# Usage:
#   bash stream_bench.sh
#   # or, after chmod +x:
#   ./stream_bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAM_DIR="${SCRIPT_DIR}/resources/stream"
STREAM_SRC="${STREAM_DIR}/stream.c"
STREAM_BIN="${STREAM_DIR}/stream"
STREAM_URL="https://www.cs.virginia.edu/stream/FTP/Code/stream.c"
OUTPUT_JSON="${SCRIPT_DIR}/python/hw_config.json"

# ---------------------------------------------------------------------------
# 1. Download STREAM 5.10 source (idempotent)
# ---------------------------------------------------------------------------
echo "==> Checking for STREAM source..."
mkdir -p "${STREAM_DIR}"

if [[ -f "${STREAM_SRC}" ]]; then
    echo "    stream.c already exists at ${STREAM_SRC}, skipping download."
else
    echo "    Downloading STREAM 5.10 from ${STREAM_URL}"
    curl -fSL -o "${STREAM_SRC}" "${STREAM_URL}"
    echo "    Download complete."
fi

# ---------------------------------------------------------------------------
# 2. Compile STREAM
#    - No -ffast-math: STREAM measures memory bandwidth, not compute.
#    - No -flto: single translation unit, no cross-module benefit.
# ---------------------------------------------------------------------------
echo "==> Compiling STREAM..."
# -fopenmp omitted: we run single-threaded (OMP_NUM_THREADS=1) and
# the system has libgomp (GCC) but not libomp (clang). STREAM compiles
# and runs correctly without OpenMP — it simply uses a single thread.
clang -O3 -march=native \
    -DSTREAM_ARRAY_SIZE=20000000 \
    -DSTREAM_TYPE=double \
    "${STREAM_SRC}" -o "${STREAM_BIN}"
echo "    Compiled: ${STREAM_BIN}"

# ---------------------------------------------------------------------------
# 3. Run STREAM (single-threaded, pinned to core 0)
# ---------------------------------------------------------------------------
echo "==> Running STREAM (OMP_NUM_THREADS=1, taskset -c 0)..."
STREAM_OUTPUT="$(OMP_NUM_THREADS=1 taskset -c 0 "${STREAM_BIN}")"
echo "${STREAM_OUTPUT}"

# ---------------------------------------------------------------------------
# 4. Parse the Triad bandwidth (MB/s) from STREAM output
# ---------------------------------------------------------------------------
echo "==> Parsing Triad result..."
TRIAD_MBS="$(echo "${STREAM_OUTPUT}" \
    | awk '/^Triad:/ { print $2 }')"

if [[ -z "${TRIAD_MBS}" ]]; then
    echo "ERROR: could not parse Triad bandwidth from STREAM output." >&2
    exit 1
fi

echo "    Triad bandwidth: ${TRIAD_MBS} MB/s"

# ---------------------------------------------------------------------------
# 5. Convert MB/s to GB/s
# ---------------------------------------------------------------------------
TRIAD_GBS="$(awk "BEGIN { printf \"%.4f\", ${TRIAD_MBS} / 1000.0 }")"
echo "    Triad bandwidth: ${TRIAD_GBS} GB/s"

# ---------------------------------------------------------------------------
# 6. Detect CPU model
# ---------------------------------------------------------------------------
echo "==> Detecting CPU model..."
CPU_MODEL="$(lscpu | awk -F: '/Model name/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2 }')"

if [[ -z "${CPU_MODEL}" ]]; then
    echo "WARNING: could not detect CPU model, using 'unknown'." >&2
    CPU_MODEL="unknown"
fi

echo "    CPU model: ${CPU_MODEL}"

# ---------------------------------------------------------------------------
# 7. Write python/hw_config.json
# ---------------------------------------------------------------------------
echo "==> Writing ${OUTPUT_JSON}"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

cat > "${OUTPUT_JSON}" <<EOF
{
  "cpu_model": "${CPU_MODEL}",
  "stream_triad_GBs": ${TRIAD_GBS}
}
EOF

echo "    Done. Contents:"
cat "${OUTPUT_JSON}"
