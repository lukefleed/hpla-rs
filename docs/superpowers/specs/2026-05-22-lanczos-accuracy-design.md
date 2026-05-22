# Lanczos Accuracy Export Design

## Purpose

The benchmark results need a reproducible accuracy check for the Lanczos
backends. The check should show that the implementations run with the same
matrix, start vector, Krylov dimension, and requested tolerance, and that their
computed vectors agree to numerical precision.

The output will be a committed CSV file plus an optional plot generated from
that CSV. This keeps accuracy evidence separate from Criterion timing data.

## Current State

`src/tests.rs` already compares Lanczos backend outputs against a faer reference
with relative L2 error below `1e-8`. The tests use the same scaling, start
vector, Krylov dimension selection, and `SAAD_TOL` used by the Criterion
harnesses.

The current tests are not enough as a report artifact. They cover a small
matrix subset, print results to stderr, and do not write a stable CSV that can
be committed or plotted.

## Proposed Shape

Add a small Rust binary:

```bash
cargo run --release --bin lanczos_accuracy -- --output python/data/lanczos_accuracy.csv
```

The binary will iterate over `LANCZOS_SUITE`, skip missing matrices with an
explicit status row, and write one CSV row per matrix, kernel, backend, and
format. It will not use Criterion and will not measure runtime.

`python/plot.py` will gain one simple command:

```bash
python3 python/plot.py accuracy
```

The command will read `python/data/lanczos_accuracy.csv` and generate accuracy
plots under `python/accuracy/`.

## Accuracy Metric

The primary metric is the relative L2 difference against the faer result
computed with the same kernel:

```text
||y_backend - y_faer||_2 / ||y_faer||_2
```

This validates cross-backend agreement. It does not claim to measure the true
absolute error against an exact dense `exp(-A)b`, which is impractical for the
large sparse matrices in the benchmark suite.

The CSV will also record the requested Saad tolerance and the final Saad
indicator used to choose the Krylov dimension, so the result states both the
requested stopping criterion and the observed backend agreement.

## CSV Schema

The output file will use these columns:

```text
kernel,matrix,backend,format,m,saad_tol,saad_estimate,rel_l2_vs_faer,norm_y,status
```

Column meanings:

- `kernel`: `lanczos_one_pass` or `lanczos_two_pass`.
- `matrix`: matrix stem from `LANCZOS_SUITE`.
- `backend`: `faer`, `eigen`, `petsc`, or `psblas`.
- `format`: `csr`, `csc`, or empty when the backend has no format variant.
- `m`: Krylov dimension selected for that matrix.
- `saad_tol`: requested tolerance, currently `SAAD_TOL`.
- `saad_estimate`: final a posteriori estimate from the faer Krylov probe.
- `rel_l2_vs_faer`: relative L2 difference against the faer result for the
  same kernel.
- `norm_y`: Euclidean norm of the backend result.
- `status`: `ok`, `missing_matrix`, `unsupported`, or `failed`.

Rows with `status != ok` keep the same schema and leave numerical fields empty
where no value exists.

## Backend Coverage

For one-pass Lanczos, the export will cover:

- faer CSC and CSR
- Eigen CSR and CSC
- PETSc CSR
- PSBLAS CSR and CSC when the setup routine returns a non-null context

For two-pass Lanczos, the export will cover:

- faer CSC and CSR
- Eigen CSR and CSC
- PETSc CSR
- PSBLAS CSR and CSC when the setup routine returns a non-null context

Null contexts are recorded as `unsupported`, matching the existing test
behavior.

## Plot

The Python plot will be intentionally simple: relative L2 error on a logarithmic
y-axis, grouped by backend and separated by kernel. It should show whether all
implemented backends remain below the expected agreement band.

The plot is a secondary artifact. The CSV is the source of truth.

## Error Handling

The binary will write rows for expected skips and return a non-zero exit code
for unexpected failures after flushing the rows already collected. This keeps
partial diagnostics available without hiding broken backend executions.

The output directory will be created if needed. Existing CSV output will be
overwritten.

## Verification

Implementation should keep the existing tests intact. Additional verification
should include:

- `cargo check --all-targets`
- running the new binary on the local matrix set
- `python3 python/plot.py accuracy`
- `python3 python/plot.py --help`

Full benchmark runs remain user-controlled.

## Out Of Scope

This change will not add a dense reference solver for exact `exp(-A)b`. It will
not change the benchmark timing harnesses, Criterion settings, throughput
formulas, or performance-profile logic.
