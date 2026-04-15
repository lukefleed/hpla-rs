#!/usr/bin/env python3

import sys
import warnings
import argparse
import json
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    sys.stderr.write(
        "error: missing Python dependency for plotting: "
        f"{exc.name}\n"
        "       install requirements with: pip3 install -r python/requirements.txt\n"
    )
    sys.exit(1)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif', 'serif'],
})

# ---------------------------------------------------------------------------
# SpMV configuration
# ---------------------------------------------------------------------------

# Backends whose storage format is CSC (plotted with hollow markers on roofline)
SPMV_CSC_BACKENDS = {'faer/csc', 'eigen/csc_map', 'mkl/csc_ie', 'psblas/csc'}

SPMV_CONFIG_ORDER = [
    'faer/csc', 'faer/csr',
    'eigen/csc_map', 'eigen/csr_map',
    'petsc/csr_inodes', 'petsc/csr_raw',
    'psblas/csr', 'psblas/csc',
    'mkl/csr_ie', 'mkl/csc_ie',
]

SPMV_BACKEND_COLORS = {
    'faer/csc':          '#0072B2',  # blue
    'faer/csr':          '#4477AA',  # steel blue
    'eigen/csc_map':     '#E69F00',  # orange
    'eigen/csr_map':     '#CC79A7',  # pink
    'petsc/csr_inodes':  '#009E73',  # green
    'petsc/csr_raw':     '#56B4E9',  # sky blue
    'psblas/csr':        '#D55E00',  # vermilion
    'psblas/csc':        '#E6550D',  # dark vermilion
    'mkl/csr_ie':        '#F0E442',  # yellow
    'mkl/csc_ie':        '#000000',  # black
}

# ---------------------------------------------------------------------------
# Lanczos two-pass configuration
# ---------------------------------------------------------------------------

LANCZOS_TWO_PASS_CONFIG_ORDER = [
    'faer_csc/two_pass', 'faer_csr/two_pass',
    'faer/two_pass',
    'eigen_csr/two_pass', 'eigen_csc/two_pass',
    'eigen/two_pass',
    'petsc_csr/two_pass',
    'psblas/two_pass',
]

LANCZOS_TWO_PASS_BACKEND_COLORS = {
    'faer_csc/two_pass':   '#0072B2',  # blue
    'faer_csr/two_pass':   '#4477AA',  # steel blue
    'faer/two_pass':       '#0072B2',  # blue (legacy naming)
    'eigen_csr/two_pass':  '#E69F00',  # orange
    'eigen_csc/two_pass':  '#CC79A7',  # pink
    'eigen/two_pass':      '#E69F00',  # orange (legacy naming)
    'petsc_csr/two_pass':  '#009E73',  # green
    'psblas/two_pass':     '#D55E00',  # vermilion
}

# ---------------------------------------------------------------------------
# Lanczos one-pass configuration
# ---------------------------------------------------------------------------

LANCZOS_ONE_PASS_CONFIG_ORDER = [
    'faer_csc/one_pass', 'faer_csr/one_pass',
    'faer/one_pass',
    'eigen_csr/one_pass', 'eigen_csc/one_pass',
    'eigen/one_pass',
    'petsc_csr/one_pass',
    'psblas/one_pass',
]

LANCZOS_ONE_PASS_BACKEND_COLORS = {
    'faer_csc/one_pass':   '#0072B2',  # blue
    'faer_csr/one_pass':   '#4477AA',  # steel blue
    'faer/one_pass':       '#0072B2',  # blue (legacy naming)
    'eigen_csr/one_pass':  '#E69F00',  # orange
    'eigen_csc/one_pass':  '#CC79A7',  # pink
    'eigen/one_pass':      '#E69F00',  # orange (legacy naming)
    'petsc_csr/one_pass':  '#009E73',  # green
    'psblas/one_pass':     '#D55E00',  # vermilion
}


def load_data(criterion_path, group_prefix, derive_nnz, exclude_prefix=None):
    """Load Criterion benchmark results into a DataFrame.

    Scans every group directory matching `{group_prefix}*` under
    *criterion_path* and extracts throughput data.

    Parameters
    ----------
    criterion_path : str or Path
        Root Criterion output directory (`target/criterion`).
    group_prefix : str
        Directory prefix to match, e.g. `"spmv_"` or
        `"lanczos_two_pass_"`.
    derive_nnz : callable or None
        Function `(elements_processed) -> nnz`.  For SpMV this is
        `lambda e: e // 2` (element count = 2*nnz).  For Lanczos the
        bench already encoded the full FLOP count, so `nnz` is not
        meaningful and should be set to the raw element count.

    Returns
    -------
    pd.DataFrame
        Columns: Matrix, Configuration, Throughput (GFLOP/s),
        GFLOP/s lower, GFLOP/s upper, nnz.
    """
    data = []
    base_dir = Path(criterion_path)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return pd.DataFrame(data)

    for group_dir in base_dir.glob(f"{group_prefix}*"):
        if exclude_prefix and group_dir.name.startswith(exclude_prefix):
            continue
        matrix_name = group_dir.name[len(group_prefix):]

        for backend_dir in group_dir.iterdir():
            if not backend_dir.is_dir() or backend_dir.name == "report":
                continue
            backend = backend_dir.name

            for config_dir in backend_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                config = config_dir.name
                full_config = f"{backend}/{config}"

                bench_file = config_dir / "new" / "benchmark.json"
                est_file = config_dir / "new" / "estimates.json"

                if bench_file.exists() and est_file.exists():
                    with open(bench_file, 'r') as f:
                        bench_data = json.load(f)
                    with open(est_file, 'r') as f:
                        est_data = json.load(f)

                    elements_processed = bench_data.get('throughput', {}).get('Elements', 0)
                    time_ns = est_data.get('mean', {}).get('point_estimate', 0)
                    ci_lower = est_data.get('mean', {}).get('confidence_interval', {}).get('lower_bound', 0)
                    ci_upper = est_data.get('mean', {}).get('confidence_interval', {}).get('upper_bound', 0)

                    if elements_processed > 0 and time_ns > 0:
                        time_s = time_ns * 1e-9
                        elements_per_sec = elements_processed / time_s
                        gflops_per_sec = elements_per_sec / 1e9

                        if ci_lower > 0 and ci_upper > 0:
                            gflops_upper = elements_processed / (ci_lower * 1e-9) / 1e9
                            gflops_lower = elements_processed / (ci_upper * 1e-9) / 1e9
                        else:
                            gflops_upper = gflops_per_sec
                            gflops_lower = gflops_per_sec

                        nnz = derive_nnz(elements_processed) if derive_nnz else elements_processed

                        data.append({
                            'Matrix': matrix_name,
                            'Configuration': full_config,
                            'Throughput (GFLOP/s)': gflops_per_sec,
                            'GFLOP/s lower': gflops_lower,
                            'GFLOP/s upper': gflops_upper,
                            'nnz': int(nnz),
                        })
    return pd.DataFrame(data)


def read_mtx_dimensions(matrices_dir):
    """Read nrows and ncols from every `.mtx` file in *matrices_dir*.

    Skips comment lines (starting with `%`) and parses the first
    non-comment line which contains `nrows ncols nnz_stored`.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping from matrix stem name to `(nrows, ncols)`.
    """
    dims = {}
    mat_path = Path(matrices_dir)
    if not mat_path.exists():
        print(f"Warning: matrices directory {mat_path} does not exist.")
        return dims

    for mtx_file in mat_path.glob("*.mtx"):
        with open(mtx_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('%'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    nrows, ncols = int(parts[0]), int(parts[1])
                    dims[mtx_file.stem] = (nrows, ncols)
                break
    return dims


def compute_arithmetic_intensity(nrows, ncols, nnz):
    """Cold-cache compulsory-traffic AI for CSR SpMV (FLOP/byte).

    bytes = (nrows+1)*4 + nnz*4 + nnz*8 + ncols*8 + nrows*16
    FLOP = 2 * nnz

    For square matrices, CSR and CSC give the same result.
    After warmup, y may be cached; effective AI is higher than plotted.
    """
    bytes_moved = (nrows + 1) * 4 + nnz * 4 + nnz * 8 + ncols * 8 + nrows * 16
    return (2 * nnz) / bytes_moved


def plot_roofline(df, matrix_dims, hw_config, output_dir, config_order,
                  backend_colors, csc_backends):
    """Generate a roofline model plot (log-log) for all backends and matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns `Matrix`, `Configuration`,
        `Throughput (GFLOP/s)` and `nnz`.
    matrix_dims : dict[str, tuple[int, int]]
        Mapping matrix name -> `(nrows, ncols)`.
    hw_config : dict
        Must contain key `stream_triad_GBs` (bandwidth in GB/s).
    output_dir : str | Path
        Directory where `roofline.png` will be saved.
    config_order : list[str]
        Backend ordering for the legend.
    backend_colors : dict[str, str]
        Backend -> hex color.
    csc_backends : set[str]
        Backends plotted with hollow markers.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stream_bw = hw_config['stream_triad_GBs']

    backends = [b for b in config_order if b in df['Configuration'].values]
    matrices = sorted(df['Matrix'].unique())

    bc = {b: backend_colors.get(b, '#999999') for b in backends}

    marker_shapes = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', '<', '>']
    matrix_markers = {m: marker_shapes[i % len(marker_shapes)] for i, m in enumerate(matrices)}

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ai_values = []

    for _, row in df.iterrows():
        mat = row['Matrix']
        cfg = row['Configuration']
        gflops = row['Throughput (GFLOP/s)']
        nnz = int(row['nnz'])

        if mat not in matrix_dims:
            continue

        nrows, ncols = matrix_dims[mat]
        ai = compute_arithmetic_intensity(nrows, ncols, nnz)
        ai_values.append(ai)

        is_csc = cfg in csc_backends
        facecolor = 'none' if is_csc else bc.get(cfg, 'gray')
        edgecolor = bc.get(cfg, 'gray')

        ax.scatter(
            ai, gflops,
            marker=matrix_markers.get(mat, 'o'),
            s=90,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=1.5,
            zorder=5,
        )

    if ai_values:
        ai_min = min(ai_values) * 0.7
        ai_max = max(ai_values) * 1.4
        ai_line = np.linspace(ai_min, ai_max, 200)
        ceiling_line = stream_bw * ai_line
        ax.plot(ai_line, ceiling_line, 'r--', linewidth=2.0, zorder=4,
                label=f'STREAM Triad ceiling ({stream_bw:.1f} GB/s)')

    peak_gflops = hw_config.get('peak_gflops', None)
    if peak_gflops is not None:
        ax.axhline(y=peak_gflops, color='blue', linestyle=':', linewidth=1.5,
                   label=f'Peak compute ({peak_gflops:.1f} GFLOP/s)', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('SpMV Roofline -- Single Core (STREAM Triad Ceiling)',
                 fontsize=14, pad=15)

    backend_handles = []
    for b in backends:
        is_csc = b in csc_backends
        fc = 'none' if is_csc else bc[b]
        h = plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=fc, markeredgecolor=bc[b],
            markeredgewidth=1.5, markersize=8, label=b,
        )
        backend_handles.append(h)

    matrix_handles = []
    for m in matrices:
        h = plt.Line2D(
            [0], [0], marker=matrix_markers[m], color='w',
            markerfacecolor='gray', markeredgecolor='gray',
            markeredgewidth=1.0, markersize=8, label=m,
        )
        matrix_handles.append(h)

    first_legend = ax.legend(
        handles=backend_handles, title='Backend',
        loc='upper left', fontsize=9, title_fontsize=10,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=matrix_handles, title='Matrix',
        loc='lower right', fontsize=9, title_fontsize=10,
    )

    ax.grid(True, which='both', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = out_path / "roofline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"[roofline] Saved roofline plot to -> {save_path}")


def generate_plots(df, output_dir, config_order, backend_colors,
                   title_prefix, matrix_dims=None, hw_config=None):
    """Generate per-matrix bar charts of throughput.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `Matrix`, `Configuration`,
        `Throughput (GFLOP/s)` and `nnz`.
    output_dir : str | Path
        Directory for per-matrix PNG files.
    config_order : list[str]
        Backend ordering.
    backend_colors : dict[str, str]
        Backend -> hex color.
    title_prefix : str
        Plot title prefix, e.g. `"SpMV Performance on"` or
        `"Two-Pass Lanczos Performance on"`.
    matrix_dims : dict[str, tuple[int, int]] or None
        Mapping matrix name -> `(nrows, ncols)`.
    hw_config : dict or None
        Must contain `stream_triad_GBs` if supplied.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matrices = df['Matrix'].unique()

    for matrix in matrices:
        matrix_df = df[df['Matrix'] == matrix].copy()

        # Only include configs that appear in the ordering; append any
        # unexpected ones at the end so nothing is silently dropped.
        present = set(matrix_df['Configuration'])
        ordered = [c for c in config_order if c in present]
        extra = sorted(present - set(ordered))
        full_order = ordered + extra

        matrix_df['Config_Cat'] = pd.Categorical(
            matrix_df['Configuration'], categories=full_order, ordered=True,
        )
        matrix_df = matrix_df.sort_values('Config_Cat')

        # Compute per-matrix bandwidth ceiling (SpMV only)
        ceiling = None
        if matrix_dims is not None and hw_config is not None and matrix in matrix_dims:
            nrows, ncols = matrix_dims[matrix]
            nnz_series = matrix_df['nnz']
            if not nnz_series.empty:
                nnz = int(nnz_series.iloc[0])
                ai = compute_arithmetic_intensity(nrows, ncols, nnz)
                ceiling = hw_config['stream_triad_GBs'] * ai

        plt.figure(figsize=(8.5, 4.5))

        x_pos = np.arange(len(matrix_df))

        bar_colors = [backend_colors.get(cfg, '#999999') for cfg in matrix_df['Configuration']]

        yerr_lower = matrix_df['Throughput (GFLOP/s)'] - matrix_df['GFLOP/s lower']
        yerr_upper = matrix_df['GFLOP/s upper'] - matrix_df['Throughput (GFLOP/s)']
        yerr = [yerr_lower.values, yerr_upper.values]

        bars = plt.bar(
            x_pos,
            matrix_df['Throughput (GFLOP/s)'],
            yerr=yerr,
            capsize=3,
            ecolor='gray',
            color=bar_colors,
            edgecolor='black',
            linewidth=1.2,
            zorder=3
        )

        plt.title(f'{title_prefix} {matrix}', pad=15, fontsize=14)
        plt.ylabel('Throughput (GFLOP/s)', labelpad=10, fontsize=12)

        for bar in bars:
            height = bar.get_height()
            if not pd.isna(height) and height > 0:
                plt.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=10)

        if ceiling is not None:
            plt.axhline(y=ceiling, linestyle='--', color='red',
                        linewidth=1.5, label='BW ceiling', zorder=4)
            plt.legend(loc='upper right', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        local_max = matrix_df['Throughput (GFLOP/s)'].max()
        ylim_top = max(local_max, ceiling) * 1.15 if ceiling is not None else local_max * 1.15
        plt.ylim(0, ylim_top)

        formatted_labels = [cfg.replace('/', '\n') for cfg in matrix_df['Configuration']]
        plt.xticks(x_pos, formatted_labels, rotation=0, ha='center', fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()

        save_path = out_path / f"{matrix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

        print(f"[{matrix}] Saved plot to -> {save_path}")


def run_spmv(criterion_dir, matrices_dir, hw_config_path):
    """SpMV plotting pipeline."""
    print("Parsing Criterion JSON results (SpMV)...")
    df = load_data(criterion_dir, "spmv_", derive_nnz=lambda e: e // 2)

    if df.empty:
        print("No SpMV data found. Run 'cargo bench --bench spmv' first.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    hw_config = _load_hw_config(hw_config_path)
    matrix_dims = _load_matrix_dims(matrices_dir)

    output_dir = Path(__file__).parent / "spmv"
    generate_plots(
        df, output_dir,
        config_order=SPMV_CONFIG_ORDER,
        backend_colors=SPMV_BACKEND_COLORS,
        title_prefix="SpMV Performance on",
        matrix_dims=matrix_dims,
        hw_config=hw_config,
    )
    if hw_config is not None and matrix_dims:
        plot_roofline(
            df, matrix_dims, hw_config, output_dir,
            config_order=SPMV_CONFIG_ORDER,
            backend_colors=SPMV_BACKEND_COLORS,
            csc_backends=SPMV_CSC_BACKENDS,
        )


def run_lanczos_two_pass(criterion_dir, matrices_dir, hw_config_path):
    """Lanczos two-pass plotting pipeline."""
    print("Parsing Criterion JSON results (lanczos_two_pass)...")
    # The bench harness already encodes the full FLOP count in the
    # throughput element count (4k*(nnz + 4n)), so no division needed.
    df = load_data(criterion_dir, "lanczos_two_pass_", derive_nnz=lambda e: e)

    if df.empty:
        print("No lanczos_two_pass data found. Run 'cargo bench --bench lanczos_two_pass' first.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    output_dir = Path(__file__).parent / "lanczos_two_pass"
    generate_plots(
        df, output_dir,
        config_order=LANCZOS_TWO_PASS_CONFIG_ORDER,
        backend_colors=LANCZOS_TWO_PASS_BACKEND_COLORS,
        title_prefix="Two-Pass Lanczos Performance on",
        # No bandwidth ceiling for Lanczos bar charts; the arithmetic
        # intensity model differs from SpMV and is not yet implemented.
        matrix_dims=None,
        hw_config=None,
    )

    # TODO: Roofline for Lanczos two-pass. The dominant cost is 2k SpMVs,
    # but the vector-work overhead (9n + 7n per step) changes the
    # effective arithmetic intensity compared to bare SpMV.  Deriving
    # a proper compulsory-traffic model requires accounting for the
    # rolling three-vector access pattern and the tridiagonal solve.
    # Skipped until the traffic model is worked out.


def run_lanczos_one_pass(criterion_dir, matrices_dir, hw_config_path):
    """Lanczos one-pass plotting pipeline."""
    print("Parsing Criterion JSON results (lanczos)...")
    # The bench harness already encodes the full FLOP count in the
    # throughput element count (m*(2*nnz + 11n)), so no division needed.
    df = load_data(criterion_dir, "lanczos_", derive_nnz=lambda e: e,
                   exclude_prefix="lanczos_two_pass_")

    if df.empty:
        print("No lanczos data found. Run 'cargo bench --bench lanczos' first.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    output_dir = Path(__file__).parent / "lanczos"
    generate_plots(
        df, output_dir,
        config_order=LANCZOS_ONE_PASS_CONFIG_ORDER,
        backend_colors=LANCZOS_ONE_PASS_BACKEND_COLORS,
        title_prefix="One-Pass Lanczos Performance on",
        # No bandwidth ceiling for Lanczos bar charts; the arithmetic
        # intensity model differs from SpMV and is not yet implemented.
        matrix_dims=None,
        hw_config=None,
    )

    # TODO: Roofline for Lanczos one-pass. The dominant cost is m SpMVs
    # plus a final n*m gemv, but the full-basis storage (O(n*m)) changes
    # the effective cache behavior compared to the rolling-vector
    # two-pass variant. Skipped until the traffic model is worked out.


def _load_hw_config(hw_config_path):
    """Load hw_config.json if it exists, or exit with instructions."""
    path = Path(hw_config_path)
    if not path.exists():
        sys.stderr.write(
            f"error: {path} not found.\n"
            f"       Run `bash stream_bench.sh` from the repo root to\n"
            f"       measure single-core STREAM Triad bandwidth and\n"
            f"       populate hw_config.json.\n"
        )
        sys.exit(1)

    with open(path, 'r') as f:
        hw_config = json.load(f)
    print(f"Loaded hardware config: STREAM Triad = {hw_config['stream_triad_GBs']:.1f} GB/s")
    return hw_config


def _load_matrix_dims(matrices_dir):
    """Load matrix dimensions or exit with instructions."""
    matrix_dims = read_mtx_dimensions(matrices_dir)
    if matrix_dims:
        print(f"Read dimensions for {len(matrix_dims)} matrices from {matrices_dir}/")
    else:
        sys.stderr.write(
            f"error: no .mtx files found in {matrices_dir}/.\n"
            f"       Run `bash download_matrices.sh` from the repo root\n"
            f"       to populate the matrix suite.\n"
        )
        sys.exit(1)
    return matrix_dims


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Criterion benchmark results for HPLA-RS."
    )
    parser.add_argument(
        'bench', nargs='?', default='spmv',
        choices=['spmv', 'lanczos', 'lanczos_two_pass'],
        help="Benchmark type to plot (default: spmv).",
    )
    args = parser.parse_args()

    criterion_dir = str(Path(__file__).parent / ".." / "target" / "criterion")
    matrices_dir = str(Path(__file__).parent / ".." / "matrices")
    hw_config_path = Path(__file__).parent / "hw_config.json"

    if args.bench == 'spmv':
        run_spmv(criterion_dir, matrices_dir, hw_config_path)
    elif args.bench == 'lanczos':
        run_lanczos_one_pass(criterion_dir, matrices_dir, hw_config_path)
    elif args.bench == 'lanczos_two_pass':
        run_lanczos_two_pass(criterion_dir, matrices_dir, hw_config_path)
