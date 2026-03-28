import os
import sys

_old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.stderr.close()
sys.stderr = _old_stderr

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif', 'serif'],
    # 'text.usetex': True
})

# Backends whose storage format is CSC (plotted with hollow markers on roofline)
CSC_BACKENDS = {'faer/csc', 'eigen/csc_map', 'mkl/csc_ie', 'psblas/csc'}

CONFIG_ORDER = [
    'faer/csc', 'faer/csr',
    'eigen/csc_map', 'eigen/csr_map',
    'petsc/csr_inodes', 'petsc/csr_raw',
    'psblas/csr', 'psblas/csc',
    'mkl/csr_ie', 'mkl/csc_ie',
]

BACKEND_COLORS = {
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


def load_data(criterion_path):
    """Load Criterion benchmark results into a DataFrame.

    Parses every ``spmv_*`` group directory under *criterion_path* and
    extracts throughput data.  The returned DataFrame contains columns
    ``Matrix``, ``Configuration``, ``Throughput (GFLOP/s)`` and ``nnz``.
    The ``nnz`` column is derived as ``elements_processed / 2`` because
    the Criterion throughput element count equals ``2 * nnz`` (one
    multiply and one add per nonzero).
    """
    data = []
    base_dir = Path(criterion_path)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return pd.DataFrame(data)

    for group_dir in base_dir.glob("spmv_*"):
        matrix_name = group_dir.name.replace("spmv_", "")

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
                        # throughput = elements / seconds -> GFLOP/s
                        time_s = time_ns * 1e-9
                        elements_per_sec = elements_processed / time_s
                        gflops_per_sec = elements_per_sec / 1e9

                        # CI bounds: lower time -> higher GFLOP/s (inverted)
                        if ci_lower > 0 and ci_upper > 0:
                            gflops_upper = elements_processed / (ci_lower * 1e-9) / 1e9
                            gflops_lower = elements_processed / (ci_upper * 1e-9) / 1e9
                        else:
                            gflops_upper = gflops_per_sec
                            gflops_lower = gflops_per_sec

                        data.append({
                            'Matrix': matrix_name,
                            'Configuration': full_config,
                            'Throughput (GFLOP/s)': gflops_per_sec,
                            'GFLOP/s lower': gflops_lower,
                            'GFLOP/s upper': gflops_upper,
                            'nnz': int(elements_processed / 2),
                        })
    return pd.DataFrame(data)


def read_mtx_dimensions(matrices_dir):
    """Read nrows and ncols from every ``.mtx`` file in *matrices_dir*.

    Skips comment lines (starting with ``%``) and parses the first
    non-comment line which contains ``nrows ncols nnz_stored``.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping from matrix stem name to ``(nrows, ncols)``.
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
    After warmup, y may be cached — effective AI is higher than plotted.
    """
    bytes_moved = (nrows + 1) * 4 + nnz * 4 + nnz * 8 + ncols * 8 + nrows * 16
    return (2 * nnz) / bytes_moved


def plot_roofline(df, matrix_dims, hw_config, output_dir):
    """Generate a roofline model plot (log-log) for all backends and matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Matrix``, ``Configuration``,
        ``Throughput (GFLOP/s)`` and ``nnz``.
    matrix_dims : dict[str, tuple[int, int]]
        Mapping matrix name -> ``(nrows, ncols)``.
    hw_config : dict
        Must contain key ``stream_triad_GBs`` (bandwidth in GB/s).
    output_dir : str | Path
        Directory where ``roofline.png`` will be saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stream_bw = hw_config['stream_triad_GBs']

    # Collect unique backends and matrices present in data
    backends = [b for b in CONFIG_ORDER if b in df['Configuration'].values]
    matrices = sorted(df['Matrix'].unique())

    # Assign colors to backends, markers to matrices
    backend_colors = {b: BACKEND_COLORS.get(b, '#999999') for b in backends}

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

        is_csc = cfg in CSC_BACKENDS
        facecolor = 'none' if is_csc else backend_colors.get(cfg, 'gray')
        edgecolor = backend_colors.get(cfg, 'gray')

        ax.scatter(
            ai, gflops,
            marker=matrix_markers.get(mat, 'o'),
            s=90,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=1.5,
            zorder=5,
        )

    # Draw bandwidth ceiling spanning the AI range of the data points
    if ai_values:
        ai_min = min(ai_values) * 0.7
        ai_max = max(ai_values) * 1.4
        ai_line = np.linspace(ai_min, ai_max, 200)
        ceiling_line = stream_bw * ai_line
        ax.plot(ai_line, ceiling_line, 'r--', linewidth=2.0, zorder=4,
                label=f'STREAM Triad ceiling ({stream_bw:.1f} GB/s)')

    # Draw compute ceiling (Rpeak) if configured.
    # Rpeak = avx_freq_GHz * doubles_per_vec * flops_per_fma * fma_ports
    # e.g. Ice Lake 2xFMA512: 2.1 * 8 * 2 * 2 = 67.2 GFLOP/s
    # Use sustained AVX-512 freq (check turbostat), not nominal base.
    peak_gflops = hw_config.get('peak_gflops', None)
    if peak_gflops is not None:
        ax.axhline(y=peak_gflops, color='blue', linestyle=':', linewidth=1.5,
                   label=f'Peak compute ({peak_gflops:.1f} GFLOP/s)', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('SpMV Roofline \u2014 Single Core (STREAM Triad Ceiling)',
                 fontsize=14, pad=15)

    # Build two-part legend: backends (color) and matrices (shape)
    backend_handles = []
    for b in backends:
        is_csc = b in CSC_BACKENDS
        fc = 'none' if is_csc else backend_colors[b]
        h = plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=fc, markeredgecolor=backend_colors[b],
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


def generate_plots(df, output_dir, matrix_dims=None, hw_config=None):
    """Generate per-matrix bar charts of SpMV throughput.

    If *matrix_dims* and *hw_config* are both provided, a horizontal
    bandwidth-ceiling line is drawn on each chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Matrix``, ``Configuration``,
        ``Throughput (GFLOP/s)`` and ``nnz``.
    output_dir : str | Path
        Directory for per-matrix PNG files.
    matrix_dims : dict[str, tuple[int, int]] or None
        Mapping matrix name -> ``(nrows, ncols)``.
    hw_config : dict or None
        Must contain ``stream_triad_GBs`` if supplied.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matrices = df['Matrix'].unique()

    for matrix in matrices:
        matrix_df = df[df['Matrix'] == matrix].copy()

        # Sort based on custom order
        matrix_df['Config_Cat'] = pd.Categorical(
            matrix_df['Configuration'], categories=CONFIG_ORDER, ordered=True,
        )
        matrix_df = matrix_df.sort_values('Config_Cat')

        # Compute per-matrix bandwidth ceiling if possible
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

        bar_colors = [BACKEND_COLORS.get(cfg, '#999999') for cfg in matrix_df['Configuration']]

        # Confidence-interval error bars (inverted: lower time -> higher GFLOP/s)
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

        plt.title(f'SpMV Performance on {matrix}', pad=15, fontsize=14)
        plt.ylabel('Throughput (GFLOP/s)', labelpad=10, fontsize=12)

        # Annotate bars with exact values
        for bar in bars:
            height = bar.get_height()
            if not pd.isna(height) and height > 0:
                plt.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=10)

        # Draw bandwidth ceiling line if available
        if ceiling is not None:
            plt.axhline(y=ceiling, linestyle='--', color='red',
                        linewidth=1.5, label='BW ceiling', zorder=4)
            plt.legend(loc='upper right', fontsize=9)

        # Grid tweaks for modern look
        plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Dynamic Y limit based on per-matrix max and optional ceiling
        local_max = matrix_df['Throughput (GFLOP/s)'].max()
        ylim_top = max(local_max, ceiling) * 1.15 if ceiling is not None else local_max * 1.15
        plt.ylim(0, ylim_top)

        # Format the X-axis labels to look cleaner in print
        formatted_labels = [cfg.replace('/', '\n') for cfg in matrix_df['Configuration']]
        plt.xticks(x_pos, formatted_labels, rotation=0, ha='center', fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()

        save_path = out_path / f"{matrix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

        print(f"[{matrix}] Saved modern plot to -> {save_path}")


if __name__ == "__main__":
    CRITERION_DIR = "../target/criterion"
    OUTPUT_DIR = "gemv"
    MATRICES_DIR = "../matrices"
    HW_CONFIG_PATH = Path(__file__).parent / "hw_config.json"

    print("Parsing Criterion JSON results...")
    df = load_data(CRITERION_DIR)

    if df.empty:
        print("No data found! Ensure you have run 'cargo bench' and paths are correct.")
    else:
        print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

        # Load hardware config (optional, graceful degradation)
        hw_config = None
        if HW_CONFIG_PATH.exists():
            with open(HW_CONFIG_PATH, 'r') as f:
                hw_config = json.load(f)
            print(f"Loaded hardware config: STREAM Triad = {hw_config['stream_triad_GBs']:.1f} GB/s")
        else:
            print(f"Warning: {HW_CONFIG_PATH} not found. "
                  "Skipping roofline and bandwidth ceiling features.")

        # Read matrix dimensions from .mtx files
        matrix_dims = read_mtx_dimensions(MATRICES_DIR)
        if matrix_dims:
            print(f"Read dimensions for {len(matrix_dims)} matrices from {MATRICES_DIR}/")
        else:
            print(f"Warning: no .mtx files found in {MATRICES_DIR}/. "
                  "Roofline and ceiling features disabled.")
            matrix_dims = None

        # Generate per-matrix bar charts (ceiling line added when possible)
        generate_plots(df, OUTPUT_DIR,
                       matrix_dims=matrix_dims if hw_config else None,
                       hw_config=hw_config)

        # Generate roofline plot if both hw_config and matrix_dims are available
        if hw_config is not None and matrix_dims is not None:
            plot_roofline(df, matrix_dims, hw_config, OUTPUT_DIR)
