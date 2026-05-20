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
# Library palette (Wong colorblind-safe)
# ---------------------------------------------------------------------------

LIBRARY_COLORS = {
    'faer':   '#0072B2',  # blue
    'eigen':  '#E69F00',  # orange
    'petsc':  '#009E73',  # green
    'psblas': '#D55E00',  # vermilion
    'mkl':    '#CC79A7',  # reddish purple
}


def _library_of_spmv(backend):
    return backend.split('/', 1)[0]


def _library_of_lanczos(backend):
    head = backend.split('/', 1)[0]
    return head.split('_', 1)[0]


def _bar_label(config):
    if config.endswith(('/one_pass', '/two_pass')):
        return config.rsplit('/', 1)[0]
    return config.replace('/', '\n')


# ---------------------------------------------------------------------------
# SpMV configuration
# ---------------------------------------------------------------------------

# Backends whose storage format is CSC (plotted with hollow markers on roofline)
SPMV_CSC_BACKENDS = {'faer/csc', 'eigen/csc_map', 'mkl/csc_ie', 'psblas/csc'}

SPMV_CONFIG_ORDER = [
    'faer/csc', 'faer/csr',
    'eigen/csc_map', 'eigen/csr_map',
    'petsc/csr',
    'psblas/csr', 'psblas/csc',
    'mkl/csr_ie', 'mkl/csc_ie',
]

SPMV_BACKEND_COLORS = {b: LIBRARY_COLORS[_library_of_spmv(b)]
                       for b in SPMV_CONFIG_ORDER}

# ---------------------------------------------------------------------------
# Lanczos two-pass configuration
# ---------------------------------------------------------------------------

LANCZOS_TWO_PASS_CONFIG_ORDER = [
    'faer_csc/two_pass', 'faer_csr/two_pass',
    'faer/two_pass',
    'eigen_csr/two_pass', 'eigen_csc/two_pass',
    'eigen/two_pass',
    'petsc_csr/two_pass',
    'psblas_csr/two_pass', 'psblas_csc/two_pass',
]

LANCZOS_TWO_PASS_BACKEND_COLORS = {
    b: LIBRARY_COLORS[_library_of_lanczos(b)]
    for b in LANCZOS_TWO_PASS_CONFIG_ORDER
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
    'psblas_csr/one_pass', 'psblas_csc/one_pass',
]

LANCZOS_ONE_PASS_BACKEND_COLORS = {
    b: LIBRARY_COLORS[_library_of_lanczos(b)]
    for b in LANCZOS_ONE_PASS_CONFIG_ORDER
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

    present = set(df['Configuration'].unique())
    backends = [b for b in config_order if b in present]
    df = df[df['Configuration'].isin(backends)].copy()
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
    ax.set_title('SpMV Roofline, Single Core (STREAM Triad Ceiling)',
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

        present = set(matrix_df['Configuration'])
        ordered = [c for c in config_order if c in present]
        if not ordered:
            continue

        matrix_df = matrix_df[matrix_df['Configuration'].isin(ordered)].copy()
        full_order = ordered

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

        formatted_labels = [_bar_label(cfg) for cfg in matrix_df['Configuration']]
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
        matrix_dims=None,
        hw_config=None,
    )

    # TODO: Roofline for Lanczos two-pass. The dominant cost is 2k SpMVs,
    # but the vector-work overhead (9n + 7n per step) changes the
    # effective arithmetic intensity compared to bare SpMV.  Deriving
    # a proper compulsory-traffic model requires accounting for the
    # rolling three-vector access pattern and the tridiagonal solve.
    # Skipped until the traffic model is worked out.


def parse_lanczos_config(config):
    """Parse a Lanczos configuration string into (library, format, kernel).

    Returns ``None`` for aliases that lack an explicit format
    suffix (e.g. ``faer/two_pass``, ``eigen/one_pass``).
    """
    if '/' not in config:
        return None
    backend, kernel = config.split('/', 1)
    if '_' not in backend:
        return None
    parts = backend.split('_')
    if len(parts) >= 3 and parts[-2] in ('csr', 'csc'):
        library = '_'.join(parts[:-2])
        fmt = parts[-2]
        variant = parts[-1]
        return f'{library}_{variant}', fmt, kernel
    library, fmt = backend.rsplit('_', 1)
    if fmt not in ('csr', 'csc'):
        return None
    return library, fmt, kernel


def lanczos_slice(df, kernel, fmt):
    """Pivot ``Matrix x Library`` for one ``(kernel, fmt)`` slice.

    Returns ``(configs, libraries, matrices, throughput)``. Matrices
    missing data for any library in the slice are dropped, so all four
    return values are aligned and free of NaNs.
    """
    active_order = (
        LANCZOS_ONE_PASS_CONFIG_ORDER
        if kernel == 'one_pass'
        else LANCZOS_TWO_PASS_CONFIG_ORDER
    )
    pairs = []
    present = set(df['Configuration'].unique())

    for cfg in active_order:
        if cfg not in present:
            continue
        parsed = parse_lanczos_config(cfg)
        if parsed is None:
            continue
        library, cfg_fmt, cfg_kernel = parsed
        if cfg_fmt == fmt and cfg_kernel == kernel:
            pairs.append((cfg, library))

    if not pairs:
        return [], [], [], np.empty((0, 0))

    configs = [c for c, _ in pairs]
    libraries = [lib for _, lib in pairs]

    sub = df[df['Configuration'].isin(configs)]
    pivot = (sub
             .pivot(index='Matrix', columns='Configuration',
                    values='Throughput (GFLOP/s)')
             .reindex(columns=configs)
             .dropna())

    if pivot.empty:
        return configs, libraries, [], np.empty((0, len(libraries)))

    matrices = list(pivot.index)
    return configs, libraries, matrices, pivot.values


def plot_perfprof(configs, libraries, throughput, backend_colors, title,
                  save_path, thmax=None):
    """Render a Dolan-Moré performance-profile PNG.

    For each library, the curve is the empirical CDF of the
    performance ratio

        r[i, lib] = best_i / throughput[i, lib],

    where ``best_i = max_lib throughput[i, lib]`` is the fastest library
    on matrix ``i``.  The curve is

        rho_lib(theta) = #{i : r[i, lib] <= theta} / n_matrices,

    drawn as a steps-post line.
    """
    n_matrices, n_libraries = throughput.shape
    if n_matrices == 0 or n_libraries == 0:
        print(f"[perfprof] No data for {title}")
        return

    best = throughput.max(axis=1)
    ratios = best[:, None] / throughput

    if thmax is None:
        thmax = float(np.nanmax(ratios)) * 1.08
    thmax = max(thmax, 1.05)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for j, (cfg, lib) in enumerate(zip(configs, libraries)):
        col = ratios[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        unique_theta, counts = np.unique(col, return_counts=True)
        cum_prob = np.cumsum(counts) / col.size
        x = np.concatenate([[1.0], unique_theta, [thmax]])
        y = np.concatenate([[0.0], cum_prob, [cum_prob[-1]]])
        color = backend_colors.get(cfg, '#999999')
        ax.plot(x, y, drawstyle='steps-post', label=lib,
                color=color, linewidth=2.0, zorder=5)

    ax.set_xscale('log')
    ax.set_xlim(1.0, thmax)
    ax.set_ylim(-0.02, 1.04)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.set_xlabel(r'Performance ratio $\theta$ = best / throughput',
                  fontsize=12)
    ax.set_ylabel(r'Fraction of matrices $\rho(\theta)$', fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=10,
              title=f'Library (n={n_matrices})', title_fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"[perfprof] Saved -> {save_path}")


def parse_spmv_config(config):
    """Parse a SpMV configuration string into (library, format, variant).

    SpMV backend ids follow ``"<library>/<fmt>[_<variant>]"``, e.g.
    ``"mkl/csr_ie"`` -> ``("mkl", "csr", "ie")`` and
    ``"faer/csr"`` -> ``("faer", "csr", None)``.  Returns ``None`` for
    strings that do not match.
    """
    if '/' not in config:
        return None
    library, suffix = config.split('/', 1)
    parts = suffix.split('_', 1)
    fmt = parts[0]
    if fmt not in ('csr', 'csc'):
        return None
    variant = parts[1] if len(parts) > 1 else None
    return library, fmt, variant


def spmv_slice(df, fmt):
    """Pivot ``Matrix x Backend`` for one SpMV storage format.

    Returns ``(configs, labels, matrices, throughput)`` aligned across
    all backends in the slice; matrices missing data for any backend
    are dropped.  Labels collapse to the library name when a single
    variant exists, falling back to ``"<library> (<variant>)"`` when a
    library exposes more than one configured variant.
    """
    pairs = []
    present = set(df['Configuration'].unique())
    for cfg in SPMV_CONFIG_ORDER:
        if cfg not in present:
            continue
        parsed = parse_spmv_config(cfg)
        if parsed is None:
            continue
        library, cfg_fmt, variant = parsed
        if cfg_fmt != fmt:
            continue
        pairs.append((cfg, library, variant))

    if not pairs:
        return [], [], [], np.empty((0, 0))

    library_counts = {}
    for _, lib, _ in pairs:
        library_counts[lib] = library_counts.get(lib, 0) + 1

    configs = []
    labels = []
    for cfg, lib, variant in pairs:
        configs.append(cfg)
        if library_counts[lib] > 1 and variant is not None:
            labels.append(f"{lib} ({variant})")
        else:
            labels.append(lib)

    sub = df[df['Configuration'].isin(configs)]
    pivot = (sub
             .pivot(index='Matrix', columns='Configuration',
                    values='Throughput (GFLOP/s)')
             .reindex(columns=configs)
             .dropna())

    if pivot.empty:
        return configs, labels, [], np.empty((0, len(labels)))

    matrices = list(pivot.index)
    return configs, labels, matrices, pivot.values


def run_perfprof(criterion_dir):
    """Generate Lanczos and SpMV throughput-exceedance plots."""
    output_dir = Path(__file__).parent / "perfprof"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing Criterion JSON results (perfprof)...")
    df_one = load_data(criterion_dir, "lanczos_", derive_nnz=lambda e: e,
                       exclude_prefix="lanczos_two_pass_")
    df_two = load_data(criterion_dir, "lanczos_two_pass_", derive_nnz=lambda e: e)
    df_spmv = load_data(criterion_dir, "spmv_", derive_nnz=lambda e: e // 2)

    if df_one.empty and df_two.empty and df_spmv.empty:
        print("No benchmark data found. Run "
              "'cargo bench --bench spmv', "
              "'cargo bench --bench lanczos' and "
              "'cargo bench --bench lanczos_two_pass' first.")
        sys.exit(1)

    lanczos_plots = [
        (df_one, 'one_pass', 'csr', LANCZOS_ONE_PASS_BACKEND_COLORS,
         'One-Pass Lanczos, CSR'),
        (df_one, 'one_pass', 'csc', LANCZOS_ONE_PASS_BACKEND_COLORS,
         'One-Pass Lanczos, CSC'),
        (df_two, 'two_pass', 'csr', LANCZOS_TWO_PASS_BACKEND_COLORS,
         'Two-Pass Lanczos, CSR'),
        (df_two, 'two_pass', 'csc', LANCZOS_TWO_PASS_BACKEND_COLORS,
         'Two-Pass Lanczos, CSC'),
    ]

    for df, kernel, fmt, colors, label in lanczos_plots:
        if df.empty:
            print(f"[skip] {kernel}/{fmt}: no benchmark data")
            continue
        configs, libraries, matrices, throughput = lanczos_slice(df, kernel, fmt)
        if not matrices:
            print(f"[skip] {kernel}/{fmt}: no aligned data")
            continue

        pp_path = output_dir / f"perfprof_{kernel}_{fmt}.png"
        plot_perfprof(configs, libraries, throughput, colors,
                      f'Performance Profile, {label}', pp_path)

    spmv_plots = [
        ('csr', 'SpMV, CSR'),
        ('csc', 'SpMV, CSC'),
    ]

    for fmt, label in spmv_plots:
        if df_spmv.empty:
            print(f"[skip] spmv/{fmt}: no benchmark data")
            continue
        configs, labels, matrices, throughput = spmv_slice(df_spmv, fmt)
        if not matrices:
            print(f"[skip] spmv/{fmt}: no aligned data")
            continue

        pp_path = output_dir / f"perfprof_spmv_{fmt}.png"
        plot_perfprof(configs, labels, throughput, SPMV_BACKEND_COLORS,
                      f'Performance Profile, {label}', pp_path)


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
        choices=['spmv', 'lanczos', 'lanczos_two_pass', 'perfprof'],
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
    elif args.bench == 'perfprof':
        run_perfprof(criterion_dir)
