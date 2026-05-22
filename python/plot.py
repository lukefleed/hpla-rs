#!/usr/bin/env python3

import sys
import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
RAW_SAMPLES_CSV = "raw_samples.csv"
SUMMARY_CSV = "summary.csv"
ACCURACY_CSV = "lanczos_accuracy.csv"
np = None
pd = None
plt = None


def _script_command():
    return f"python3 {Path(sys.argv[0]).as_posix()}"


def _load_data_deps():
    global np, pd
    if np is not None:
        return
    try:
        import numpy as _np
        import pandas as _pd
    except ModuleNotFoundError as exc:
        sys.stderr.write(
            "error: missing Python dependency: "
            f"{exc.name}\n"
            "       install requirements with: python3 -m pip install -r python/requirements.txt\n"
        )
        sys.exit(1)

    np = _np
    pd = _pd


def _load_plot_deps():
    global plt
    _load_data_deps()
    if plt is not None:
        return
    try:
        import matplotlib.pyplot as _plt
    except ModuleNotFoundError as exc:
        sys.stderr.write(
            "error: missing Python dependency for plotting: "
            f"{exc.name}\n"
            "       install requirements with: python3 -m pip install -r python/requirements.txt\n"
        )
        sys.exit(1)

    plt = _plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif', 'serif'],
    })
LIBRARY_COLORS = {
    'faer': '#0072B2',
    'eigen': '#E69F00',
    'petsc': '#009E73',
    'psblas': '#D55E00',
    'mkl': '#CC79A7',
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
LANCZOS_TWO_PASS_CONFIG_ORDER = [
    'faer_csc/two_pass', 'faer_csr/two_pass',
    'eigen_csr/two_pass', 'eigen_csc/two_pass',
    'petsc_csr/two_pass',
    'psblas_csr/two_pass', 'psblas_csc/two_pass',
]

LANCZOS_TWO_PASS_BACKEND_COLORS = {
    b: LIBRARY_COLORS[_library_of_lanczos(b)]
    for b in LANCZOS_TWO_PASS_CONFIG_ORDER
}
LANCZOS_ONE_PASS_CONFIG_ORDER = [
    'faer_csc/one_pass', 'faer_csr/one_pass',
    'eigen_csr/one_pass', 'eigen_csc/one_pass',
    'petsc_csr/one_pass',
    'psblas_csr/one_pass', 'psblas_csc/one_pass',
]

LANCZOS_ONE_PASS_BACKEND_COLORS = {
    b: LIBRARY_COLORS[_library_of_lanczos(b)]
    for b in LANCZOS_ONE_PASS_CONFIG_ORDER
}

LANCZOS_MATRICES = {
    "kron_g500-logn18",
    "coPapersDBLP",
    "thermal2",
    "as-Skitter",
    "roadNet-CA",
    "delaunay_n22",
    "caidaRouterLevel",
    "citationCiteseer",
    "coAuthorsCiteseer",
    "coPapersCiteseer",
    "preferentialAttachment",
    "smallworld",
    "rgg_n_2_20_s0",
    "belgium_osm",
    "auto",
}


def _benchmark_from_group(group):
    if group.startswith("lanczos_two_pass_"):
        return "lanczos_two_pass", group[len("lanczos_two_pass_"):]
    if group.startswith("lanczos_"):
        return "lanczos", group[len("lanczos_"):]
    if group.startswith("spmv_"):
        return "spmv", group[len("spmv_"):]
    return None, None


def _time_unit_to_ns_factor(unit):
    if unit != "ns":
        raise ValueError(f"unsupported Criterion time unit: {unit}")
    return 1.0


def _throughput_to_gflops(elements, time_ns):
    return elements / (time_ns * 1e-9) / 1e9


def _slope_through_origin(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    denom = np.dot(xs, xs)
    if denom <= 0.0:
        return np.nan
    return float(np.dot(xs, ys) / denom)


def _matrix_stems(matrices_dir):
    path = Path(matrices_dir)
    if not path.exists():
        return None
    return {p.stem for p in path.glob("*.mtx")}


def _current_samples(samples, matrices_dir):
    if samples.empty:
        return samples

    spmv_matrices = _matrix_stems(matrices_dir)
    config_sets = {
        "spmv": set(SPMV_CONFIG_ORDER),
        "lanczos": set(LANCZOS_ONE_PASS_CONFIG_ORDER),
        "lanczos_two_pass": set(LANCZOS_TWO_PASS_CONFIG_ORDER),
    }

    keep = []
    for _, row in samples.iterrows():
        benchmark = row["Benchmark"]
        matrix = row["Matrix"]
        config = row["Configuration"]

        if config not in config_sets.get(benchmark, set()):
            keep.append(False)
            continue
        if benchmark == "spmv" and spmv_matrices is not None:
            keep.append(matrix in spmv_matrices)
            continue
        if benchmark in ("lanczos", "lanczos_two_pass"):
            keep.append(matrix in LANCZOS_MATRICES)
            continue
        keep.append(True)

    return samples.loc[keep].reset_index(drop=True)


def collect_raw_samples(criterion_dir):
    rows = []
    criterion_path = Path(criterion_dir)
    for raw_file in criterion_path.glob("*/*/*/new/raw.csv"):
        frame = pd.read_csv(raw_file)
        for _, row in frame.iterrows():
            benchmark, matrix = _benchmark_from_group(row["group"])
            if benchmark is None:
                continue
            function = str(row["function"])
            value = str(row["value"])
            configuration = f"{function}/{value}"
            elements = int(row["throughput_num"])
            sample_value = float(row["sample_measured_value"])
            unit = str(row["unit"])
            iteration_count = int(row["iteration_count"])
            sample_time_ns = sample_value * _time_unit_to_ns_factor(unit)
            time_per_iter_ns = sample_time_ns / iteration_count
            rows.append({
                "Criterion group": row["group"],
                "Criterion function": function,
                "Criterion value": value,
                "Benchmark": benchmark,
                "Matrix": matrix,
                "Configuration": configuration,
                "Elements": elements,
                "Throughput type": row["throughput_type"],
                "Sample measured value": sample_value,
                "Unit": unit,
                "Iteration count": iteration_count,
                "Sample time (ns)": sample_time_ns,
                "Time per iter (ns)": time_per_iter_ns,
                "Throughput (GFLOP/s)": _throughput_to_gflops(
                    elements, time_per_iter_ns,
                ),
                "Criterion path": str(raw_file.relative_to(criterion_path)),
            })
    columns = [
        "Criterion group", "Criterion function", "Criterion value",
        "Benchmark", "Matrix", "Configuration", "Elements",
        "Throughput type", "Sample measured value", "Unit",
        "Iteration count", "Sample time (ns)", "Time per iter (ns)",
        "Throughput (GFLOP/s)", "Criterion path",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_samples(samples):
    if samples.empty:
        return pd.DataFrame()

    data = []
    grouped = samples.groupby(
        ["Benchmark", "Matrix", "Configuration"],
        sort=True,
        as_index=False,
    )
    for (benchmark, matrix, config), group in grouped:
        elements_values = group["Elements"].unique()
        if len(elements_values) != 1:
            raise ValueError(
                f"inconsistent throughput elements for {benchmark}/{matrix}/{config}"
            )
        elements = int(elements_values[0])
        iters = group["Iteration count"].to_numpy(dtype=float)
        times = group["Sample time (ns)"].to_numpy(dtype=float)

        slope = _slope_through_origin(iters, times)
        throughput = _throughput_to_gflops(elements, slope)

        nnz = elements // 2 if benchmark == "spmv" else elements
        data.append({
            "Benchmark": benchmark,
            "Matrix": matrix,
            "Configuration": config,
            "Elements": elements,
            "Time estimate (ns)": slope,
            "Throughput (GFLOP/s)": throughput,
            "nnz": int(nnz),
            "Samples": int(len(group)),
        })
    return pd.DataFrame(data)


def relative_score_table(summary, config_order):
    if summary.empty:
        return pd.DataFrame()
    active = summary[summary["Configuration"].isin(config_order)].copy()
    if active.empty:
        return pd.DataFrame()
    active["Configuration"] = pd.Categorical(
        active["Configuration"], categories=config_order, ordered=True,
    )
    best = active.groupby("Matrix", observed=False)["Throughput (GFLOP/s)"].transform("max")
    active["Relative score"] = active["Throughput (GFLOP/s)"] / best
    return active.sort_values(["Matrix", "Configuration"]).reset_index(drop=True)


def export_csv_data(criterion_dir, data_dir, matrices_dir):
    _load_data_deps()
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _current_samples(collect_raw_samples(criterion_dir), matrices_dir)
    if samples.empty:
        sys.stderr.write(
            f"error: no Criterion raw.csv files found under {criterion_dir}.\n"
            "       Re-run the current benchmarks with Criterion's csv_output feature enabled.\n"
        )
        sys.exit(1)

    summary = summarize_samples(samples)
    samples.to_csv(out_dir / RAW_SAMPLES_CSV, index=False)
    summary.to_csv(out_dir / SUMMARY_CSV, index=False)

    print(f"[data] wrote {out_dir / RAW_SAMPLES_CSV}")
    print(f"[data] wrote {out_dir / SUMMARY_CSV}")


def load_summary_data(data_dir):
    _load_data_deps()
    path = Path(data_dir) / SUMMARY_CSV
    if not path.exists():
        sys.stdout.flush()
        sys.stderr.write(
            f"error: {path} not found.\n"
            f"       Use `{_script_command()} export-csv`\n"
            "       after running the Criterion benchmarks, or commit the generated CSVs.\n"
        )
        sys.exit(1)
    return pd.read_csv(path)


def read_mtx_dimensions(matrices_dir):
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
    bytes_moved = (nrows + 1) * 4 + nnz * 4 + nnz * 8 + ncols * 8 + nrows * 16
    return (2 * nnz) / bytes_moved


def plot_roofline(df, matrix_dims, hw_config, output_dir, config_order,
                  backend_colors, csc_backends):
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

        bars = plt.bar(
            x_pos,
            matrix_df['Throughput (GFLOP/s)'],
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


def benchmark_summary(df, benchmark):
    return df[df["Benchmark"] == benchmark].copy()


def run_spmv(data_dir, matrices_dir, hw_config_path):
    print("Reading CSV benchmark summary (SpMV)...")
    df = benchmark_summary(load_summary_data(data_dir), "spmv")

    if df.empty:
        print(f"No SpMV data found in {Path(data_dir) / SUMMARY_CSV}.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    hw_config = _load_hw_config(hw_config_path)
    matrix_dims = _load_matrix_dims(matrices_dir)

    output_dir = SCRIPT_DIR / "spmv"
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


def run_lanczos_two_pass(data_dir):
    print("Reading CSV benchmark summary (lanczos_two_pass)...")
    df = benchmark_summary(load_summary_data(data_dir), "lanczos_two_pass")

    if df.empty:
        print(f"No lanczos_two_pass data found in {Path(data_dir) / SUMMARY_CSV}.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    output_dir = SCRIPT_DIR / "lanczos_two_pass"
    generate_plots(
        df, output_dir,
        config_order=LANCZOS_TWO_PASS_CONFIG_ORDER,
        backend_colors=LANCZOS_TWO_PASS_BACKEND_COLORS,
        title_prefix="Two-Pass Lanczos Performance on",
    )

def parse_lanczos_config(config):
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


def plot_violin_scores(scores, configs, labels, backend_colors, title, save_path):
    if scores.empty:
        print(f"[violin] No data for {title}")
        return

    data = []
    active_configs = []
    active_labels = []
    for cfg, label in zip(configs, labels):
        values = scores.loc[
            scores["Configuration"] == cfg, "Relative score"
        ].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        data.append(values)
        active_configs.append(cfg)
        active_labels.append(label)

    if not data:
        print(f"[violin] No aligned data for {title}")
        return

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(1, len(data) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body, cfg in zip(parts["bodies"], active_configs):
        color = backend_colors.get(cfg, "#999999")
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.28)
        body.set_linewidth(1.0)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.4)

    rng = np.random.default_rng(42)
    for pos, cfg, values in zip(positions, active_configs, data):
        x = rng.normal(loc=pos, scale=0.045, size=len(values))
        ax.scatter(
            x,
            values,
            s=28,
            color=backend_colors.get(cfg, "#999999"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=4,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(positions)
    ax.set_xticklabels(active_labels, rotation=0, ha="center", fontsize=11)
    ax.set_ylabel("Relative throughput (best backend per matrix = 1)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"[violin] Saved -> {out_path}")


def run_violin(data_dir):
    summary = load_summary_data(data_dir)
    output_dir = SCRIPT_DIR / "violin"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_one = benchmark_summary(summary, "lanczos")
    df_two = benchmark_summary(summary, "lanczos_two_pass")
    df_spmv = benchmark_summary(summary, "spmv")

    lanczos_plots = [
        (df_one, "one_pass", "csr", LANCZOS_ONE_PASS_BACKEND_COLORS,
         "One-Pass Lanczos, CSR", "violin_one_pass_csr.png"),
        (df_one, "one_pass", "csc", LANCZOS_ONE_PASS_BACKEND_COLORS,
         "One-Pass Lanczos, CSC", "violin_one_pass_csc.png"),
        (df_two, "two_pass", "csr", LANCZOS_TWO_PASS_BACKEND_COLORS,
         "Two-Pass Lanczos, CSR", "violin_two_pass_csr.png"),
        (df_two, "two_pass", "csc", LANCZOS_TWO_PASS_BACKEND_COLORS,
         "Two-Pass Lanczos, CSC", "violin_two_pass_csc.png"),
    ]

    for df, kernel, fmt, colors, label, filename in lanczos_plots:
        if df.empty:
            print(f"[skip] violin {kernel}/{fmt}: no benchmark data")
            continue
        configs, libraries, matrices, _ = lanczos_slice(df, kernel, fmt)
        if not matrices:
            print(f"[skip] violin {kernel}/{fmt}: no aligned data")
            continue
        aligned = df[df["Matrix"].isin(matrices)]
        scores = relative_score_table(aligned, configs)
        plot_violin_scores(
            scores, configs, libraries, colors,
            f"Relative Throughput, {label}",
            output_dir / filename,
        )

    spmv_plots = [
        ("csr", "SpMV, CSR", "violin_spmv_csr.png"),
        ("csc", "SpMV, CSC", "violin_spmv_csc.png"),
    ]
    for fmt, label, filename in spmv_plots:
        if df_spmv.empty:
            print(f"[skip] violin spmv/{fmt}: no benchmark data")
            continue
        configs, labels, matrices, _ = spmv_slice(df_spmv, fmt)
        if not matrices:
            print(f"[skip] violin spmv/{fmt}: no aligned data")
            continue
        aligned = df_spmv[df_spmv["Matrix"].isin(matrices)]
        scores = relative_score_table(aligned, configs)
        plot_violin_scores(
            scores, configs, labels, SPMV_BACKEND_COLORS,
            f"Relative Throughput, {label}",
            output_dir / filename,
        )


def load_accuracy_data(data_dir):
    _load_data_deps()
    path = Path(data_dir) / ACCURACY_CSV
    if not path.exists():
        sys.stderr.write(
            f"error: {path} not found.\n"
            "       Generate it with:\n"
            "       cargo run --release --bin lanczos_accuracy -- "
            "--output python/data/lanczos_accuracy.csv\n"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    required = {
        "kernel", "matrix", "backend", "format", "m", "saad_tol",
        "saad_estimate", "rel_l2_vs_faer", "norm_y", "status",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        sys.stderr.write(f"error: {path} is missing columns: {missing_list}\n")
        sys.exit(1)

    df["rel_l2_vs_faer"] = pd.to_numeric(
        df["rel_l2_vs_faer"], errors="coerce"
    )
    df["Configuration"] = df.apply(_accuracy_config, axis=1)
    return df


def _accuracy_config(row):
    backend = "" if pd.isna(row["backend"]) else str(row["backend"])
    fmt = "" if pd.isna(row["format"]) else str(row["format"])
    if not fmt:
        return backend
    return f"{backend}/{fmt}"


def _accuracy_label(config):
    backend, fmt = config.split("/", 1)
    return f"{backend}\n{fmt.upper()}"


def plot_accuracy_kernel(df, kernel, title, save_path):
    sub = df[
        (df["kernel"] == kernel)
        & (df["status"].isin(["ok", "diverged"]))
        & df["rel_l2_vs_faer"].notna()
    ].copy()
    if sub.empty:
        print(f"[accuracy] No data for {title}")
        return

    order = ["faer/csr", "eigen/csr", "eigen/csc",
             "petsc/csr", "psblas/csr", "psblas/csc"]
    configs = [cfg for cfg in order if cfg in set(sub["Configuration"])]
    if not configs:
        print(f"[accuracy] No plottable configurations for {title}")
        return

    sub = sub[sub["Configuration"].isin(configs)].copy()
    positive = sub.loc[sub["rel_l2_vs_faer"] > 0.0, "rel_l2_vs_faer"]
    floor = min(float(positive.min()) * 0.1, 1e-16) if not positive.empty else 1e-16
    sub["Correct digits"] = -np.log10(sub["rel_l2_vs_faer"].clip(lower=floor))

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rng = np.random.default_rng(42)
    positions = np.arange(1, len(configs) + 1)

    for pos, cfg in zip(positions, configs):
        values = sub.loc[
            sub["Configuration"] == cfg, "Correct digits"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        backend = cfg.split("/", 1)[0]
        color = LIBRARY_COLORS.get(backend, "#999999")
        x = rng.normal(loc=pos, scale=0.045, size=len(values))
        ax.scatter(
            x,
            values,
            s=30,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=4,
        )
        median = float(np.median(values))
        ax.hlines(
            median,
            pos - 0.22,
            pos + 0.22,
            colors="black",
            linewidth=1.6,
            zorder=5,
        )

    min_digits = max(0.0, float(sub["Correct digits"].min()) - 0.5)
    max_digits = float(sub["Correct digits"].max()) + 0.5
    ax.set_ylim(min_digits, max_digits)
    ax.set_xticks(positions)
    ax.set_xticklabels([_accuracy_label(cfg) for cfg in configs], rotation=0, ha="center")
    ax.set_xlabel("Implementation / storage format", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$-\log_{10}$ relative L2 error vs faer/CSC", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"[accuracy] Saved -> {out_path}")


def run_accuracy(data_dir):
    df = load_accuracy_data(data_dir)
    output_dir = SCRIPT_DIR / "accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_kernel(
        df,
        "lanczos_one_pass",
        "Lanczos One-Pass: Output Agreement vs faer/CSC",
        output_dir / "accuracy_one_pass.png",
    )
    plot_accuracy_kernel(
        df,
        "lanczos_two_pass",
        "Lanczos Two-Pass: Output Agreement vs faer/CSC",
        output_dir / "accuracy_two_pass.png",
    )


def run_perfprof(data_dir):
    output_dir = SCRIPT_DIR / "perfprof"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CSV benchmark summary (perfprof)...")
    summary = load_summary_data(data_dir)
    df_one = benchmark_summary(summary, "lanczos")
    df_two = benchmark_summary(summary, "lanczos_two_pass")
    df_spmv = benchmark_summary(summary, "spmv")

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


def run_lanczos_one_pass(data_dir):
    print("Reading CSV benchmark summary (lanczos)...")
    df = benchmark_summary(load_summary_data(data_dir), "lanczos")

    if df.empty:
        print(f"No lanczos data found in {Path(data_dir) / SUMMARY_CSV}.")
        sys.exit(1)

    print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")

    output_dir = SCRIPT_DIR / "lanczos"
    generate_plots(
        df, output_dir,
        config_order=LANCZOS_ONE_PASS_CONFIG_ORDER,
        backend_colors=LANCZOS_ONE_PASS_BACKEND_COLORS,
        title_prefix="One-Pass Lanczos Performance on",
    )

def _load_hw_config(hw_config_path):
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
        description=(
            "Export Criterion raw CSV data and regenerate HPLA-RS benchmark plots. "
            "Plot commands read python/data/summary.csv. They do not read "
            "target/criterion directly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical workflow:\n"
            "  1. Run cargo bench for spmv, lanczos, and lanczos_two_pass.\n"
            f"  2. {_script_command()} export-csv\n"
            f"  3. {_script_command()} all\n\n"
            "Commands:\n"
            "  export-csv        collect target/criterion/**/new/raw.csv into python/data/*.csv\n"
            "  spmv              SpMV bar charts and roofline plot\n"
            "  lanczos_one_pass  one-pass Lanczos bar charts\n"
            "  lanczos_two_pass  two-pass Lanczos bar charts\n"
            "  perfprof          performance profiles for SpMV, one-pass, and two-pass\n"
            "  violin            normalized-throughput violin plots for all kernels\n"
            "  accuracy          Lanczos output-agreement plots\n"
            "  all               run spmv, lanczos_one_pass, lanczos_two_pass, perfprof, violin"
        ),
    )
    parser.add_argument(
        'bench', nargs='?', default='spmv',
        choices=[
            'export-csv',
            'spmv',
            'lanczos_one_pass',
            'lanczos_two_pass',
            'perfprof',
            'violin',
            'accuracy',
            'all',
        ],
        metavar='command',
        help=(
            "command to run: export-csv, spmv, lanczos_one_pass, "
            "lanczos_two_pass, perfprof, violin, accuracy, or all "
            "(default: spmv)"
        ),
    )
    parser.add_argument(
        '--criterion-dir',
        default=str(SCRIPT_DIR / ".." / "target" / "criterion"),
        help=(
            "Criterion output directory containing **/new/raw.csv files "
            "(used only by export-csv). Default: repo target/criterion"
        ),
    )
    parser.add_argument(
        '--data-dir',
        default=str(DEFAULT_DATA_DIR),
        help=(
            "directory containing raw_samples.csv and summary.csv, or where "
            "export-csv writes them. Default: python/data"
        ),
    )
    parser.add_argument(
        '--matrices-dir',
        default=str(SCRIPT_DIR / ".." / "matrices"),
        help=(
            "Matrix Market directory used only for SpMV roofline dimensions. "
            "Default: repo matrices"
        ),
    )
    parser.add_argument(
        '--hw-config',
        default=str(SCRIPT_DIR / "hw_config.json"),
        help=(
            "hardware config JSON generated by stream_bench.sh. Required for "
            "spmv/all roofline. Default: python/hw_config.json"
        ),
    )
    args = parser.parse_args()

    if args.bench == 'export-csv':
        export_csv_data(
            args.criterion_dir,
            args.data_dir,
            args.matrices_dir,
        )
    elif args.bench == 'spmv':
        _load_plot_deps()
        run_spmv(args.data_dir, args.matrices_dir, args.hw_config)
    elif args.bench == 'lanczos_one_pass':
        _load_plot_deps()
        run_lanczos_one_pass(args.data_dir)
    elif args.bench == 'lanczos_two_pass':
        _load_plot_deps()
        run_lanczos_two_pass(args.data_dir)
    elif args.bench == 'perfprof':
        _load_plot_deps()
        run_perfprof(args.data_dir)
    elif args.bench == 'violin':
        _load_plot_deps()
        run_violin(args.data_dir)
    elif args.bench == 'accuracy':
        _load_plot_deps()
        run_accuracy(args.data_dir)
    elif args.bench == 'all':
        _load_plot_deps()
        run_spmv(args.data_dir, args.matrices_dir, args.hw_config)
        run_lanczos_one_pass(args.data_dir)
        run_lanczos_two_pass(args.data_dir)
        run_perfprof(args.data_dir)
        run_violin(args.data_dir)
