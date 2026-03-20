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


def load_data(criterion_path):
    data = []
    base_dir = Path(criterion_path)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return data

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
                    
                    if elements_processed > 0 and time_ns > 0:
                        # throughput = elements / seconds -> GFLOP/s
                        time_s = time_ns * 1e-9
                        elements_per_sec = elements_processed / time_s
                        gflops_per_sec = elements_per_sec / 1e9
                        
                        data.append({
                            'Matrix': matrix_name,
                            'Configuration': full_config,
                            'Throughput (GFLOP/s)': gflops_per_sec
                        })
    return pd.DataFrame(data)

def generate_plots(df, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    matrices = df['Matrix'].unique()
    
    # Sort configurations to have a consistent order
    config_order = ['faer/csc', 'eigen/csc_map', 'petsc/csr_inodes', 'petsc/csr_raw', 'psblas/csr', 'mkl/csr_ie']
    

    
    for matrix in matrices:
        matrix_df = df[df['Matrix'] == matrix].copy()
        
        # Sort based on custom order (if config is missing, it will still sort correctly among existing)
        matrix_df['Config_Cat'] = pd.Categorical(matrix_df['Configuration'], categories=config_order, ordered=True)
        matrix_df = matrix_df.sort_values('Config_Cat')
        
        plt.figure(figsize=(8.5, 4.5))
        
        # Pure matplotlib bar chart to avoid seaborn/scipy/numpy binary conflicts in Intel Python
        x_pos = np.arange(len(matrix_df))
        
        # Use default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        bar_colors = [colors[i % len(colors)] for i in range(len(matrix_df))]
        
        bars = plt.bar(
            x_pos, 
            matrix_df['Throughput (GFLOP/s)'],
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
        
        # Grid tweaks for modern look
        plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylim(0, df['Throughput (GFLOP/s)'].max() * 1.15) # Dynamic Y limit based on global max
        
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
    
    print("Parsing Criterion JSON results...")
    df = load_data(CRITERION_DIR)
    
    if df.empty:
        print("No data found! Ensure you have run 'cargo bench' and paths are correct.")
    else:
        print(f"Found {len(df)} configurations across {df['Matrix'].nunique()} matrices.")
        generate_plots(df, OUTPUT_DIR)
