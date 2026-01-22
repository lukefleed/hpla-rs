//! SpMV Benchmark: faer
//!
//! Operation: y ← A·x + y
//! Compile: RUSTFLAGS="-C target-cpu=icelake-server" cargo build --release

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use faer::col::Col;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Accum, Par};
use matrix_market_rs::MtxData;

#[derive(Parser)]
#[command(name = "spmv-bench-faer")]
struct Args {
    #[arg(short, long)]
    matrix_dir: PathBuf,

    #[arg(short, long, default_value = "results_faer.csv")]
    output: PathBuf,

    #[arg(short, long, default_value_t = 100)]
    warmup: usize,

    #[arg(short = 'n', long, default_value_t = 1000)]
    min_iterations: usize,

    #[arg(short = 't', long, default_value_t = 5.0)]
    min_time_secs: f64,
}

struct BenchResult {
    name: String,
    rows: usize,
    cols: usize,
    nnz: usize,
    median_s: f64,
    mean_s: f64,
    std_s: f64,
    min_s: f64,
    max_s: f64,
    gflops: f64,
    bw_gbs: f64,
    iters: usize,
}

fn load_mtx(path: &PathBuf) -> Result<SparseColMat<usize, f64>, String> {
    let data = MtxData::<f64>::from_file(path).map_err(|e| format!("{}", e))?;

    let MtxData::Sparse([nrows, ncols], coords, values, _) = data else {
        return Err("Only sparse matrices supported".into());
    };

    let triplets: Vec<_> = coords
        .iter()
        .zip(values.iter())
        .map(|([i, j], &v)| Triplet::new(*i, *j, v))
        .collect();

    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|e| format!("{:?}", e))
}

#[inline(never)]
fn spmv(a: &SparseColMat<usize, f64>, x: &Col<f64>, y: &mut Col<f64>) {
    sparse_dense_matmul(
        y.as_mat_mut(),
        Accum::Add,
        a.as_ref(),
        x.as_mat(),
        1.0,
        Par::Seq,
    );
}

fn stats(times: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    let median = if n.is_multiple_of(2) {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    } else {
        times[n / 2]
    };
    let mean = times.iter().sum::<f64>() / n as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    (median, mean, var.sqrt(), times[0], times[n - 1])
}

fn benchmark(
    path: &PathBuf,
    warmup: usize,
    min_iters: usize,
    min_time: f64,
) -> Result<BenchResult, String> {
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    eprintln!("Benchmarking: {}", name);

    let a = load_mtx(path)?;
    let (rows, cols) = (a.nrows(), a.ncols());
    let nnz = a.compute_nnz();
    eprintln!("  {}x{}, nnz={}", rows, cols, nnz);

    let x: Col<f64> = Col::from_fn(cols, |_| 1.0);
    let y_init: Col<f64> = Col::from_fn(rows, |i| (i as f64) * 1e-9);
    let mut y = y_init.clone();

    // Warm-up
    for _ in 0..warmup {
        y.copy_from(&y_init);
        spmv(&a, &x, &mut y);
    }

    // Timed runs
    let mut times = Vec::with_capacity(min_iters * 2);
    let mut total = std::time::Duration::ZERO;

    while times.len() < min_iters || total.as_secs_f64() < min_time {
        y.copy_from(&y_init);
        let t0 = Instant::now();
        spmv(&a, &x, &mut y);
        let dt = t0.elapsed();
        std::hint::black_box(&y);
        times.push(dt.as_secs_f64());
        total += dt;
    }

    let (median, mean, std, min, max) = stats(&mut times);
    let gflops = (2.0 * nnz as f64) / (median * 1e9);
    // CSC: col_ptr + row_idx + values + x + y
    let bytes = ((cols + 1) * 8 + nnz * 8 + nnz * 8 + cols * 8 + rows * 8) as f64;
    let bw = bytes / (median * 1e9);

    eprintln!(
        "  median={:.3}ms, {:.2} GFLOP/s, {:.1} GB/s",
        median * 1e3,
        gflops,
        bw
    );

    Ok(BenchResult {
        name,
        rows,
        cols,
        nnz,
        median_s: median,
        mean_s: mean,
        std_s: std,
        min_s: min,
        max_s: max,
        gflops,
        bw_gbs: bw,
        iters: times.len(),
    })
}

fn write_csv(path: &PathBuf, results: &[BenchResult]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "matrix,library,rows,cols,nnz,median_s,mean_s,std_s,min_s,max_s,gflops,bw_gbs,iters"
    )?;
    for r in results {
        writeln!(
            w,
            "{},faer,{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.3},{:.3},{}",
            r.name,
            r.rows,
            r.cols,
            r.nnz,
            r.median_s,
            r.mean_s,
            r.std_s,
            r.min_s,
            r.max_s,
            r.gflops,
            r.bw_gbs,
            r.iters
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("=== SpMV Benchmark: faer ===");
    eprintln!("Operation: y = A*x + y (f64, sequential)\n");

    let mut files: Vec<_> = std::fs::read_dir(&args.matrix_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mtx"))
        .collect();
    files.sort();

    if files.is_empty() {
        return Err(format!("No .mtx files in {}", args.matrix_dir.display()).into());
    }

    eprintln!("Found {} matrices\n", files.len());

    let mut results = Vec::new();
    for path in &files {
        match benchmark(path, args.warmup, args.min_iterations, args.min_time_secs) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("Error {}: {}", path.display(), e),
        }
    }

    write_csv(&args.output, &results)?;
    eprintln!("\nResults written to {}", args.output.display());

    Ok(())
}
