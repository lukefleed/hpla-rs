//! Custom build script for compiling the FFI C wrapper.
//!
//! Automatically locates the PETSc and Eigen installations strictly via Spack
//! and passes aggressive optimization flags to `clang`/`clang++` before
//! linking them statically to our Rust crate.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let petsc_output = Command::new("spack")
        .args(["location", "-i", "petsc"])
        .output()
        .expect("Failed to execute spack command for PETSc. Is spack sourced & in your PATH?");

    if !petsc_output.status.success() {
        panic!(
            "spack location -i petsc failed. Make sure PETSc is installed via Spack.\nError: {}",
            String::from_utf8_lossy(&petsc_output.stderr)
        );
    }
    let petsc_dir = String::from_utf8_lossy(&petsc_output.stdout)
        .trim()
        .to_string();
    let petsc_dir_path = PathBuf::from(&petsc_dir);

    // Spack installs PETSc globally in its prefix, never uses PETSC_ARCH subdirs
    let petsc_include = petsc_dir_path.join("include");
    let petsc_lib = petsc_dir_path.join("lib");

    println!("cargo::rustc-link-search=native={}", petsc_lib.display());
    println!("cargo::rustc-link-lib=dylib=petsc");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", petsc_lib.display());

    cc::Build::new()
        .compiler("clang")
        .file("petsc_wrapper.c")
        .include(petsc_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("petsc_wrapper");

    // ----------------------------------------------------
    // strictly use Spack for Eigen (ignore EIGEN_DIR)
    // ----------------------------------------------------
    let eigen_output = Command::new("spack")
        .args(["location", "-i", "eigen"])
        .output()
        .expect("Failed to execute spack command for Eigen. Is spack sourced & in your PATH?");

    if !eigen_output.status.success() {
        panic!(
            "spack location -i eigen failed. Make sure Eigen is installed via Spack.\nError: {}",
            String::from_utf8_lossy(&eigen_output.stderr)
        );
    }
    // ----------------------------------------------------
    // C++ Eigen compilation
    // ----------------------------------------------------
    let eigen_dir = PathBuf::from(
        String::from_utf8_lossy(&eigen_output.stdout).trim(),
    );
    let eigen_include = eigen_dir.join("include/eigen3");
    cc::Build::new()
        .cpp(true)
        .file("eigen_wrapper.cpp")
        .include(eigen_include)
        .compiler("clang++")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-w") // Suppress all internal Eigen C++ warnings
        .flag("-flto")
        .compile("eigen_wrapper");

    // ----------------------------------------------------
    // C Intel MKL compilation
    // ----------------------------------------------------
    let mkl_output = Command::new("spack")
        .args(["location", "-i", "intel-oneapi-mkl"])
        .output()
        .expect("Failed to execute spack command for MKL. Is spack sourced & in your PATH?");

    if !mkl_output.status.success() {
        panic!(
            "spack location -i intel-oneapi-mkl failed. Make sure MKL is installed via Spack.\nError: {}",
            String::from_utf8_lossy(&mkl_output.stderr)
        );
    }
    let mkl_prefix = PathBuf::from(
        String::from_utf8_lossy(&mkl_output.stdout).trim(),
    );
    let mkl_include = mkl_prefix.join("mkl/latest/include");
    let mkl_lib = mkl_prefix.join("mkl/latest/lib");

    println!("cargo::rustc-link-search=native={}", mkl_lib.display());
    // MKL Sequential Single-Threaded Link Line
    // Force the linker to keep all MKL libraries even if not directly referenced by our object files
    // We pass them as a single comma-separated linker argument to bypass rustc reordering
    println!("cargo::rustc-link-arg=-Wl,--no-as-needed,-lmkl_intel_lp64,-lmkl_sequential,-lmkl_core,--as-needed");
    println!("cargo::rustc-link-lib=dylib=pthread");
    println!("cargo::rustc-link-lib=dylib=m");
    println!("cargo::rustc-link-lib=dylib=dl");
    // Inject runtime library path (RPATH) so cargo bench executes without LD_LIBRARY_PATH
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", mkl_lib.display());

    cc::Build::new()
        .file("mkl_wrapper.c")
        .include(mkl_include)
        .compiler("clang")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("mkl_wrapper");

    // ----------------------------------------------------
    // C++ PSBLAS compilation
    // ----------------------------------------------------
    let psblas_dir = PathBuf::from("local/psblas3");
    let psblas_include = psblas_dir.join("include");
    let psblas_lib = psblas_dir.join("lib");

    let mpi_output = Command::new("spack")
        .args(["location", "-i", "openmpi"])
        .output()
        .expect("Failed to execute spack command for OpenMPI. Is spack sourced & in your PATH?");

    if !mpi_output.status.success() {
        panic!(
            "spack location -i openmpi failed. Make sure OpenMPI is installed via Spack.\nError: {}",
            String::from_utf8_lossy(&mpi_output.stderr)
        );
    }
    let mpi_dir = String::from_utf8_lossy(&mpi_output.stdout)
        .trim()
        .to_string();
    let mpi_lib = PathBuf::from(&mpi_dir).join("lib");
    let mpi_include = PathBuf::from(&mpi_dir).join("include");

    println!("cargo::rustc-link-search=native={}", psblas_lib.display());
    println!("cargo::rustc-link-search=native={}", mpi_lib.display());

    cc::Build::new()
        .cpp(true)
        .file("psblas_wrapper.cpp")
        .include(psblas_include)
        .include(mpi_include)
        .compiler("clang++")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-Wno-return-type-c-linkage") // Suppress PSBLAS third-party header warnings for std::complex C-linkage
        .flag("-Wno-unused-parameter")
        .flag("-flto")
        .compile("psblas_wrapper");

    // Force linkage by using link-arg instead of link-lib static, as rustc will
    // aggressively discard static archives if no Rust FFI directly calls them,
    // ignoring our hidden C++ wrapper dependencies.
    println!("cargo::rustc-link-arg=-Wl,--push-state,--no-as-needed");
    println!("cargo::rustc-link-arg=-lpsb_cbind");
    println!("cargo::rustc-link-arg=-lpsb_linsolve");
    println!("cargo::rustc-link-arg=-lpsb_prec");
    println!("cargo::rustc-link-arg=-lpsb_ext");
    println!("cargo::rustc-link-arg=-lpsb_util");
    println!("cargo::rustc-link-arg=-lpsb_base");
    println!("cargo::rustc-link-arg=-Wl,--pop-state");

    // Link Fortran runtime
    println!("cargo::rustc-link-lib=dylib=gfortran");

    // Link MPI
    println!("cargo::rustc-link-lib=dylib=mpi_usempif08");
    println!("cargo::rustc-link-lib=dylib=mpi_usempi_ignore_tkr");
    println!("cargo::rustc-link-lib=dylib=mpi_mpifh");
    println!("cargo::rustc-link-lib=dylib=mpi");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", mpi_lib.display());

    // Recompilation triggers
    println!("cargo::rerun-if-changed=petsc_wrapper.c");
    println!("cargo::rerun-if-changed=eigen_wrapper.cpp");
    println!("cargo::rerun-if-changed=mkl_wrapper.c");
    println!("cargo::rerun-if-changed=psblas_wrapper.cpp");
}
