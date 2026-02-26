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
        .file("petsc_wrapper.c")
        .include(petsc_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
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
    let eigen_dir = PathBuf::from("resources/eigen");
    cc::Build::new()
        .cpp(true)
        .file("eigen_wrapper.cpp")
        .include(eigen_dir)
        .compiler("clang++")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-w") // Suppress all internal Eigen C++ warnings
        .compile("eigen_wrapper");

    // ----------------------------------------------------
    // C Intel MKL compilation
    // ----------------------------------------------------
    // Using the explicit Spack MKL path provided by the user
    let mkl_prefix = PathBuf::from("/roberto/llombardo/spack/opt/spack/linux-icelake/intel-oneapi-mkl-2024.2.2-3yphy2srhn5noy4jw7njcwotmjszg3ap");
    let mkl_include = mkl_prefix.join("mkl/2024.2/include");
    let mkl_lib = mkl_prefix.join("mkl/2024.2/lib");

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
        .compile("mkl_wrapper");

    // Recompilation triggers
    println!("cargo::rerun-if-changed=petsc_wrapper.c");
    println!("cargo::rerun-if-changed=eigen_wrapper.cpp");
}
