//! Custom build script for compiling the FFI C wrapper.
//!
//! Automatically locates the PETSc and Eigen installations strictly via Spack
//! and passes aggressive optimization flags to `clang`/`clang++` before
//! linking them statically to our Rust crate.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // ----------------------------------------------------
    // strictly use Spack for PETSc (ignore PETSC_DIR/PETSC_ARCH)
    // ----------------------------------------------------
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
    let eigen_dir = String::from_utf8_lossy(&eigen_output.stdout)
        .trim()
        .to_string();
    let eigen_include = PathBuf::from(&eigen_dir).join("include").join("eigen3");

    cc::Build::new()
        .cpp(true)
        .file("eigen_wrapper.cpp")
        .include(eigen_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .compile("eigen_wrapper");

    // Recompilation triggers
    println!("cargo::rerun-if-changed=petsc_wrapper.c");
    println!("cargo::rerun-if-changed=eigen_wrapper.cpp");
}
