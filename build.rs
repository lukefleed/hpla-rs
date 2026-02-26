//! Custom build script for compiling the FFI C wrapper.
//!
//! Automatically locates the PETSc installation via Spack and passes
//! aggressive optimization flags to `clang` before linking it statically
//! to our Rust crate.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Attempt to read PETSC_DIR from env, fallback to spack if not present
    let petsc_dir = env::var("PETSC_DIR").unwrap_or_else(|_| {
        let output = Command::new("spack")
            .args(["location", "-i", "petsc"])
            .output()
            .expect(
                "Failed to execute spack. Make sure spack is in PATH or set PETSC_DIR manually",
            );

        if !output.status.success() {
            panic!(
                "spack location -i petsc failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        String::from_utf8_lossy(&output.stdout).trim().to_string()
    });

    let petsc_dir_path = PathBuf::from(&petsc_dir);

    // Auto-detect PETSC_ARCH.
    // Spack installations typically don't use PETSC_ARCH (they install directly into prefix),
    // but we can check if a sub-directory in lib/petsc named like arch-* exists just in case.
    let petsc_arch = env::var("PETSC_ARCH").unwrap_or_else(|_| {
        "".to_string() // Assume prefix installation where include/ and lib/ are top-level
    });

    let (include_dir1, include_dir2, lib_dir) = if petsc_arch.is_empty() {
        (
            petsc_dir_path.join("include"),
            petsc_dir_path.join("include"), // Duplicate just to make logic symmetric
            petsc_dir_path.join("lib"),
        )
    } else {
        (
            petsc_dir_path.join("include"),
            petsc_dir_path.join(&petsc_arch).join("include"),
            petsc_dir_path.join(&petsc_arch).join("lib"),
        )
    };

    println!("cargo::rustc-link-search=native={}", lib_dir.display());
    println!("cargo::rustc-link-lib=dylib=petsc");

    cc::Build::new()
        .file("petsc_wrapper.c")
        .include(include_dir1)
        .include(include_dir2)
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("petsc_wrapper");

    // Auto-detect Eigen via Spack
    let eigen_dir = env::var("EIGEN_DIR").unwrap_or_else(|_| {
        let output = Command::new("spack")
            .args(["location", "-i", "eigen"])
            .output()
            .expect("Failed to execute spack for eigen");

        if !output.status.success() {
            panic!(
                "spack location -i eigen failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        String::from_utf8_lossy(&output.stdout).trim().to_string()
    });

    let eigen_include = PathBuf::from(&eigen_dir).join("include").join("eigen3");

    cc::Build::new()
        .cpp(true)
        .file("eigen_wrapper.cpp")
        .include(eigen_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("eigen_wrapper");

    println!("cargo::rerun-if-changed=petsc_wrapper.c");
    println!("cargo::rerun-if-changed=eigen_wrapper.cpp");
    println!("cargo::rerun-if-env-changed=PETSC_DIR");
    println!("cargo::rerun-if-env-changed=PETSC_ARCH");
    println!("cargo::rerun-if-env-changed=EIGEN_DIR");
}
