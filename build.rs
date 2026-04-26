//! Custom build script for compiling the FFI C/C++/Fortran wrappers.
//!
//! Locates all library installations strictly via `spack location -i <pkg>`.
//! C/C++ wrappers use clang/clang++ with `-flto` for cross-language LTO.
//! Fortran wrappers use gfortran with `-ffat-lto-objects` for lld compatibility.
//! The `-mtune=native` flag is required because Clang (unlike GCC) does not
//! imply it from `-march=native`.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Resolve clang/clang++ from the spack-managed LLVM installation.
    // All C/C++ FFI wrappers use clang for cross-language LTO with Rust.
    let llvm_output = Command::new("spack")
        .args(["location", "-i", "llvm"])
        .output()
        .expect("Failed to execute spack command for LLVM. Is spack sourced & in your PATH?");

    assert!(
        llvm_output.status.success(),
        "spack location -i llvm failed. Make sure LLVM is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&llvm_output.stderr)
    );
    let llvm_dir = PathBuf::from(String::from_utf8_lossy(&llvm_output.stdout).trim());
    let clang = llvm_dir.join("bin/clang");
    let clangxx = llvm_dir.join("bin/clang++");

    // Verify that the spack-installed LLVM major version matches rustc's
    // bundled LLVM.  A major-version mismatch causes rust-lld to reject the
    // clang-produced bitcode with "Unknown attribute kind", which is hard to
    // diagnose.  Fail early with a clear message instead.
    let clang_version_out = Command::new(&clang)
        .arg("--version")
        .output()
        .expect("Failed to run clang --version");
    let clang_ver_str = String::from_utf8_lossy(&clang_version_out.stdout);
    // clang --version: "clang version 21.1.8 ..."
    let clang_major: u32 = clang_ver_str
        .split_whitespace()
        .nth(2) // "21.1.8"
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse().ok())
        .expect("Could not parse clang major version from 'clang --version'");

    let rustc_llvm_out = Command::new("rustc")
        .args(["--version", "--verbose"])
        .output()
        .expect("Failed to run rustc --version --verbose");
    let rustc_llvm_str = String::from_utf8_lossy(&rustc_llvm_out.stdout);
    // Output contains: "LLVM version: 21.1.8"
    let rustc_llvm_major: u32 = rustc_llvm_str
        .lines()
        .find(|l| l.starts_with("LLVM version:"))
        .and_then(|l| l.split_whitespace().nth(2))
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse().ok())
        .expect("Could not parse LLVM version from 'rustc --version --verbose'");

    assert!(
        clang_major == rustc_llvm_major,
        "LLVM major version mismatch: clang is LLVM {clang_major}, \
         rustc ships LLVM {rustc_llvm_major}. \
         Cross-language LTO requires matching major versions. \
         Update spack.yaml: change 'llvm@{clang_major}' to 'llvm@{rustc_llvm_major}', \
         then run 'spack concretize -f && spack install'."
    );

    // Resolve gfortran from the spack-managed gcc installation.
    // Used for the Fortran PSBLAS wrapper.
    let gcc_output = Command::new("spack")
        .args(["location", "-i", "gcc"])
        .output()
        .expect("Failed to execute spack command for gcc. Is spack sourced & in your PATH?");

    assert!(
        gcc_output.status.success(),
        "spack location -i gcc failed. Make sure gcc is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&gcc_output.stderr)
    );
    let gcc_dir = PathBuf::from(String::from_utf8_lossy(&gcc_output.stdout).trim());
    let gfortran = gcc_dir.join("bin/gfortran");
    let gcc_lib = gcc_dir.join("lib");
    let gcc_lib64 = gcc_dir.join("lib64");

    if gcc_lib.exists() {
        println!("cargo::rustc-link-search=native={}", gcc_lib.display());
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", gcc_lib.display());
    }
    if gcc_lib64.exists() {
        println!("cargo::rustc-link-search=native={}", gcc_lib64.display());
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", gcc_lib64.display());
    }

    let petsc_output = Command::new("spack")
        .args(["location", "-i", "petsc"])
        .output()
        .expect("Failed to execute spack command for PETSc. Is spack sourced & in your PATH?");

    assert!(
        petsc_output.status.success(),
        "spack location -i petsc failed. Make sure PETSc is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&petsc_output.stderr)
    );
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
        .compiler(&clang)
        .file("ffi/spmv/petsc.c")
        .include(&petsc_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("petsc_wrapper");

    // One-pass Lanczos via PETSc.
    cc::Build::new()
        .compiler(&clang)
        .file("ffi/lanczos/petsc_lanczos.c")
        .include(&petsc_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("petsc_lanczos_wrapper");

    // Two-pass Lanczos via PETSc.
    cc::Build::new()
        .compiler(&clang)
        .file("ffi/lanczos/petsc_lanczos_two_pass.c")
        .include(&petsc_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("petsc_lanczos_two_pass_wrapper");

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
    // Eigen is header-only: no rustc-link-lib, rustc-link-search, or rpath needed.
    let eigen_dir = PathBuf::from(String::from_utf8_lossy(&eigen_output.stdout).trim());
    let eigen_include = eigen_dir.join("include/eigen3");
    cc::Build::new()
        .cpp(true)
        .file("ffi/spmv/eigen.cpp")
        .include(&eigen_include)
        .compiler(&clangxx)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-w") // Suppress all internal Eigen C++ warnings
        .flag("-flto")
        .compile("eigen_wrapper");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("ffi/lanczos/eigen_lanczos_two_pass.cpp")
        .include(&eigen_include)
        .compiler(&clangxx)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-w") // Suppress all internal Eigen C++ warnings
        .flag("-flto")
        .compile("eigen_lanczos_two_pass_wrapper");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("ffi/lanczos/eigen_lanczos.cpp")
        .include(&eigen_include)
        .compiler(&clangxx)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-w")
        .flag("-flto")
        .compile("eigen_lanczos_wrapper");

    // ----------------------------------------------------
    // C Intel MKL compilation
    // ----------------------------------------------------
    let mkl_output = Command::new("spack")
        .args(["location", "-i", "intel-oneapi-mkl"])
        .output()
        .expect("Failed to execute spack command for MKL. Is spack sourced & in your PATH?");

    assert!(
        mkl_output.status.success(),
        "spack location -i intel-oneapi-mkl failed. Make sure MKL is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&mkl_output.stderr)
    );
    let mkl_prefix = PathBuf::from(String::from_utf8_lossy(&mkl_output.stdout).trim());
    let mkl_include = mkl_prefix.join("mkl/latest/include");
    let mkl_lib = mkl_prefix.join("mkl/latest/lib");

    println!("cargo::rustc-link-search=native={}", mkl_lib.display());
    // Force the linker to keep all MKL libraries even if not directly referenced by our object files
    // We pass them as a single comma-separated linker argument to bypass rustc reordering
    println!(
        "cargo::rustc-link-arg=-Wl,--no-as-needed,-lmkl_intel_lp64,-lmkl_sequential,-lmkl_core,--as-needed"
    );
    println!("cargo::rustc-link-lib=dylib=pthread");
    println!("cargo::rustc-link-lib=dylib=m");
    println!("cargo::rustc-link-lib=dylib=dl");
    // Inject runtime library path (RPATH) so cargo bench executes without LD_LIBRARY_PATH
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", mkl_lib.display());

    cc::Build::new()
        .file("ffi/spmv/mkl.c")
        .include(mkl_include)
        .compiler(&clang)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-flto")
        .compile("mkl_wrapper");

    // ----------------------------------------------------
    // C++ PSBLAS compilation
    // ----------------------------------------------------
    let psblas_output = Command::new("spack")
        .args(["location", "-i", "psblas"])
        .output()
        .expect("Failed to execute spack command for PSBLAS. Is spack sourced & in your PATH?");

    assert!(
        psblas_output.status.success(),
        "spack location -i psblas failed. Make sure PSBLAS is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&psblas_output.stderr)
    );
    let psblas_dir = PathBuf::from(String::from_utf8_lossy(&psblas_output.stdout).trim());
    let psblas_include = psblas_dir.join("include");
    let psblas_lib = psblas_dir.join("lib");

    let mpi_output = Command::new("spack")
        .args(["location", "-i", "openmpi"])
        .output()
        .expect("Failed to execute spack command for OpenMPI. Is spack sourced & in your PATH?");

    assert!(
        mpi_output.status.success(),
        "spack location -i openmpi failed. Make sure OpenMPI is installed via Spack.\nError: {}",
        String::from_utf8_lossy(&mpi_output.stderr)
    );
    let mpi_dir = String::from_utf8_lossy(&mpi_output.stdout)
        .trim()
        .to_string();
    let mpi_lib = PathBuf::from(&mpi_dir).join("lib");
    let mpi_include = PathBuf::from(&mpi_dir).join("include");

    println!("cargo::rustc-link-search=native={}", psblas_lib.display());
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", psblas_lib.display());
    println!("cargo::rustc-link-search=native={}", mpi_lib.display());

    cc::Build::new()
        .cpp(true)
        .file("ffi/spmv/psblas.cpp")
        .include(&psblas_include)
        .include(&mpi_include)
        .compiler(&clangxx)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-Wno-return-type-c-linkage") // Suppress PSBLAS third-party header warnings for std::complex C-linkage
        .flag("-Wno-unused-parameter")
        .flag("-flto")
        .compile("psblas_wrapper");

    let psblas_modules = psblas_dir.join("modules");

    // gfortran deposits .mod files in the cwd by default. Redirect them
    // to OUT_DIR via -J so the crate root stays clean and parallel
    // builds do not race on shared .mod files.
    let fortran_mod_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let fortran_mod_flag = format!("-J{fortran_mod_dir}");

    // One-pass and two-pass Lanczos for f(A)b = exp(-A)b. Fortran kernels +
    // shared C++ FFI shim must be compiled separately: clang++ has no Fortran
    // frontend.
    cc::Build::new()
        .file("ffi/lanczos/psblas_lanczos.f90")
        .include(&psblas_modules)
        .compiler(&gfortran)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-ffat-lto-objects")
        .flag(&fortran_mod_flag)
        .flag("-Wno-unused-dummy-argument")
        .compile("psblas_lanczos_fortran");

    cc::Build::new()
        .file("ffi/lanczos/psblas_lanczos_two_pass.f90")
        .include(&psblas_modules)
        .compiler(&gfortran)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-ffat-lto-objects")
        .flag(&fortran_mod_flag)
        .flag("-Wno-unused-dummy-argument")
        .compile("psblas_lanczos_two_pass_fortran");

    cc::Build::new()
        .cpp(true)
        .file("ffi/lanczos/psblas_lanczos.cpp")
        .include(&psblas_include)
        .include(&mpi_include)
        .compiler(&clangxx)
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-ffast-math")
        .flag("-Wno-return-type-c-linkage")
        .flag("-Wno-unused-parameter")
        .flag("-flto")
        .compile("psblas_lanczos_cpp");

    // Use link-arg instead of link-lib: rustc discards static archives when no
    // Rust FFI symbol references them directly, breaking our C++ wrapper deps.
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

    // Recompilation triggers: wrapper source files
    println!("cargo::rerun-if-changed=ffi/spmv/petsc.c");
    println!("cargo::rerun-if-changed=ffi/spmv/eigen.cpp");
    println!("cargo::rerun-if-changed=ffi/lanczos/eigen_lanczos.cpp");
    println!("cargo::rerun-if-changed=ffi/lanczos/eigen_lanczos_two_pass.cpp");
    println!("cargo::rerun-if-changed=ffi/spmv/mkl.c");
    println!("cargo::rerun-if-changed=ffi/spmv/psblas.cpp");
    println!("cargo::rerun-if-changed=ffi/lanczos/psblas_lanczos.f90");
    println!("cargo::rerun-if-changed=ffi/lanczos/psblas_lanczos_two_pass.f90");
    println!("cargo::rerun-if-changed=ffi/lanczos/psblas_lanczos.cpp");
    println!("cargo::rerun-if-changed=ffi/lanczos/petsc_lanczos.c");
    println!("cargo::rerun-if-changed=ffi/lanczos/petsc_lanczos_two_pass.c");
    // Force rebuild when spack environment changes (e.g. package reinstall)
    println!("cargo::rerun-if-env-changed=SPACK_ROOT");
}
