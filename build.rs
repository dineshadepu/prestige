use std::process::Command;
use std::path::Path;

fn main() {
    let kernel_dir = Path::new("cuda_kernels");

    for entry in std::fs::read_dir(kernel_dir).unwrap() {
        let path = entry.unwrap().path();

        if path.extension().and_then(|s| s.to_str()) == Some("cu") {
            println!("cargo:rerun-if-changed={}", path.display());

            let filename = path.file_name().unwrap().to_str().unwrap();

            // ---- Grid compiled as shared library ----
            if filename == "particles_grid.cu" {
                let status = Command::new("nvcc")
                    .args([
                        "-lib",
                        "-Xcompiler", "-fPIC",
                        path.to_str().unwrap(),
                        "-o",
                        "cuda_kernels/libgrid.a",
                        "-O3",
                        "-arch=sm_86",
                    ])
                    .status()
                    .expect("failed to compile grid");

                assert!(status.success());
            }

            // ---- Everything else → PTX ----
            else {
                let out = path.with_extension("ptx");

                let status = Command::new("nvcc")
                    .args([
                        "-ptx",
                        path.to_str().unwrap(),
                        "-o",
                        out.to_str().unwrap(),
                        "-O3",
                        "-arch=sm_86",
                        "--use_fast_math",
                    ])
                    .status()
                    .expect("failed to run nvcc");

                assert!(status.success());
            }
        }
    }
    println!("cargo:rustc-link-search=native=cuda_kernels");
    println!("cargo:rustc-link-lib=static=grid");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
