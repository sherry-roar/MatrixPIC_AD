# MatrixPIC-AD

This repository contains the source code, build scripts, and performance results for the MatrixPIC project. The project focuses on optimizing the **WarpX** Particle-in-Cell (PIC) simulation framework for the **LX2 architecture**, leveraging advanced features of MPU.

The provided modifications are based on [WarpX v24.07](https://github.com/BLAST-WarpX/warpx.git) and [AMReX v24.07](https://github.com/AMReX-Codes/amrex.git) and are designed to be compiled within a specific High-Performance Computing (HPC) environment.

## Key Features

  * **Architecture Optimization**: Highly optimized for the LX2 platform.
  * **Advanced Vectorization**: Utilizes VPU and MPU with a 512-bit vector width for maximum performance.
  * **Hybrid Parallelism**: Employs an MPI + OpenMP parallel programming model.
  * **Custom Memory Management**: Integrates the `memkind` library to enable explicit allocation on On-Package Memory (OPM).
  * **Specialized Compiler Stack**: Built and tested with the BiSheng Compiler and HMPI.

## Prerequisites and Environment

This project is intended to be built on an HPC cluster that uses the `module` system for environment management.

### 1\. Required Software Modules

The build scripts will automatically load the following environment modules. Please ensure they are available on your system.

```bash
module use /path/to/HPCKit_root/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml
```

### 2\. External Libraries

  * **Memkind**: This library is required for OPM support. The build scripts expect it to be pre-installed in a known location (e.g., `/path/to/memkind_root/`).

## Directory Structure

Before running the build scripts, ensure your project is structured as follows. The official source codes for AMReX and PICSAR should be placed in the `deps` directory.

```
.
├── deps/
│   └── amrex-24.07/      # AMReX v24.07 source code (modified)
├── warpx-24.07/          # WarpX v24.07 source code (modified)
├── input_files/
├── utils/
├── build_release_and_run.sh.sh
├── build_debug_and_run.sh.sh
└── sub-job-full.sh
```

## Setup and Integration

To enable the MatrixPIC optimizations, you must first apply the source code modifications to the official v24.07 releases of WarpX and AMReX.

### 1\. WarpX Modifications (v24.07)

Copy or replace the following files from this repository into your local WarpX source tree.

1.  **Add OPM Allocator Header**:

      * Copy `warpx-24.07/Source/OPM_Allocator.H` to your `warpx-24.07/Source/` directory.
  
2.  **Add RankSortStats Header**:

      * Copy `warpx-24.07/Source/RankSortStats.H` to your `warpx-24.07/Source/` directory.

3.  **Replace Core Source Files**:

      * Replace `main.cpp`, `WarpX.cpp`, and `WarpX.H` in `warpx-24.07/Source/`.

4.  **Replace Particle Container Files**:

      * Replace the corresponding files in `warpx-24.07/Source/Particles/`.
          * `WarpXParticleContainer.cpp` / `.H`
          * `PhysicalParticleContainer.cpp` / `.H`
          * `MultiParticleContainer.cpp` / `.H`

5.  **Replace Current Deposition Header**:

      * Replace `CurrentDeposition.H` in `warpx-24.07/Source/Particles/Deposition/`.

6.  **Replace CMake Build File**:

      * Replace `CMakeLists.txt` in `warpx-24.07/`.

### 2\. AMReX Modifications (v24.07)

Replace the following header file in your local AMReX source tree.

1.  **Replace Particle Tile Header**:
      * Replace `AMReX_ParticleTile.H` in `deps/amrex-24.07/Src/Particle/`.

## Building and Running

Once the source files are in place, you can compile and execute the simulation using the provided scripts.

1.  **Configure Build Options (Optional)**:
    You can modify the variables at the top of the `build_release_and_run.sh.sh` script to control the build:

      * `ENABLE_ROOFLINE_PROFILING`: Set to `"ON"` to enable Roofline performance profiling.
      * `ENABLE_SANITIZE`: Set to `"ON"` to enable AddressSanitizer for debugging.
      * `install_prefix`: Defines the build output directory name.

2.  **Execute the Build Script**:
    Run the release build script from the project's root directory:

    ```bash
    bash ./build_release_and_run.sh.sh
    ```

    This script automates the entire process:

      * Sets environment variables and loads modules.
      * Configures the project using CMake with architecture-specific optimization flags.
      * Compiles the code in parallel (`-j16`).
      * Updates the `sub-job-full.sh` run script with the correct build path.
      * Submits the job to a compute node via `ssh` and saves the output to `warpx_release.txt`.

### Debug Build

For development and debugging, a separate script is provided which builds with debug symbols and without high-level optimizations.

```bash
bash ./build_debug_and_run.sh.sh
```

## Included Components

  * **`input_files/`**: Contains sample input files (`inputs3d`, `LWFA`, etc.) for running different simulation scenarios.
  * **`utils/`**: Contains helper scripts for post-processing and analysis, such as `calculate_particle_avg_metric.py`.
  * **`eurosys_outputs/`**: Contains sample output logs and performance data from various test runs, corresponding to results presented in our EuroSys '26 paper.

-----

## Project File Tree

```
.
├── .gitignore
├── LICENSE
├── README.md
├── build_debug_and_run.sh.sh
├── build_release_and_run.sh.sh
├── deps
│   └── amrex-24.07
│       └── Src
│           └── Particle
│               └── AMReX_ParticleTile.H
├── eurosys_outputs
│   ├── sec6.2
│   │   └── ... (benchmark results)
│   └── sec6.3
│       └── ... (benchmark results)
├── input_files
│   ├── LWFA
│   ├── inputs3d
│   └── inputs3d-l
├── sub-job-full.sh
├── sub-job.sh
├── utils
│   ├── calculate_particle_avg_metric.py
│   └── transfer_cycles2times.sh
└── warpx-24.07
    ├── CMakeLists.txt
    └── Source
        ├── OPM_Allocator.H
        ├── RankSortStats.H
        ├── main.cpp
        ├── WarpX.cpp
        ├── WarpX.H
        └── Particles
            ├── Deposition
            │   └── CurrentDeposition.H
            ├── MultiParticleContainer.cpp
            ├── MultiParticleContainer.H
            ├── PhysicalParticleContainer.cpp
            ├── PhysicalParticleContainer.H
            ├── WarpXParticleContainer.cpp
            └── WarpXParticleContainer.H
```