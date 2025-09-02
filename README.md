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

Before running the build scripts, ensure your project is structured as follows. The official source codes for AMReX should be placed in the `deps` directory.

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
    You can modify the variables at the top of the `build_release_and_run.sh` script to control the build:

      * `ENABLE_ROOFLINE_PROFILING`: Set to `"ON"` to enable Roofline performance profiling.
      * `ENABLE_SANITIZE`: Set to `"ON"` to enable AddressSanitizer for debugging.
      * `install_prefix`: Defines the build output directory name.

2.  **Select the Deposition Algorithm for Ablation Studies**:
    Before compiling, you must manually select the particle deposition algorithm you wish to test. This is a required step for conducting ablation studies.

      * **File to Modify**: `warpx-24.07/Source/Particles/WarpXParticleContainer.cpp`
      * **Location**: Lines `1295` to `1387`.
      * **Action**: Within this code block, you will find several function calls, each corresponding to a different experiment. To run a specific experiment, ensure its function call is **uncommented**, while the others are **commented out**.

    For example, to enable "Experiment 1," which uses the `doDepositionShapeN_3d_sme` function, the code should be configured as follows:

    ```cpp
    // =================Experments 1================
    doDepositionShapeN_3d_sme<1>(
        GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
        uyp.dataPtr() + offset, uzp.dataPtr() + offset, ion_lev,
        jx_fab,  jy_fab, jz_fab, np_to_deposit, relative_time, dinv,
        xyzmin, lo, hi,len, q, test,test0,test1,ptile,tbox,
        test_rhocells,test_sxwq,WarpX::n_rz_azimuthal_modes,inner_timer);

    // Ensure other experimental or baseline versions are commented out.
    // doDepositionShapeN<1>(
    //     GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
    //     uyp.dataPtr() + offset, uzp.dataPtr() + offset, ion_lev,
    //     jx_fab, jy_fab, jz_fab, np_to_deposit, relative_time, dinv,
    //     xyzmin, lo, q,  WarpX::n_rz_azimuthal_modes);
    ```

3.  **Execute the Build Script**:
    Run the release build script from the project's root directory:

    ```bash
    bash ./build_release_and_run.sh
    ```

    This script automates the entire process:

      * Sets environment variables and loads modules.
      * Configures the project using CMake with architecture-specific optimization flags.
      * Compiles the code in parallel (`-j16`).
      * Updates the `sub-job-full.sh` run script with the correct build path.
      * Submits the job to a compute node via `ssh` and saves the output to `warpx_release.txt`.

### Debug Build

For development purposes, a separate script is provided to build the code in debug mode. This build process disables high-level compiler optimizations and includes debug symbols, making it suitable for use with debuggers like GDB.

Remember to select the desired particle deposition algorithm in the source code before running the debug build.

```bash
bash ./build_debug_and_run.sh.sh
```

## Included Components

  * **`input_files/`**: Contains sample input files (`inputs3d`, `LWFA`, etc.) for running different simulation scenarios.
  * **`utils/`**: Contains helper scripts for post-processing and analysis, such as `calculate_particle_avg_metric.py`.
  * **`eurosys_outputs/`**: Contains origin outputs results presented in our EuroSys '26 paper.

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
│   ├── sec6.1.xlsx
│   ├── sec6.2.xlsx
│   └── sec6.3.xlsx
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