# Integrating MatrixPIC Modifications into WarpX and AMReX

This document provides instructions on how to integrate the MatrixPIC source code modifications into the official releases of WarpX (v24.07) and AMReX (v24.07). These changes are necessary to enable the MPU-accelerated features of MatrixPIC.

The entire framework is designed and optimized for the LX2 CPU architecture. If you have access to an LX2 system, you can build and run the code directly using the provided `build_warpx-release.sh` script after completing the setup below.

-----

### **1. WarpX Configuration**

To update the WarpX source code, copy or replace the following files from our provided artifact into your local WarpX repository.

1.  **Add the HBM Allocator Header**:

      * Copy `HBM_allocator.H` to `warpx-24.07/Source/`

2.  **Replace Core Source Files**:

      * Replace `warpx-24.07/Source/main.cpp` with the provided version.
      * Replace `warpx-24.07/Source/WarpX.cpp` with the provided version.
      * Replace `warpx-24.07/Source/WarpX.H` with the provided version.

3.  **Replace Particle Container Files**:

      * Replace `warpx-24.07/Source/Particles/WarpXParticleContainer.cpp` with the provided version.
      * Replace `warpx-24.07/Source/Particles/WarpXParticleContainer.H` with the provided version.
      * Replace `warpx-24.07/Source/Particles/PhysicalParticleContainer.H` with the provided version.
      * Replace `warpx-24.07/Source/Particles/PhysicalParticleContainer.cpp` with the provided version.
      * Replace `warpx-24.07/Source/Particles/MultiParticleContainer.cpp` with the provided version.
      * Replace `warpx-24.07/Source/Particles/MultiParticleContainer.H` with the provided version.

4.  **Replace Current Deposition Header**:

      * Replace `warpx-24.07/Source/Particles/Deposition/CurrentDeposition.H` with the provided version.

5.  **Replace CMake Build File**:

      * Replace `warpx-24.07/CMakeLists.txt` with the provided version.

After these steps, the WarpX codebase will be correctly configured to run on an MPU-enabled device.

-----

### **2. AMReX Configuration**

The modifications for AMReX are minimal and involve replacing a single header file.

1.  **Replace Particle Tile Header**:
      * Replace `amrex-24.07/Src/Particle/AMReX_ParticleTile.H` with the provided version.

With this change, the AMReX library is ready to support the MatrixPIC framework.

-----

### **3. Building and Running**

Once all files have been correctly placed, you can compile and run the project.

  * **On an LX2 System**: Navigate to the root directory of the project and execute the build script:
    ```bash
    ./build_warpx-release.sh
    ```

This script will handle the compilation and launch the simulation.
