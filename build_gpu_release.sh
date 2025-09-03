#!/bin/bash
set -e
module purge

echo "Loading modules for GPU build..."
module load mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2
module load CUDA/12.2

echo "Loaded modules:"
module list

home_path="/path/to/your/working/directory"
warpx_path="$home_path/warpx-24.07/"
install_path="$home_path/warpx.release.gpu"
build_dir="warpx.build.gpu.release" 

cd "$warpx_path"

echo "Configuring WARPX for NVIDIA A800 GPU..."

cmake -S . -B "$build_dir" \
        -DCMAKE_INSTALL_PREFIX="$install_path" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DWarpX_COMPUTE=CUDA \
        -DWarpX_amrex_src=../deps/amrex-development \
        -DWarpX_DIMS="3" \
        -DWarpX_MPI=ON \
        -DWarpX_PYTHON=OFF \
        -DWarpX_GPU_BACKEND=CUDA \
        -DAMReX_CUDA_ARCH=8.0 \
        -DWarpX_FFT=OFF \
        -DWarpX_OPENPMD=OFF \
        -DWarpX_QED=OFF \
        -DWarpX_MPI_THREAD_MULTIPLE=OFF \
        -DAMReX_BASE_PROFILE=OFF \
        -DAMReX_TRACE_PROFILE=OFF  \
        -DAMReX_COMM_PROFILE=OFF \
        -DAMReX_TINY_PROFILE=ON \
        -DWarpX_IPO=ON \
        -DCMAKE_CXX_FLAGS=" -O3 -flto"

echo "Building WARPX GPU version..."
cmake --build "$build_dir" --target install -j 16 


echo "Build Complete! Running Test..."
cd "$home_path" 

export OMP_NUM_THREADS=1 


EXECUTABLE_NAME="warpx.3d.MPI.CUDA.DP.PDP" 

if [ ! -f "$install_path/bin/$EXECUTABLE_NAME" ]; then
    echo "Error: Executable '$EXECUTABLE_NAME' not found!"
    echo "Please check the '$install_path/bin/' directory for the correct file name."
    ls -l "$install_path/bin/"
    exit 1
fi

sbatch run_warpx_gpu_release.sh

echo "Done!"