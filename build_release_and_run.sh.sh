#!/bin/bash

set -e
module purge
module use /path/to/HPCKit_root/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml

cd /path/to/your/working/directory
cd warpx-24.07/
echo "====================Building WARPX (Release)===================="

ENABLE_SANITIZE="OFF"
CXX_FLAGS_ADD=""
if [ "$ENABLE_SANITIZE" = "ON" ]; then
    CXX_FLAGS_ADD="${LINKER_FLAGS_ADD} -fsanitize=address,alignment,leak,undefined"
fi

export LD_LIBRARY_PATH=/path/to/memkind_root/lib:$LD_LIBRARY_PATH

build_prefix="warpx.build.release"
install_prefix="warpx.install.release"

cmake -S . -B $build_prefix \
        -DWarpX_amrex_src=../deps/amrex-24.07 \
        -DWarpX_DIMS="3" \
        -DWarpX_MPI=ON \
        -DWarpX_PYTHON=OFF \
        -DWarpX_IPO=ON -DWarpX_FFT=OFF \
        -DWarpX_COMPUTE=OMP \
        -DWarpX_OPENPMD=OFF -DWarpX_QED=OFF \
        -DWarpX_MPI_THREAD_MULTIPLE=OFF \
        -DAMReX_BASE_PROFILE=OFF \
        -DAMReX_TRACE_PROFILE=OFF  \
        -DAMReX_COMM_PROFILE=OFF \
        -DAMReX_TINY_PROFILE=ON \
        -DCMAKE_INSTALL_PREFIX=/path/to/your/working/directory/$install_prefix \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++  \
        -DDEBUG_LOG=OFF -DBREAKDOWN=OFF -DMEASURE_IPC=OFF -DENABLE_OPM=ON -DENABLE_FLOPS=ON \
        -DCMAKE_CXX_FLAGS="-mcpu=hip11 \
                -march=armv9+sve+sve2+sme -rtlib=compiler-rt -ffast-math \
                -msve-vector-bits=512 -O3 -optimize-loop -flto=full -fopenmp -fveclib=MATHLIB -finline-functions -ftree-vectorize \
                -funroll-loops -fno-range-check -falign-functions   \
                -I/path/to/memkind_root/include/ \
                -mllvm -min-prefetch-stride=16 -mllvm -prefetch-distance=940 -w ${CXX_FLAGS_ADD} " \
        -DCMAKE_C_FLAGS="-mcpu=hip11 \
                -march=armv9+sve+sve2+sme -rtlib=compiler-rt -ffast-math \
                -msve-vector-bits=512 -O3 -optimize-loop -flto=full -fopenmp -fveclib=MATHLIB -finline-functions -ftree-vectorize \
                -funroll-loops -fno-range-check -falign-functions   \
                -I/path/to/memkind_root/include/ \
                -mllvm -min-prefetch-stride=16 -mllvm -prefetch-distance=940 -w ${CXX_FLAGS_ADD} "  \
        -DCMAKE_EXE_LINKER_FLAGS=" -fuse-ld=lld ${LINKER_FLAGS_ADD} \
                -Wl,-rpath,/path/to/HPCKit_root/25.3.30/compiler/bisheng/lib/aarch64-unknown-linux-gnu/ -lunwind \
                -Wl,-Bdynamic -L/path/to/HPCKit_root/25.3.30/compiler/bisheng/lib -ljemalloc -lmathlib -lm \
                -L/path/to/memkind_root/memkind/lib -lmemkind \
                -L/path/to/HPCKit_root/25.3.30/kml/bisheng-0126/lib/sve512/ -lkfft" \

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/HPCKit_root/25.3.30/kml/bisheng-0126/lib/sve
export MEMKIND_HBW_NODES=$OPM_NODES

cmake --build $build_prefix --target install -j16

ldd /path/to/your/working/directory/$install_prefix/bin/warpx.3d.MPI.OMP.DP.PDP
echo "====================Building WARPX (Release) Complete===================="
echo "Running... ..."
cd eurosys_outputs/

ssh NODE_NAME "sh /path/to/your/working/directory/sub-job-full.sh" 2>&1 
        | tee  "warpx_release.txt"

echo "Done!"
