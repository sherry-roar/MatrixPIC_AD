#!/bin/bash

set -e
module purge
module use /path/to/HPCKit_root/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml

cd /path/to/your/working/directory
cd warpx-24.07/
echo "====================Building WARPX (Debug)===================="

ENABLE_SANITIZE="OFF"
CXX_FLAGS_ADD=""
if [ "$ENABLE_SANITIZE" = "ON" ]; then
    CXX_FLAGS_ADD="${LINKER_FLAGS_ADD} -fsanitize=address,alignment,leak,undefined"
fi

export LD_LIBRARY_PATH=/path/to/memkind_root/lib:$LD_LIBRARY_PATH

build_prefix="warpx.build.debug"
install_prefix="warpx.install.debug"

cmake -S . -B $build_prefix \
        -DWarpX_amrex_src=../deps/amrex-development \
        -DWarpX_DIMS="3" \
        -DWarpX_MPI=ON \
        -DWarpX_PYTHON=OFF \
        -DWarpX_IPO=OFF -DWarpX_FFT=OFF \
        -DWarpX_COMPUTE=OMP \
        -DWarpX_OPENPMD=OFF -DWarpX_QED=OFF \
        -DWarpX_MPI_THREAD_MULTIPLE=OFF \
        -DAMReX_BASE_PROFILE=OFF \
        -DAMReX_TRACE_PROFILE=OFF  \
        -DAMReX_COMM_PROFILE=OFF \
        -DAMReX_TINY_PROFILE=ON \
        -DCMAKE_INSTALL_PREFIX=/path/to/your/working/directory/$install_prefix \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++  \
        -DDEBUG_LOG=ON -DBREAKDOWN=ON -DMEASURE_IPC=OFF -DENABLE_HBM=OFF -DENABLE_FLOPS=OFF \
        -DCMAKE_CXX_FLAGS="-mcpu=hip11 \
                -march=armv8.6-a+sve+sve2+sme -rtlib=compiler-rt \
                -I/path/to/memkind_root/include/ \
                -msve-vector-bits=512 -O0 -g -w \
                ${CXX_FLAGS_ADD} " \
        -DCMAKE_C_FLAGS="-mcpu=hip11 \
                -march=armv8.6-a+sve+sve2+sme -rtlib=compiler-rt \
                -I/path/to/memkind_root/include/ \
                -msve-vector-bits=512 -O0 -g -w \
                ${CXX_FLAGS_ADD} "  \
        -DCMAKE_EXE_LINKER_FLAGS=" -fuse-ld=lld ${LINKER_FLAGS_ADD} \
                -Wl,-rpath,/path/to/HPCKit_root/25.3.30/compiler/bisheng/lib/aarch64-unknown-linux-gnu/ -lunwind \
                -L/path/to/memkind_root/lib/ -lmemkind \
                -Wl,-Bdynamic -L/path/to/HPCKit_root/25.3.30/compiler/bisheng/lib -ljemalloc -lmathlib -lm \
                -L/path/to/HPCKit_root/25.3.30/kml/bisheng-0126/lib/sve512/ -lkfft -lkfft_omp \
                "      
    
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/HPCKit_root/25.3.30/kml/bisheng-0126/lib/sve
export MEMKIND_HBW_NODES=$HBM_NODES

cmake --build $build_prefix --target install -j16


ldd /path/to/your/working/directory/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG 
echo "====================Building WARPX (Debug) Complete===================="
echo "Running Debug Test... ..."

ssh NODE_NAME 'sh /path/to/your/working/directory/sub-job.sh' 2>&1 
        | tee "warpx_debug.txt"

echo "Done!"
