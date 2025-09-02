#!/bin/bash
# dsub -n "warpx" -nn 1 -rpn 608 -I bash
# dsub -n "warpx" -nn 1 -rpn 608 -nl J003H1S04N06  -I bash
# source ~/SYSU/env.sh
# allocnode 1 J004H1S15N05
set -e
# source /pacific_ext/huawei/SYSU/env.sh
module purge
module use /pacific_fs/HPCKit/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml
# module load bisheng/compiler4.1.0/bishengmodule
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_ext/huawei/HPCKit/25.3.30/compiler/bisheng/lib
# export PATH=$PATH:/pacific_ext/huawei/HPCKit/25.3.30/compiler/bisheng/bin
# module load bisheng/hmpi2.4.3/hmpi
# module load bisheng/kml25.0.0/kspblas/omp
# module load bisheng/kml25.0.0/kml

# module use /pacific_fs/HPCKit/latest/modulefiles
# module load bisheng/compiler4.2.0.1/bishengmodule
# module load bisheng/hmpi25.1.0/release
# module load bisheng/kml25.1.0/kml

# module load bisheng/compiler4.2.0.1/bishengmodule
# module load bisheng/hmpi25.0.0/hmpi
# module load bisheng/kml25.0.0/kml

# module use /pacific_ext/huawei/hpctool/HPCKit/latest/modulefiles

# module load bisheng/compiler/bishengmodule
# module load gcc/compiler/gccmodule
# module load bisheng/hmpi/hmpi
# module load gcc/hmpi/hmpi

# module load bisheng/kml/kblas/omp
# module list
# export KML_FFT_TYPE=omp
# export FFTW3_INCLUDE_DIRS=/pacific_ext/huawei/HPCKit/latest/kml/bisheng-0126/include
# export FFTW3_LIBRARIES=/pacific_ext/huawei/HPCKit/latest/kml/bisheng-0126/lib/noarch

cd /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/
cd WarpX-development/
echo "Building WARPX ... ..."

foldername="/pacific_ext/SYSU/sysu04/Test_install_warpx_hw/WarpX-development/warpx.vay.sve.25.test"
export devkit_root="/pacific_fs/software/tool/DevKit-CLI-25.1.RC1-Linux-Kunpeng"
gfortran_path="/pacific_fs/HPCKit/25.3.30/compiler/gcc/lib64"
ENABLE_ROOFLINE_PROFILING="ON"
ENABLE_SANITIZE="OFF"
# 找到设置 CMAKE_CXX_FLAGS 的地方，在后面追加 Roofline 的参数
# -DROOFLINE_PROFILING 激活源码中的 #ifdef
# -I/path/to/include 指定头文件路径
CXX_FLAGS_ADD=""
if [ "$ENABLE_ROOFLINE_PROFILING" = "ON" ]; then
    CXX_FLAGS_ADD="-DROOFLINE_EVENTS -I$devkit_root/tuner/include/roofline"
fi
if [ "$ENABLE_SANITIZE" = "ON" ]; then
    CXX_FLAGS_ADD="${LINKER_FLAGS_ADD} -fsanitize=address,alignment,leak,undefined"
fi

# 找到设置 CMAKE_EXE_LINKER_FLAGS 的地方，在后面追加 Roofline 的参数
# -L/path/to/lib 指定库路径
# -lrfevents 指定链接的库
LINKER_FLAGS_ADD=""
if [ "$ENABLE_ROOFLINE_PROFILING" = "ON" ]; then
        LINKER_FLAGS_ADD="-L$devkit_root/tuner/lib -lrfevents"
        # -L 告诉链接器在哪里找到库
        LINKER_FLAGS_ADD="${LINKER_FLAGS_ADD} -L${gfortran_path}"
        # -Wl,-rpath 将路径嵌入可执行文件，供运行时使用
        LINKER_FLAGS_ADD="${LINKER_FLAGS_ADD} -Wl,-rpath,${gfortran_path}"
fi

export LD_LIBRARY_PATH=/pacific_ext/SYSU/sysu04/memkind/lib:$LD_LIBRARY_PATH

# if [ ! -d "$foldername"  ]; then
        cmake -S . -B warpx.vay.sve.25.test \
        -DWarpX_amrex_src=../deps/amrex-development \
        -DWarpX_picsar_src=../deps/picsar-development \
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
        -DCMAKE_INSTALL_PREFIX=../warpx.vay.sve.25.test \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++  \
        -DDEBUG_LOG=ON -DBREAKDOWN=ON -DMEASURE_IPC=ON -DENABLE_HBM=OFF \
        -DROOFLINE_PROFILING="ENABLE_ROOFLINE_PROFILING" \
        -DCMAKE_CXX_FLAGS="-mcpu=hip11 \
                -march=armv8.6-a+sve+sve2+sme -rtlib=compiler-rt \
                -I/pacific_ext/SYSU/sysu04/memkind/include/ \
                -msve-vector-bits=512 -O0 -g -w \
                ${CXX_FLAGS_ADD} " \
        -DCMAKE_C_FLAGS="-mcpu=hip11 \
                -march=armv8.6-a+sve+sve2+sme -rtlib=compiler-rt \
                -I/pacific_ext/SYSU/sysu04/memkind/include/ \
                -msve-vector-bits=512 -O0 -g -w \
                ${CXX_FLAGS_ADD} "  \
        -DCMAKE_EXE_LINKER_FLAGS=" -fuse-ld=lld ${LINKER_FLAGS_ADD} \
                -Wl,-rpath,/pacific_fs/HPCKit/25.3.30/compiler/bisheng/lib/aarch64-unknown-linux-gnu/ -lunwind \
                -L/pacific_ext/SYSU/sysu04/memkind/lib/ -lmemkind \
                -Wl,-Bdynamic -L/pacific_fs/HPCKit/25.3.30/compiler/bisheng/lib -ljemalloc -lmathlib -lm \
                -L/pacific_fs/HPCKit/25.3.30/kml/bisheng-0126/lib/sve512/ -lkfft -lkfft_omp \
                "  
#-fuse-ld=lld 
# else
#         echo "NO CMAKE"
# fi       
    
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/25.3.30/kml/bisheng-0126/lib/sve
export LD_LIBRARY_PATH=${gfortran_path}:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/latest/kml/bisheng/lib/sve512
# 在这里追加新路径
if [ "$ENABLE_ROOFLINE_PROFILING" = "ON" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$devkit_root/tuner/lib
fi
export MEMKIND_HBW_NODES=16-31
cmake --build warpx.vay.sve.25.test --target install -j32

ldd /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG 
echo "Build Complete! Running Test... ..."
cd ..
# dsub -n "warpx" -nn 1 -rpn 608 -I -nl J003H1S16N0[3-8] 'sh /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/sub-job.sh' 2>&1 | tee sshout2.txt
# dsub -n "warpx" -nn 1 -rpn 608 -I -nl J003H1S04N0[1-8] 'sh /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/sub-job.sh' 2>&1 | tee sshout2.txt
ssh scguest01@J003H1S16N07 'sh /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/sub-job.sh' 2>&1 | tee sshout2.txt

# dsub -n "warpx" -nn 1 -rpn 608 -I "/pacific_ext/SYSU/sysu04/Test_install_warpx_hw/sub-job.sh" tee out.txt
# dsub -n "warpx" -nn 1 -rpn 608 -nl J004H1S12N05  -I "/pacific_ext/SYSU/sysu04/Test_install_warpx_hw/sub-job.sh" | tee out.txt
# export OMP_NUM_THREADS=1
# mpirun -np 1 --bind-to numa --map-by numa --allow-run-as-root \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.PSATD.QED.DEBUG \
#      inputs_3d_uniform | tee out.txt

echo "Done!"
