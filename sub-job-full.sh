#!/bin/bash
# dsub -n "warpx" -nn 1 -rpn 608 -I bash
# dsub -n "warpx" -nn 1 -rpn 608 -nl J003H1S07N08  -I bash
# source ~/SYSU/env.sh
# allocnode 1 J004H1S15N05
set -e
# source /pacific_fs/SYSU/env.sh
module purge
# module use /pacific_fs/HPCKit/latest/modulefiles
# module load bisheng/compiler4.1.0/bishengmodule
# module load bisheng/hmpi2.4.3/hmpi
# # module load bisheng/kml25.0.0/kspblas/omp
# module load bisheng/kml25.0.0/kml
module use /pacific_fs/HPCKit/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml
# module use /pacific_fs/hpctool/HPCKit/latest/modulefiles

# module load bisheng/compiler/bishengmodule
# module load gcc/compiler/gccmodule
# module load bisheng/hmpi/hmpi
# module load gcc/hmpi/hmpi
# module use /pacific_fs/HPCKit/HPCKit-0407/modulefiles
# module load bisheng/compiler4.2.0.1/bishengmodule
# module load bisheng/hmpi25.1.0/release
# module load bisheng/kml25.1.0/kml
# module load bisheng/compiler4.2.0/bishengmodule
# module load bisheng/hmpi25.0.0/hmpi
# module load bisheng/kml25.0.0/kml
# module load bisheng/kml/kblas/omp
module list
# export KML_FFT_TYPE=omp
# export FFTW3_INCLUDE_DIRS=/pacific_fs/HPCKit/latest/kml/bisheng-0126/include
# export FFTW3_LIBRARIES=/pacific_fs/HPCKit/latest/kml/bisheng-0126/lib/noarch
sudo cpupower frequency-set -f  1.55GHz #设置一次就行

cd /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/
echo "Running job !!!!!"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/latest/kml/bisheng/lib/sve512
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/25.3.30/kml/bisheng-0126/lib/sve512/

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=true
export MEMKIND_HBW_NODES=16-31

gfortran_path="/pacific_fs/HPCKit/25.3.30/compiler/gcc/lib64"
export devkit_root="/pacific_fs/software/tool/DevKit-CLI-25.1.RC1-Linux-Kunpeng"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$devkit_root/tuner/lib
export LD_LIBRARY_PATH=${gfortran_path}:$LD_LIBRARY_PATH

# prefix="warpx.globalsort.v516"
prefix="warpx.main"
input_file="inputs3d-l"
cat $input_file > out.$prefix
# prefix="warpx.full.v516.gap0.25.timer"
# prefix="warpx.full.v516.gap0.static"

# mpirun -np 1 --bind-to none --allow-run-as-root \
#     numactl -N 0 \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
#     inputs3d-l-order3 2>&1 | tee out.$prefix

# mpirun -np 1 --bind-to none --allow-run-as-root \
#     numactl --physcpubind=0-31 --membind=0 \
#     perf stat -I 1000 -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,branch-misses,dTLB-load-misses,l2d_cache,l2d_cache_refill,stalled-cycles-backend,stalled-cycles-frontend \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
#     inputs3d-l-order3 2>&1 | tee out.$prefix

# mpirun -np 1 --bind-to none --allow-run-as-root \
#     numactl --physcpubind=0-32 --membind=0 \
#     perf record -e cache-misses,L1-dcache-load-misses \
#     -g -- /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
#     inputs3d-l-order3 2>&1 | tee out.$prefix

# perf report --sort dso,symbol,srcline

# mpirun -np 1 --bind-to numa --map-by numa --allow-run-as-root \
#     devkit tuner roofline -m total --hbm-mode cache \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
#     inputs3d-l-order3 2>&1 | tee out.$prefix

$devkit_root/devkit tuner roofline -m region --hbm-mode cache \
    /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
    $input_file 2>&1 | tee -a out.$prefix
# mpirun -np 16 --bind-to numa --map-by numa --allow-run-as-root \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/$prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
#     $input_file 2>&1 | tee -a out.$prefix
    # inputs3d-l-order3 2>&1 | tee out.$prefix
    # LWFA
    #  inputs3d-l | tee out.txt

echo "Done!"
