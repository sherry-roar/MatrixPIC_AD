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

# module use /pacific_fs/hpctool/HPCKit/latest/modulefiles

# module load bisheng/compiler/bishengmodule
# module load gcc/compiler/gccmodule
# module load bisheng/hmpi/hmpi
# module load gcc/hmpi/hmpi
# module use /pacific_fs/HPCKit/latest/modulefiles
# module load bisheng/compiler4.2.0.1/bishengmodule
# module load bisheng/hmpi25.1.0/release
# module load bisheng/kml25.1.0/kml
module use /pacific_fs/HPCKit/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml
# module load bisheng/kml/kblas/omp
# module list
# export KML_FFT_TYPE=omp
# export FFTW3_INCLUDE_DIRS=/pacific_fs/HPCKit/latest/kml/bisheng-0126/include
# export FFTW3_LIBRARIES=/pacific_fs/HPCKit/latest/kml/bisheng-0126/lib/noarch
cd /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/

# sudo cpupower frequency-set -f  1.55GHz #设置一次就行
echo "Running job !!!!!"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/latest/kml/bisheng/lib/sve512
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pacific_fs/HPCKit/25.3.30/kml/bisheng-0126/lib/sve512/
export ASAN_OPTIONS=abort_on_error=1

export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true
export MEMKIND_HBW_NODES=16-31

gfortran_path="/pacific_fs/HPCKit/25.3.30/compiler/gcc/lib64"
export devkit_root="/pacific_fs/software/tool/DevKit-CLI-25.1.RC1-Linux-Kunpeng"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$devkit_root/tuner/lib
export LD_LIBRARY_PATH=${gfortran_path}:$LD_LIBRARY_PATH

ldd /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG 
cat inputs3d > out.txt

# sudo sysctl -w kernel.perf_event_paranoid=-1
# sudo sysctl -w kernel.kptr_restrict=0
mpirun -np 2 --bind-to numa --map-by numa --allow-run-as-root -mca orte_base_help_aggregate 0 \
    $devkit_root/devkit tuner roofline -m region --hbm-mode cache \
    /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG \
    inputs3d 2>&1 | tee out.txt
# mpirun -np 2 --bind-to numa --map-by numa --allow-run-as-root \
#     /pacific_ext/SYSU/sysu04/Test_install_warpx_hw/warpx.vay.sve.25.test/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG \
#     inputs3d 2>&1 | tee out.txt
    # inputs3d-order3 2>&1 | tee out.txt
    #  LWFA | tee out.txt

echo "Done!"
