#!/bin/bash
set -e
module purge
module use /path/to/HPCtoolkit/modulefiles
module load LScompiler25/
module load LSmpi25/
module load LSmathlib25/
module list

sudo cpupower frequency-set -f  "xxx GHz"

cd /path/to/your/working/directory
echo "Running Release Job !!!!!"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/HPCtoolkit/ml/lib/sve

export OMP_NUM_THREADS=xx
export OMP_PROC_BIND=true
export MEMKIND_HBW_NODES=$OPM_NODES


input_file="input_files/inputs3d-l" # input_files/LWFA
install_prefix="warpx.install.release"

cat $input_file > "output_files/warpx_release_$(date '+%Y-%m-%d_%H-%M-%S').txt"

mpirun -np xx --bind-to numa --map-by numa --allow-run-as-root \
    /path/to/your/working/directory/$install_prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
    $input_file 2>&1 | tee -a "output_files/warpx_release_$(date '+%Y-%m-%d_%H-%M-%S').out"


echo "Done!"
