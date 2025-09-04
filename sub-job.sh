#!/bin/bash
set -e
module purge
module use /path/to/HPCtoolkit/modulefiles
module load LScompiler25/
module load LSmpi25/
module load LSmathlib25/
module list

cd /path/to/your/working/directory
echo "Running Debug Job !!!!!"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/HPCtoolkit/ml/lib/sve

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export MEMKIND_HBW_NODES=$OPM_NODES
export ASAN_OPTIONS=abort_on_error=1


input_file="input_files/inputs3d" # input_files/LWFA
install_prefix="warpx.install.debug"

cat $input_file > "output_files/warpx_debug_$(date '+%Y-%m-%d_%H-%M-%S').txt"

mpirun -np 1 --bind-to numa --map-by numa --allow-run-as-root \
    /path/to/your/working/directory/$install_prefix/bin/warpx.3d.MPI.OMP.DP.PDP.DEBUG \
    $input_file 2>&1 | tee -a "output_files/warpx_debug_$(date '+%Y-%m-%d_%H-%M-%S').out"

echo "Done!"
