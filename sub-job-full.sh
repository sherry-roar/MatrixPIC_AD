#!/bin/bash
set -e
module purge
module use /path/to/HPCKit_root/25.3.30/modulefiles
module load bisheng/compiler4.1.0/bishengmodule
module load bisheng/hmpi2.4.3/hmpi
module load bisheng/kml25.0.0/kml
module list

sudo cpupower frequency-set -f  "xxx GHz"

cd /path/to/your/working/directory
echo "Running Release Job !!!!!"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/HPCKit_root/25.3.30/kml/bisheng-0126/lib/sve512/

export OMP_NUM_THREADS=xx
export OMP_PROC_BIND=true
export MEMKIND_HBW_NODES=$HBM_NODES


input_file="input_files/inputs3d-l" # input_files/LWFA
install_prefix="warpx.install.release"

cat $input_file > "output_files/warpx_release_$(date '+%Y-%m-%d_%H-%M-%S').txt"

mpirun -np xx --bind-to numa --map-by numa --allow-run-as-root \
    /path/to/your/working/directory/$install_prefix/bin/warpx.3d.MPI.OMP.DP.PDP \
    $input_file 2>&1 | tee -a "output_files/warpx_release_$(date '+%Y-%m-%d_%H-%M-%S').out"


echo "Done!"
