#!/bin/bash

#SBATCH -p gpu               
#SBATCH -N 1                  
#SBATCH -n 8             
#SBATCH --job-name=release_warpxGPU
##SBATCH --gpus-per-node=8    
#SBATCH -o /path/to/your/working/directory/out/warpx_release_%j.out   
#SBATCH -e /path/to/your/working/directory/err/warpx_release_%j.err   

echo "#############################################"
echo "Job Started at $(date)"
echo "Job submitted to partition: $SLURM_JOB_PARTITION"
echo "Job running on specific node: $SLURM_NODELIST"
echo "#############################################"


set -e


module purge
module load mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2
module load CUDA/12.2

echo "Loaded modules:"
module list


home_path="/path/to/your/working/directory"
warpx_path="$home_path/warpx-24.07"
install_path="$home_path/warpx.release.gpu"

cd "$home_path" 


export OMP_NUM_THREADS=1

EXECUTABLE_NAME="warpx.3d.MPI.CUDA.DP.PDP" 
EXECUTABLE_PATH="$install_path/bin/$EXECUTABLE_NAME"


if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable '$EXECUTABLE_NAME' not found!"
    echo "Please check the '$install_path/bin/' directory for the correct file name."
    ls -l "$install_path/bin/"
    exit 1
fi

echo "Starting WarpX on node an45..."
cd "$home_path/result_gpu"
inputfile="$home_path/inputs_3d_l"

cat $inputfile > out.gpu.release.txt
srun bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  exec "$@"
' bash "$EXECUTABLE_PATH" "$inputfile" 2>&1 | tee -a out.gpu.release.txt


echo "Done!"
echo "#############################################"
echo "Job Finished at $(date)"
echo "#############################################"