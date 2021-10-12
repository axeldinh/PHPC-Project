#! /bin/bash -l

#SBATCH --nodes=1
#SBATCH --account=phpc2021
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free

module purge
module load gcc openblas cuda

for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 5000 10000
do
  for ((i = 1; i <= 1024; i = i*2))
  do
    srun ./cgsolver lap2D_5pt_n100.mtx $i $j
  done
done
