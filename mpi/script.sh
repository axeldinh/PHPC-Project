#! /bin/bash -l

#SBATCH -N 4
#SBATCH -n 112
#SBATCH --account=phpc2021

module purge
module load gcc openmpi openblas

for i in 1 2 4 8 16 32 64 112
do
  srun -n $i ./cgsolver lap2D_5pt_n100.mtx
done
