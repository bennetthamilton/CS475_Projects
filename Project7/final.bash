
#!/bin/bash
#SBATCH -J AutoCorr
#SBATCH -A cs475-575
#SBATCH -p classmpifinal
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --constraint=ib
#SBATCH -o autocorr.out
#SBATCH -e autocorr.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hamibenn@oregonstate.edu

module load openmpi
mpic++ proj07.cpp -o proj07 -lm

for b in 1 2 4 8 16
do
    mpiexec -np $b proj07 $PPSize
done