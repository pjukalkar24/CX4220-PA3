#!/bin/bash
#SBATCH --job-name=single-cpu-tests
#SBATCH --output=single-cpu-tests.out
#SBATCH --error=single-cpu-tests.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:05:30
lscpu
module load openmpi
make clean
make
python3 ./single-cpu-tests.py