#!/bin/bash

#SBATCH --job-name=autograder
#SBATCH --output=autograder.out
#SBATCH --error=autograder.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:05:30

lscpu
module load openmpi
make clean
make

python3 ./autograder.py
