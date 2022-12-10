#!/bin/bash

#SBATCH --output=slurm/%j.out                              
#SBATCH --error=slurm/%j.out                                 
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition scavenger

srun "$@"