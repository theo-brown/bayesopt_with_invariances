#!/bin/bash
#SBATCH -J ex1mvr
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p sapphire
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

# Set up hardware
unset OMP_NUM_THREADS

# Set up environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source /home/tab53/.bashrc

# Change directory
cd /home/tab53/rds/rds-ukaea-ap002-mOlK9qn0PlQ/tab53/bayesopt_with_invariances/experiments/synthetic

# Run the script
julia --project --threads auto experiment_1_mvr.jl