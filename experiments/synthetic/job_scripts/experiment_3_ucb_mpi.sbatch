#!/bin/bash
#SBATCH -J ex3ucb
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p sapphire
#SBATCH --output=%x_%j.out
#SBATCH --nodes=6
#SBATCH --ntasks=128     # 32 repeats x 4 kernels
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

# Set up hardware
unset OMP_NUM_THREADS
export I_MPI_PIN_DOMAIN=omp:compact
export I_MPI_PIN_ORDER=scatter

# Set up environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source /home/tab53/.bashrc

# Change directory
cd /home/tab53/rds/rds-ukaea-ap002-mOlK9qn0PlQ/tab53/bayesopt_with_invariances/experiments/synthetic

# Run the script
# export JULIA_DEBUG=user
mpirun -np $SLURM_NTASKS julia --threads auto --project experiment_3_ucb_mpi.jl