#!/bin/bash
#SBATCH -J Ex3_MVR_MPI_v1.0
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p sapphire
#SBATCH --output=%x_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=128     # 32 repeats x 4 kernels
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

# Set up hardware
export OMP_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=omp:compact
export I_MPI_PIN_ORDER=scatter

# Set up environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source /home/tab53/.bashrc

# Change directory
cd /home/tab53/rds/rds-ukaea-ap002-mOlK9qn0PlQ/tab53/bayesopt_with_invariances/experiments

# Run the script
# export JULIA_DEBUG=user
mpirun -np $SLURM_NTASKS julia --threads auto --project experiment_3_mvr_mpi.jl