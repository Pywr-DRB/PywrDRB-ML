#!/bin/bash
#SBATCH --job-name=cvali               # Job name
#SBATCH --output=/home/fs01/cl2769/Github/PywrDRB-ML/logs/cvali_%j.out   # Standard output log file with job ID
#SBATCH --error=/home/fs01/cl2769/Github/PywrDRB-ML/logs/cvali_%j.err    # Standard error log file with job ID
#SBATCH --nodes=2                           # Number of nodes to use
#SBATCH --ntasks-per-node=40                # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job
#SBATCH --exclude=c0004                      # Exclude node 0004
#SBATCH --mail-type=END                    # Send email at job end
#SBATCH --mail-user=cl2769@cornell.edu     # Email for notifications

# Remember to create ./logs/ first!

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source ~/VEnvs/drb/bin/activate

# Set environment variables for reproducible parallel execution
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "Start JobID $SLURM_JOB_ID"
#time mpirun -np $n_processors python /home/fs01/cl2769/Github/PywrDRB-ML/TempLSTM_cross_validation.py 
time mpirun -np $n_processors python /home/fs01/cl2769/Github/PywrDRB-ML/lstm_cross_validation.py 
#wait
echo "Complete"