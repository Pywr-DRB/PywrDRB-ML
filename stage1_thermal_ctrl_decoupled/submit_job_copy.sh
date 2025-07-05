#!/bin/bash
#SBATCH --job-name=stage1               # Job name
#SBATCH --output=/home/fs01/cl2769/Github/PywrDRB-ML/logs/stage1_%j.out   # Standard output log file with job ID
#SBATCH --error=/home/fs01/cl2769/Github/PywrDRB-ML/logs/stage1_%j.err    # Standard error log file with job ID
#SBATCH --nodes=2                           # Number of nodes to use
#SBATCH --ntasks-per-node=40                # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job
#SBATCH --mail-type=END                    # Send email at job end
#SBATCH --mail-user=cl2769@cornell.edu     # Email for notifications

# Remember to create ./logs/ first!

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source ~/VEnvs/drb/bin/activate

# Set environment variables for reproducible parallel execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Define arrays for policy types and borg seeds
policy_types=("cubic_rbf")  # Add your policy types here
borg_seeds=(1 2 3)  # Add your borg seeds here

# Function to submit the job
submit_job() {
    local policy_type=$1
    local borg_seed=$2
    
    # Print start message and the number of nodes and tasks per node
    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "[JobID $SLURM_JOB_ID] Running thermal control optimization ..."
    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total number of processors: $n_processors"
    echo "Datetime: $datetime"
    echo "Policy type: $policy_type"
    echo "Borg seed: $borg_seed"

    # Run the script with MPI and time the execution
    echo "Start JobID $SLURM_JOB_ID, policy: $policy_type, seed: $borg_seed"
    time mpirun -np $n_processors python /home/fs01/cl2769/Github/PywrDRB-ML/stage1_thermal_ctrl_decoupled/borg_dps_stage1.py $SLURM_JOB_ID $policy_type $borg_seed
    #wait
    echo "Complete: $policy_type, seed: $borg_seed"
}

# Loop over policy types and borg seeds to submit jobs
for borg_seed in "${borg_seeds[@]}"; do
    for policy_type in "${policy_types[@]}"; do
        submit_job $policy_type $borg_seed
    done
done

echo "All optimization jobs completed!"