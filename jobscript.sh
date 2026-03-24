#!/bin/bash -l
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000
#SBATCH --account=torch_pr_292_courant
#SBATCH --time=00:30:00
#SBATCH --array=0-7              # 8 parallel tasks (task IDs 0..7)
#SBATCH --output=../logs/slurm_%A_%a.out   # stdout log (%A=job id, %a=array index)
#SBATCH --error=../logs/slurm_%A_%a.err    # stderr log

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Load modules and activate conda environment
module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate desc


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# Run QUADCOIL on configs
# SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT are set automatically by
# SLURM when using --array; the script reads them to select its data shard.
python ./download_constellaration.py \
    --device_scaling "stellaris" \
    --num_config 500 \
    --nfp 3 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks $SLURM_ARRAY_TASK_COUNT

echo "End time: $(date)"
echo "Job completed!"