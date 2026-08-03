#!/bin/bash -l

# This is were we go for slurm documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --job-name=MHANet_KUL_test_20260802_ws01              # Name of the job
#SBATCH --comment="MHANet test"           # Comment for your job
#SBATCH --account=spatt


# %x is job name and %j is job id
#SBATCH --output=./log/%x_%j/%x_%j.out                      # Output file
#SBATCH --error=./log/%x_%j/%x_%j.err                        # Error file

#SBATCH --mail-user=slack:@at2163                # Slack username to notify
#SBATCH --mail-type=END                         # Type of slack notifications to send

#SBATCH --time=1-00:00:00                        # Time limit day-hour:minutes:seconds
#SBATCH --nodes=1                               # How many nodes to run on
#SBATCH --partition=tier3                       # Partitions {debug, tier3} 
#SBATCH --mem=64G                                # Job memory
#SBATCH --ntasks=1                              # How many tasks per node
#SBATCH --cpus-per-task=18                      # Number of CPUs per task
#SBATCH --gres=gpu:a100:1                            # Number of GPUs per task
##SBATCH --mem-per-cpu=64G                      # Memory per CPU (this line is here as an example but commented out) 

set -euo pipefail

echo "========== Job Information =========="
echo "Hostname : $(hostname)"
echo "Date     : $(date)"
echo "GPU(s):"
nvidia-smi

echo "========== Python Environment =========="
python --version
uv --version

# # (Optional) Load modules if required by SPORC
# module load cuda/12.4
# module load gcc/...

# Change to the project directory
cd "../"

# Ensure the log directory exists
mkdir -p logs

echo "========== Running =========="

uv run python main_akira.py --dataset KUL --time_len 0.1 --batch_size 

echo "========== Finished =========="
