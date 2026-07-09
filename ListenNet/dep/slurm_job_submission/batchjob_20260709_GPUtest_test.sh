#!/bin/bash -l

# This is were we go for slurm documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --job-name=ListenNet_test_20260709              # Name of the job
#SBATCH --comment="ListenNet test"           # Comment for your job
#SBATCH --account=spatt


# %x is job name and %j is job id
#SBATCH --output=./log/%x_%j/%x_%j.out                      # Output file
#SBATCH --error=./log/%x_%j/%x_%j.err                        # Error file

#SBATCH --mail-user=slack:@at2163                # Slack username to notify
#SBATCH --mail-type=END                         # Type of slack notifications to send

#SBATCH --time=0-16:00:00                        # Time limit day-hour:minutes:seconds
#SBATCH --nodes=1                               # How many nodes to run on
#SBATCH --partition=debug                       # Partitions {debug, tier3} 
#SBATCH --mem=32G                                # Job memory
#SBATCH --ntasks=1                              # How many tasks per node
#SBATCH --cpus-per-task=18                      # Number of CPUs per task
#SBATCH --gres=gpu:a100:1                            # Number of CPUs per task
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

uv run python - <<'EOF'
import os
import torch
import subprocess

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))

try:
    print(subprocess.check_output(["nvidia-smi"]).decode()[:300])
except Exception as e:
    print(e)

print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.version.cuda =", torch.version.cuda)
EOF

echo "========== Finished =========="
