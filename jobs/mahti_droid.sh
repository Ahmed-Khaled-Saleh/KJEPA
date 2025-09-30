#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=vitg16_train
#SBATCH --partition=gpusmall
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=320G
#SBATCH --gres=gpu:a100:2                     
#SBATCH --output=./logs/err_%j_%x_%N.out
#SBATCH --error=./logs/err_%j_%x_%N.err




# Load required modules
module --force purge
module load pytorch

# Activate python environment
source /projappl/project_2009050/mytorch/bin/activate
cd /projappl/project_2009050/KJEPA
# git fetch
# git pull

# Launch with torchrun
srun torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=2  \
  -m app.main \
  --fname configs/train/vitg16/droid-256px-8f.yaml
