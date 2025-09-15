#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=vitg16_train
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8        
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=320G                     
#SBATCH --output=./logs/err_%j_%x_%N.out
#SBATCH --error=./logs/err_%j_%x_%N.err




# Load required modules
module --force purge
module load pytorch

# Activate python environment
source /scratch/project_2009050/venvs/torchy/bin/activate
cd /projappl/project_2009050/KJEPA

# Launch with torchrun
srun torchrun \
  --nnodes=1 \
  --nproc_per_node=2  \
  app/main.py \
  --fname configs/train/vitg16/droid-256px-8f.yaml
