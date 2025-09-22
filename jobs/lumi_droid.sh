#!/bin/bash
#SBATCH --account=project_462000702
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=8

module purge
module load LUMI/24.03
module load PyTorch/2.3.1-rocm-6.0.3-python-3.12-Fooocus-singularity-20240923

# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
# export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
# mkdir -p $HF_HOME

# # set pytorch to use the cache folder
# export TORCH_HOME=$HF_HOME/torch
# mkdir -p $TORCH_HOME

cd /projappl/project_462000702/KJEPA

# # Path to where the trained model and logging data will go
# OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-data
# mkdir -p $OUTPUT_DIR

# # Disable internal parallelism of huggingface's tokenizer since we
# # want to retain direct control of parallelism options.
# export TOKENIZERS_PARALLELISM=false


# set -xv  # print the command so that we can verify setting arguments correctly from the logs

# All commands must follow the #SBATCH directives
export CONTAINER=/project/project_462000702/EasyBuild/SW/container/PyTorch/2.3.1-rocm-6.0.3-python-3.12-Fooocus-singularity-20240923/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1-dockerhash-2c1c14cafd28.sif

# Launch MPI code 
srun singularity exec $CONTAINER torchrun --standalone \
     --nnodes=1 \
     --nproc_per_node=$SLURM_GPUS_PER_NODE \
     app/main.py $* \
     --fname configs/train/vitg16/droid-256px-8f.yaml






# srun torchrun --standalone \
#      --nnodes=1 \
#      --nproc-per-node=$SLURM_GPUS_PER_NODE \
#      finetuning.py $* \
#      --output-path $OUTPUT_DIR \
#      --num-workers 7