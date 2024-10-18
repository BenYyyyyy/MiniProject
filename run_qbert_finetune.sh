#!/bin/bash -l
#SBATCH --job-name=qbert_pretrain   # Job name
#SBATCH --output=project_/logs/examplejob.o%j # Name of stdout output file
#SBATCH --error=project_/logs/examplejob.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_  # Project for billing

export WANDB_API_KEY=''
export HYDRA_FULL_ERROR=1
source project_465000986/miniconda3/etc/profile.d/conda.sh
cd project_465000986/MiniProject/quantum_model
conda activate qbert
srun python src/finetune.py experiment=finetune_tweethate1_clean-33.yaml
#srun python src/classical_finetune.py experiment=classical_finetune_wnli.yaml 
