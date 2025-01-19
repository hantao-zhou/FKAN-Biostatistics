#!/bin/bash -e 
#SBATCH --time=0-48:00:00
#SBATCH --output=slurm/KAN-output-%x-%j.log
#SBATCH --error=slurm/KAN-errors-%x-%j.log
#SBATCH --partition=topfgpu
#SBATCH --nodes=1                        
#SBATCH --constraint="A100"
#SBATCH --open-mode append


# Load Micromamba environment
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /gpfs/cssb/user/alsaadiy/micromamba/envs/tensorflow2.10

# 1. Initialize the module system
source /etc/profile.d/modules.sh  

# 2. Load CUDA module
module load cuda/11.8


# Set the project path
PROJECT_PATH="/gpfs/cssb/user/alsaadiy/FKAN-Biostatistics/KAN_Model_Py"
cd "$PROJECT_PATH"

# Run the training script
python KAN_Fourier.py