#!/bin/bash
#Set job requirements
#SBATCH --gpus=1
#SBATCH --job-name={{{GW_ID}}}
#SBATCH --output=/home/user/ckng/project/jim_GWTC3/jim_runs/logs/%x.out
#SBATCH --error=/home/user/ckng/project/jim_GWTC3/jim_runs/logs/%x.err

# Define dirs
export GW_ID={{{GW_ID}}}

# Initialize Conda
source /home/user/ckng/.bashrc
conda activate /home/user/ckng/.conda/envs/jim

# Check the GPU model
nvidia-smi

# Run the script
python /home/user/ckng/project/jim_GWTC3/jim_runs/run.py --event-id $GW_ID --outdir /home/user/ckng/project/jim_GWTC3/jim_runs/outdir
