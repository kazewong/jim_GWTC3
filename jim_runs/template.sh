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

# Check if the script produced any results
if [ "$(find /home/user/ckng/project/jim_GWTC3/jim_runs/outdir/$GW_ID -type f | wc -l)" -gt 0 ]; then
  echo "Job completed successfully for $GW_ID"
else
  echo "Job failed for $GW_ID"
  # Resubmit the job with specific node list
  sbatch --nodelist=a[1-10],c[1-10] /home/user/ckng/project/jim_GWTC3/jim_runs/slurm_scripts/submit_${GW_ID}.sh
fi
