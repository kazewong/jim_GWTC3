#!/bin/bash
#Set job requirements
#SBATCH --gpus=1
#SBATCH --job-name={{{GW_ID}}}
#SBATCH --output=/home/user/ckng/project/jim_GWTC3/jim_runs/logs/%x.out
#SBATCH --error=/home/user/ckng/project/jim_GWTC3/jim_runs/logs/%x.err

# Define dirs
export GW_ID={{{GW_ID}}}

# Define the CSV file path
csv_file="/home/user/ckng/project/jim_GWTC3/jim_runs/event_status.csv"

# Update the status to "running" in the CSV file using flock
{
  flock -x 200
  sed -i "/^$GW_ID,/s/,.*/,$GW_ID,running,/" $csv_file
} 200>$csv_file.lock

# Initialize Conda
source /home/user/ckng/.bashrc
conda activate /home/user/ckng/.conda/envs/jim

# Check the GPU model
nvidia-smi

# Run the script
python /home/user/ckng/project/jim_GWTC3/jim_runs/run.py --event-id $GW_ID --outdir /home/user/ckng/project/jim_GWTC3/jim_runs/outdir

# Update the status to "done" in the CSV file using flock
{
  flock -x 200
  sed -i "/^$GW_ID,/s/,.*/,$GW_ID,done,/" $csv_file
} 200>$csv_file.lock

echo "DONE"
