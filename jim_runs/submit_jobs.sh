#!/bin/bash

# Define the path to the template script
template_file="template.sh"

# Create directories to store the SLURM scripts and logs
mkdir -p slurm_scripts
mkdir -p logs

# Read the event IDs from the CSV file
csv_file="/home/user/ckng/project/jim_GWTC3/event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Define the result directory path
  result_dir="/home/user/ckng/project/jim_GWTC3/jim_runs/outdir/${gw_id}"
  
  # Check if the result directory contains any files
  if [ -d "$result_dir" ] && [ "$(find "$result_dir" -type f | wc -l)" -gt 0 ]; then
    echo "Result already exists for $gw_id, skipping submission."
    continue
  fi
  
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${gw_id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{GW_ID}}}/$gw_id/g" $new_script

  # Make the script executable
  chmod +x $new_script
  
  # Submit the job to SLURM
  sbatch $new_script
  
  echo "Submitted job for $gw_id"
done
