#!/bin/bash

# Define the path to the template script
template_file="template.sh"

# Create directories to store the SLURM scripts and logs
mkdir -p slurm_scripts
mkdir -p logs

# Read the event IDs from the CSV file
csv_file="/home/user/ckng/project/jim_GWTC3/jim_runs/event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${gw_id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{GW_ID}}}/$gw_id/g" $new_script
  
  # Update the status to "submitted" in the CSV file using flock
  {
    flock -x 200
    sed -i "/^$gw_id,/s/,.*/,$gw_id,submitted,/" $csv_file
  } 200>$csv_file.lock
  
  # Make the script executable
  chmod +x $new_script
  
  # Submit the job to SLURM
  sbatch $new_script
  
  echo "Submitted job for $gw_id"
  
  # Add a short delay between submissions
  sleep 0.5
done
