#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-n] (use -n to enable node preference for a/c nodes)"
    exit 1
}

# Default to no node preference
USE_NODE_PREFERENCE=false

# Parse command line options
while getopts "n" opt; do
    case $opt in
        n) USE_NODE_PREFERENCE=true ;;
        ?) usage ;;
    esac
done

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
    continue
  fi
  
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${gw_id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{GW_ID}}}/$gw_id/g" $new_script

  # Make the script executable
  chmod +x $new_script

  if [ "$USE_NODE_PREFERENCE" = true ]; then
    # Find available a or c node
    AVAILABLE_NODE=$(sinfo -h -t idle -o "%n" | grep -E '^(a|c)' | head -n1)
    
    # Submit the job to SLURM with node preference
    if [ -n "$AVAILABLE_NODE" ]; then
      sbatch --nodelist=$AVAILABLE_NODE $new_script
      echo "Submitted job for $gw_id on node $AVAILABLE_NODE"
    else
      sbatch $new_script
      echo "Submitted job for $gw_id on any available node (no a/c nodes available)"
    fi

    # Wait for 5 seconds before submitting the next job
    sleep 5
  else
    # Submit without node preference
    sbatch $new_script
    echo "Submitted job for $gw_id without node preference"
  fi
done
