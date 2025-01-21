#!/bin/bash

# Activate the conda environment
conda activate /home/sylvia.biscoveanu/.conda/envs/bilby_o4review_230314/

# Read the event IDs from the CSV file
csv_file="/home/thomas.ng/project/jim_GWTC3/event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Define the result directory path
  result_dir="/home/thomas.ng/project/jim_GWTC3/bilby_runs/outdir"
  
  # Check if the result directory contains a subdirectory named as the GW ID
  if [ -d "$result_dir/$gw_id" ]; then
    echo "Result already exists for $gw_id, skipping submission."
    sleep 0.5
    continue
  fi

  # Submit the job and wait for it to finish
  bilby_pipe /home/thomas.ng/project/jim_GWTC3/bilby_runs/config/${gw_id}_config.ini --submit
  wait
  
  echo "Submitted job for $gw_id"

  # Deactivate the conda environment
  conda deactivate
done
