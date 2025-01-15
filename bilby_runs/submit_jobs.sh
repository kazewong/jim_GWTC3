#!/bin/bash

# Read the event IDs from the CSV file
csv_file="/home/thomas.ng/project/jim_GWTC3/bilby_runs/event_status.csv"
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

  # Update the status to "submitted" in the CSV file using flock
  {
    flock -x 200
    sed -i "/^$gw_id,/s/,.*/,$gw_id,submitted,/" $csv_file
  } 200>$csv_file.lock
  
  # Submit the job and wait for it to finish
  bilby_pipe /home/thomas.ng/project/jim_GWTC3/bilby_runs/config/${gw_id}_config.ini --submit
  wait
  
  echo "Submitted job for $gw_id"
done