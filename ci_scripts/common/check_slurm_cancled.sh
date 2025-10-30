#!/bin/bash
set -x

exit_code=$1
jobname=$2

# get jobid
jobid=$(sacct -u $USER --name=$jobname --format=JobID --noheader | tail -n 1)
swatch examine $jobid | head -n 3 |grep 'CANCELLED by'
if [[ $? -eq 0 || $exit_code -ne 0 ]];then
    echo "Please check slurm job status again!"
    exit 1
fi
