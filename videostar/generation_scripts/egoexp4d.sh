#!/bin/bash

set -e
cur_fname="$(basename $0 .sh)"

# Cluster parameters
partition="pasteur"
account="pasteur"

num_chunks=85

for ((i=0; i<num_chunks; i++)); do
    # Construct the command to run
    cmd="python videostar/generation_scripts/egoexp4d_convert_mp4.py --chunk_index $i"
    echo "Constructed Command:\n$cmd"
    # Uncomment below to submit the job
    sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=$i-convert
#SBATCH --output=slurm_logs/$i-convert-%j-out.txt
#SBATCH --error=slurm_logs/$i-convert-%j-err.txt
#SBATCH --mem=9gb
#SBATCH -c 4
#SBATCH -p $partition
#SBATCH -A $account
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
echo \"$cmd\"
# Uncomment below to actually run the command
eval \"$cmd\"
"
done