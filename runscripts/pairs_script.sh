#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-18:00
#SBATCH -p medium
#SBATCH --mem=4G
#SBATCH -o outs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load gcc/6.2.0

python pairs_script.py --num_chunks=20

# sbatch --array=0-19 pairs_script.sh