#!/bin/bash
#SBATCH -c 1               # Number of cores (-c)
#SBATCH -t 0-6:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p short            # Partition to submit to
#SBATCH --mem=3M          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load gcc/9.2.0
module load gsl/2.7.1

readarray beta_arr < betas.txt
arr_len=${#beta_arr[@]}

for ((i=0; i<$arr_len; i++)); 
do
    echo ${beta_arr[$i]} $i
    ./sims_selec 20000 20000 0.01 0.0 15 ${beta_arr[$i]} ${beta_arr[$i]} 0.5 $SLURM_ARRAY_TASK_ID | gzip > sim_outputs/selection/single/sim_outs_single_${i}_${SLURM_ARRAY_TASK_ID}.txt.gz
done
