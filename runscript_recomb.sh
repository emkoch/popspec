#!/bin/bash
#SBATCH -c 1               # Number of cores (-c)
#SBATCH -t 1-22:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p medium            # Partition to submit to
#SBATCH --mem=1200M          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

## This runscript generates simulations with selection using the finger grained
# beta values. The main difference is that it simulates the out of Africa bottleneck
# and thus uses a varying population size

module load gcc/9.2.0
module load gsl/2.7.1

readarray beta_arr < betas.txt
arr_len=${#beta_arr[@]}
recomb_rates=(0.1 1 10 100 1000)

for ((i=0; i<$arr_len; i++)); 
do
    for ((j=0; j<5; j++))
    do
        # For polygenic selection
        ./sims_selec 10000 20000 ${recomb_rates[$j]} 0.0 10 ${beta_arr[$i]} ${beta_arr[$i]} 0.5 $SLURM_ARRAY_TASK_ID 1 0 | gzip > /n/scratch/users/s/sjg319/sim_outputs/recomb/${recomb_rates[$j]}/underdom/${i}_${SLURM_ARRAY_TASK_ID}.txt.gz
        # This one is for non-underdom selection
        # ./sims_selec 10000 20000 ${recomb_rates[$j]} 0.0 10 ${beta_arr[$i]} ${beta_arr[$i]} 0.5 $SLURM_ARRAY_TASK_ID 0 0 | gzip > /n/scratch/users/s/sjg319/sim_outputs/recomb/${recomb_rates[$j]}/negative/old_${i}_${SLURM_ARRAY_TASK_ID}.txt.gz
    done
done

# The typical line to run this from o2 will be the following: 
# sbatch --array=1-1000 runscript_recomb.sh
