#!/bin/bash
#SBATCH -c 1               # Number of cores (-c)
#SBATCH -t 0-5:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p short           # Partition to submit to
#SBATCH --mem=3000M           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# # run code
# beta_arr=("0.00223607" "0.00423899" "0.00803601" "0.01523415" "0.02887993" "0.05474871" "0.10378908" "0.19675666" "0.37299862" "0.70710678")


# module load gcc/9.2.0
# module load gsl/2.7.1

# step_val=( 1 3 5 10 15 )

# for i in ${step_val[@]}; do
#     ./sims_selec 100000 20000 0.01 0.0 $i 0.0 0.0 0.5 $SLURM_ARRAY_TASK_ID | gzip > sim_outputs/sim_outputs_${i}_${SLURM_ARRAY_TASK_ID}.txt.gz
#     # ./sims_selec 10 20000 0.01 0.0 $i 0.0 0.0 0.5 2
# done

python concat.py
