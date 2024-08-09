"""
twoloc_sims.py
A simple script to perform fast simulations of two-locus models with selection, 
mutation and recombination
"""

import numpy as np
import numba as nb
import pandas as pd
import argparse

@nb.njit
def random_choice(probability):
    return 1 if np.random.random() < probability else -1

@nb.njit # use numba for just-in-time compilation
def simulate_population(N, mu, r=0.0, run_mult=5, beta_1 = 0.0, beta_2 = 0.0):
    # Setting up the population and non-selection parameters
    pop = np.array([N, 0, 0, 0], dtype=np.int64)
    run = 0
    
    # Setting up selection parameters
    # Assuming symmetric selection here    
    w_sing = np.exp(-beta_1**2)
    w_doub = np.exp(-(beta_1 + beta_2)**2)
    
    probs = np.array([1,0,0,0], dtype=np.float64)
    
    while run < run_mult * N:
        # If we fix one of the alleles restart the simulation
        if pop[3] + pop[1] == N or pop[3] + pop[2] == N:
            pop = np.array([N, 0, 0, 0], dtype=np.int64)
        
        # If population is monomorphic for ancestral haplotype:
        if pop[0] == N:
            # Spike in a mutation at position i or j
            if np.random.rand() < 0.5:
                pop[1] = 1
                pop[0] = N - 1
            else:
                pop[2] = 1
                pop[0] = N - 1

            # sample a geometrically distributed number of generations before that mutation occured
            gens_to_mut = np.random.geometric(2 * N * mu)
            if (run_mult*N - run) < gens_to_mut: # Return a monomorphic population if we overshoot the run
                pop = np.array([N, 0, 0, 0], dtype=np.int64)
                break
            run += gens_to_mut
        else:
            run += 1

        probs[0] = pop[0] - mu*pop[0] + r*(pop[1]*pop[2] - pop[0]*pop[3]) 
        probs[1] = w_sing*(pop[1] + mu*pop[0] + r*(pop[0]*pop[3] - pop[1]*pop[2])) 
        probs[2] = w_sing*(pop[2] + mu*pop[0] + r*(pop[0]*pop[3] - pop[1]*pop[2])) 
        probs[3] = w_doub*(pop[3] + mu*(pop[1] + pop[2]) + r*(pop[1]*pop[2] - pop[0]*pop[3])) 

        # The weird stuff down here is because of numerical instability issues in numba 
        # when some probabilities are zero. This is a hacky way to deal with it, but don't mess with it.
        probs[probs < 0] = 0
        probs /= np.sum(probs)
        pop[probs > 0] = np.random.multinomial(N, probs[probs>0])
        pop[probs == 0] = 0
    
    return pop

@nb.njit
def sim_batch(N_sims=10000,
              N=20000, 
              theta=0.01, 
              rho=0.0, 
              run_mult=5, 
              eff_size_float = 0.0, 
              symm_param = 0.5):
    
    mu = theta / N
    r  = rho / N
    
    beta_arr = []
    pop_arr = []
    
    for i in range(N_sims):
        # Any other choice of betas may be specified here
        beta_1 = eff_size_float * random_choice(symm_param)
        beta_2 = eff_size_float * random_choice(symm_param)
        beta_arr.append([beta_1, beta_2])
        
        pop = simulate_population(N, mu, r, run_mult, beta_1, beta_2)
        pop_arr.append(pop)
        
    return beta_arr, pop_arr

argparser = argparse.ArgumentParser(description='Simulate two-locus models with selection, mutation and recombination')
argparser.add_argument('--N_sims', type=int, default=10000, help='Number of simulations')
argparser.add_argument('--N', type=int, default=20000, help='Population size')
argparser.add_argument('--theta', type=float, default=0.01, help='Mutation rate')
argparser.add_argument('--rho', type=float, default=0.0, help='Recombination rate')
argparser.add_argument('--run_mult', type=int, default=5, help='Factor to multiply population size by for run length')
argparser.add_argument('--eff_size_float', type=float, default=0.0, help='Effect size')
argparser.add_argument('--symm_param', type=float, default=0.5, help='Mutational symmetry parameter')
argparser.add_argument('--csv_tag', type=str, default='', help='tag to put on end of csv')
args = argparser.parse_args()

beta_arr, pop_arr = sim_batch(N_sims=args.N_sims,
                              N=args.N, 
                              theta=args.theta, 
                              rho=args.rho, 
                              run_mult=args.run_mult, 
                              eff_size_float=args.eff_size_float, 
                              symm_param=args.symm_param)

print(args)

# Turn the betas and pops into dataframes and save them
pd.DataFrame(beta_arr).to_csv(f'dataframes/beta_arr_{args.csv_tag}.csv', sep='\t')
pd.DataFrame(pop_arr).to_csv(f'dataframes/pop_arr_{args.csv_tag}.csv', sep='\t')

print("#beta_1\tbeta_2\tN_00\tN_01\tN_10\tN_11")
for i in range(args.N_sims):
    print(f'{beta_arr[i][0]}\t{beta_arr[i][1]}\t{pop_arr[i][0]}\t{pop_arr[i][1]}\t{pop_arr[i][2]}\t{pop_arr[i][3]}')
