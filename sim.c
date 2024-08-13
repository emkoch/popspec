// Compile on O2: gcc -I/n/app/gsl/2.7.1-gcc-9.2.0/include/ -L/n/app/gsl/2.7.1-gcc-9.2.0/lib/ sim.c -lgsl -lgslcblas -std=c99 -lm -o sims_selec
// Compile locally: gcc -I/n/app/gsl/2.7.1-gcc-9.2.0/include/ -L/n/app/gsl/2.7.1-gcc-9.2.0/include/gsl sim.c -lgsl -lgslcblas -lm -o sims
// Run: ./sims_selec 10 2000 0.01 0.0 10 0.02 0.02 0.5 1 1 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <stdbool.h>

#define INITIAL_SIZE 10  // Initial size of the array

// Function to simulate a random choice based on probability
int random_choice(double probability) {
//    printf("random choice");
    return ((double)rand() / RAND_MAX) < probability ? 1 : -1;
}

void singleGeneration(int N, double beta_1, double beta_2, double rec_val, double mu, unsigned int *pop, int underdominant, gsl_rng * r){
    double ww[4] = {0.0, 0.0, 0.0, 0.0};    // Fitnesses of haplotypes
    double probs[4] = {1.0, 0.0, 0.0, 0.0};
    double w_sing_1 = exp(-pow(beta_1, 2));
    double w_sing_2 = exp(-pow(beta_2, 2));
    double w_doub = exp(-pow(beta_1 + beta_2, 2));
    
    double deviation = 0.0;
    double total = 0.0;

    if (underdominant == 1) {
        // This is where the underdominant fitness effect comes in
        deviation = (beta_1 * (pop[1] + pop[3]) + beta_2 * (pop[2] + pop[3])) / N;
        ww[0] = 1 - pow(deviation, 2);
        ww[1] = 1 - pow(beta_1 - deviation, 2);
        ww[2] = 1 - pow(beta_2 - deviation, 2);
        ww[3] = 1 - pow(beta_1 + beta_2 - deviation, 2);

        probs[0] = ww[0] * (pop[0] - mu * pop[0] + rec_val * (pop[1] * pop[2] - pop[0] * pop[3]));
        probs[1] = ww[1] * (pop[1] + mu * pop[0] + rec_val * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[2] = ww[2] * (pop[2] + mu * pop[0] + rec_val * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[3] = ww[3] * (pop[3] + mu * (pop[1] + pop[2]) + rec_val * (pop[1] * pop[2] - pop[0] * pop[3]));

    } else {
        // This is for the standard quadratic fitness function
        probs[0] = pop[0] - mu * pop[0] + rec_valr * (pop[1] * pop[2] - pop[0] * pop[3]);
        probs[1] = w_sing_1 * (pop[1] + mu * pop[0] + rec_val * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[2] = w_sing_2 * (pop[2] + mu * pop[0] + rec_val * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[3] = w_doub * (pop[3] + mu * (pop[1] + pop[2]) + rec_val * (pop[1] * pop[2] - pop[0] * pop[3]));
    }

    // Normalize probabilities
    total = probs[0] + probs[1] + probs[2] + probs[3];
    probs[0] /= total;
    probs[1] /= total;
    probs[2] /= total;
    probs[3] /= total;

    // Sample from multinomial distribution
    gsl_ran_multinomial(r, 4, N, probs, pop);
}

// Read population sizes from a file
// Used in the case of sims with demography
int *read_integers_from_file(const char *filename, int *n) {
    // filename: name of the file to read, frovided by the user
    // n: pointer to the number of integers read from the file
    FILE *file = fopen(filename, "r");
    int *arr = malloc(INITIAL_SIZE * sizeof(int));

    size_t size = INITIAL_SIZE;
    
    *n = 0;
    while (fscanf(file, "%d", &arr[*n]) == 1) {
        (*n)++;
        if (*n >= size) {
            size_t new_size = size * 2;
            int *new_array = realloc(arr, new_size * sizeof(int));
            arr = new_array;
            size = new_size;
        }
    }

    fclose(file);

    return arr;
}

// Function to simulate the population
void simulate_population(unsigned int N, 
                         double theta, 
                         double rho, 
                         int run_mult, 
                         double beta_1, 
                         double beta_2, 
                         unsigned int *pop, 
                         int sim_run, 
                         int slurm_seed, 
                         int underdominant, 
                         int demography) {
    
    //// Initialize values
    // Random seed for gsl
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    double mySeed = (double)rand() + (double)sim_run + (double)slurm_seed;
    gsl_rng_set(r, mySeed);

    int run = 0;
    double w_sing_1 = exp(-pow(beta_1, 2));
    double w_sing_2 = exp(-pow(beta_2, 2));
    double w_doub = exp(-pow(beta_1 + beta_2, 2));
    
    double deviation = 0.0;
    double total = 0.0;
    double mu = theta / (2 * N); // theta = 2*Ne*mu in a haploid population
    double rec_val = rho / (N * N);

    // Read in the population
    if (demography == 1){
        int n = 0;
        int *population = NULL;
        population = read_integers_from_file("demo_Ns.txt", &n);
    }
    ////

    // We start with the burn in period
    while (run < run_mult * N) {
        // If population is monomorphic for a derived allele
        if (pop[1] + pop[3] == N) {
            pop[0] = N - pop[3];
            pop[1] = 0;
            pop[2] = pop[3];
            pop[3] = 0;
        } else if (pop[2] + pop[3] == N) {
            pop[0] = N - pop[3];
            pop[1] = pop[3];
            pop[2] = 0;
            pop[3] = 0;
        }

        // If population is monomorphic for ancestral haplotype
        if (pop[0] == N) {
            // Spike in a mutation at position i or j
            if ((double)rand() / RAND_MAX < 0.5) {
                pop[1] = 1;
                pop[0] = N - 1;
            } else {
                pop[2] = 1;
                pop[0] = N - 1;
            }

            // Sample a geometrically distributed number of generations before that mutation occurred
            int gens_to_mut = 1 + gsl_ran_geometric(r, theta);
            if ((run_mult * N - run) < gens_to_mut) {
                pop[0] = N;
                pop[1] = 0;
                pop[2] = 0;
                pop[3] = 0;
                break;
            }
            run += gens_to_mut;
        } else {
            run += 1;
        }

        singleGeneration(N, beta_1, beta_2, rec_val, mu, pop, underdominant, r);
    }

    /////
    if (demography == 1){
        // And now we simulate with the demography
        for (size_t i = 0; i < n; i++){
            // If population is monomorphic for a derived allele
            if (pop[1] + pop[3] == population[i]) {
                pop[0] = population[i] - pop[3];
                pop[1] = 0;
                pop[2] = pop[3];
                pop[3] = 0;
            } else if (pop[2] + pop[3] == population[i]) {
                pop[0] = population[i] - pop[3];
                pop[1] = pop[3];
                pop[2] = 0;
                pop[3] = 0;
            }

        singleGeneration(population[i], beta_1, beta_2, rec_val, mu, pop, underdominant, r);
        }
    }
    /////

    gsl_rng_free(r);
}

// Function to simulate batch of populations
void sim_batch(int N_sims, 
               unsigned int N, 
               double theta, 
               double rho, 
               int run_mult, 
               float eff_size_float_1,
               float eff_size_float_2, 
               double symm_param, 
               int slurm_seed, 
               int underdominant, 
               int demography) {

    double beta_1, beta_2;
    double beta_arr[N_sims][2];
    int pop_arr[N_sims][4];

    srand(time(NULL));

    for (int i = 0; i < N_sims; ++i) {
                
        // Randomly choose beta values
        beta_1 = eff_size_float_1 * random_choice(symm_param);
        beta_2 = eff_size_float_2 * random_choice(symm_param);
        beta_arr[i][0] = beta_1;
        beta_arr[i][1] = beta_2;
        
        // Simulate population
        unsigned int pop[4] = {N, 0, 0, 0};

        simulate_population(N, theta, rho, run_mult, beta_1, beta_2, pop, i, slurm_seed, underdominant, demography);

        // Store population results
        for (int j = 0; j < 4; ++j) {
            pop_arr[i][j] = pop[j];
        }
    }

    // Output results (example: printing beta values and final populations)
    printf("beta_1\tbeta_2\tN00\tN01\tN10\tN11\n");
    for (int i = 0; i < N_sims; ++i) {
        printf("%.6f\t%.6f\t%d\t%d\t%d\t%d\n", beta_arr[i][0], beta_arr[i][1], pop_arr[i][0], pop_arr[i][1], pop_arr[i][2], pop_arr[i][3]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 12) {
        fprintf(stderr, "Usage: %s N_sims N_popsize rho run_mult eff_size symm_param\n", argv[0]);
        return 1;
    }

    int N_sims = atoi(argv[1]);
    unsigned int N = atoi(argv[2]);
    double theta = atof(argv[3]);
    double rho = atof(argv[4]);
    int run_mult = atoi(argv[5]);
    float eff_size_float_1 = atof(argv[6]);
    float eff_size_float_2 = atof(argv[7]);
    double symm_param = atof(argv[8]);
    double slurm_seed = atof(argv[9]);
    int underdominant = atoi(argv[10]);
    int demography = atoi(argv[11])

    fprintf(stderr, "starting\n");

    clock_t start = clock(); // Start the timer

    // 10000 20000 0.01 0.0 5 0.0 0.5
    sim_batch(N_sims, N, theta, rho, run_mult, eff_size_float_1, eff_size_float_2, symm_param, slurm_seed, underdominant, demography);

    clock_t end = clock(); // Stop the timer

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC; // Calculate the elapsed time

    fprintf(stderr, "Execution time: %.2f seconds\n", time_taken);

    return 0;
}
