/**
 * @file sim.c
 * @brief Wright-Fisher simulation with recombination, selection (underdominant/quadratic), and demography.
 *
 * Compilation on O2 Cluster:
 * gcc -I/n/app/gsl/2.8-gcc-14.2.0/include -O3 -std=c99 -o wf_sim sim.c -L/n/app/gsl/2.8-gcc-14.2.0/lib -lgsl -lgslcblas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // REQUIRED for strcmp
#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>

/* --- CONSTANTS & MACROS --- */

// Haplotype indices for clarity
// Assuming 2 loci with alleles A/a and B/b
// 0: Ancestral (AB), 1: Derived A (aB), 2: Derived B (Ab), 3: Double Derived (ab)
enum Haplotype {
    HAP_ANC = 0,
    HAP_MUT_1 = 1,
    HAP_MUT_2 = 2,
    HAP_DOUBLE = 3,
    NUM_HAPS = 4
};

#define INITIAL_VEC_SIZE 100

/* --- STRUCTS --- */

/**
 * @brief Holds parameters for the simulation run.
 * Grouping arguments makes function signatures cleaner.
 */
typedef struct {
    unsigned int N;         // Population size
    double theta;           // Mutation rate param (2*Ne*mu)
    double rho;             // Recombination rate param
    int run_mult;           // Multiplier for generation run time
    double beta_1;          // Selection coefficient 1
    double beta_2;          // Selection coefficient 2
    int polygenic;          // Flag: 1 for underdominant, 0 for quadratic
    int demography;         // Flag: 1 to use demographic file
} SimParams;

/* --- HELPER FUNCTIONS --- */

/**
 * @brief Replaces the custom random_choice function.
 * Uses GSL for uniform random number generation.
 * @return 1 or -1
 */
int get_random_direction(gsl_rng *r, double probability) {
    // gsl_rng_uniform returns [0, 1)
    return (gsl_rng_uniform(r) < probability) ? 1 : -1;
}

/**
 * @brief Reads population sizes from a file.
 * Returns a dynamically allocated array. User must free result.
 */
int *load_demography_file(const char *filename, int *count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open demography file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    size_t capacity = INITIAL_VEC_SIZE;
    int *arr = malloc(capacity * sizeof(int));
    if (!arr) { perror("Malloc failed"); exit(EXIT_FAILURE); }

    *count = 0;
    while (fscanf(file, "%d", &arr[*count]) == 1) {
        (*count)++;
        // CAST FIX: Cast *count to size_t to match capacity
        if ((size_t)(*count) >= capacity) {
            capacity *= 2;
            int *temp = realloc(arr, capacity * sizeof(int));
            if (!temp) { perror("Realloc failed"); free(arr); exit(EXIT_FAILURE); }
            arr = temp;
        }
    }
    fclose(file);
    return arr;
}

/**
 * @brief Calculates fitness vector based on selection model.
 */
void calculate_fitness_probs(unsigned int *pop, double *probs, int N, double beta_1, double beta_2, double mu, double rec_val, int polygenic) {
    // Calculate linkage disequilibrium factor D
    // D = (x_0 * x_3) - (x_1 * x_2)
    // Note: pop is counts, so D is scaled by N^2 relative to frequency D
    double D = ((double)pop[HAP_ANC] * pop[HAP_DOUBLE]) - ((double)pop[HAP_MUT_1] * pop[HAP_MUT_2]);
    
    // Scale recombination by D
    double rec_effect = rec_val * D;
    double N_dbl = (double)N;

    double ww[NUM_HAPS] = {0}; 

    if (polygenic == 1) {
        // Underdominant fitness effect
        // Calculate deviation from mean phenotype
        double deviation = (beta_1 * (pop[HAP_MUT_1] + pop[HAP_DOUBLE]) + 
                            beta_2 * (pop[HAP_MUT_2] + pop[HAP_DOUBLE])) / N_dbl;

        // Gaussian fitness function centered at optimal 0 or 1? 
        // Logic preserved from original: Fitness decreases with distance from optimum
        ww[HAP_ANC]    = 1.0 - pow(deviation, 2);
        ww[HAP_MUT_1]  = 1.0 - pow(beta_1 - deviation, 2);
        ww[HAP_MUT_2]  = 1.0 - pow(beta_2 - deviation, 2);
        ww[HAP_DOUBLE] = 1.0 - pow(beta_1 + beta_2 - deviation, 2);

        // Apply mutation and recombination
        probs[HAP_ANC]    = ww[HAP_ANC]    * (pop[HAP_ANC] - mu * pop[HAP_ANC] - rec_effect);
        probs[HAP_MUT_1]  = ww[HAP_MUT_1]  * (pop[HAP_MUT_1] + mu * pop[HAP_ANC] + rec_effect);
        probs[HAP_MUT_2]  = ww[HAP_MUT_2]  * (pop[HAP_MUT_2] + mu * pop[HAP_ANC] + rec_effect);
        probs[HAP_DOUBLE] = ww[HAP_DOUBLE] * (pop[HAP_DOUBLE] + mu * (pop[HAP_MUT_1] + pop[HAP_MUT_2]) - rec_effect);

    } else {
        // Standard quadratic fitness / multiplicative selection
        double w_sing_1 = exp(-pow(beta_1, 2));
        double w_sing_2 = exp(-pow(beta_2, 2));
        double w_doub   = exp(-pow(beta_1 + beta_2, 2));

        // Note: probs[0] fitness is 1.0 implicitly in original code
        probs[HAP_ANC]    = pop[HAP_ANC] - mu * pop[HAP_ANC] - rec_effect;
        probs[HAP_MUT_1]  = w_sing_1 * (pop[HAP_MUT_1] + mu * pop[HAP_ANC] + rec_effect);
        probs[HAP_MUT_2]  = w_sing_2 * (pop[HAP_MUT_2] + mu * pop[HAP_ANC] + rec_effect);
        probs[HAP_DOUBLE] = w_doub   * (pop[HAP_DOUBLE] + mu * (pop[HAP_MUT_1] + pop[HAP_MUT_2]) - rec_effect);
    }
}

/**
 * @brief Advances the population by one generation.
 * Calculates selection/mutation/recombination probs -> Multinomial sampling.
 */
void evolve_generation(int N, double beta_1, double beta_2, double r_prob, double mu, unsigned int *pop, int polygenic, gsl_rng *r){
    double probs[NUM_HAPS] = {0};
    
    // N from previous step is sum of current haplotypes
    // (Should equal N, but safe to recalculate if N varies in demography)
    int N_prev = pop[HAP_ANC] + pop[HAP_MUT_1] + pop[HAP_MUT_2] + pop[HAP_DOUBLE];
    
    // Recombination probability per individual
    double rec_val = r_prob / (double)N_prev;

    calculate_fitness_probs(pop, probs, N_prev, beta_1, beta_2, mu, rec_val, polygenic);

    // Normalize probabilities (ensure sum = 1.0 for multinomial)
    double total = probs[0] + probs[1] + probs[2] + probs[3];
    // Safety check for extinction or numerical underflow
    if (total <= 0.0) total = 1.0; 

    for(int i=0; i<NUM_HAPS; i++) {
        probs[i] /= total;
        // Clamp negative probabilities caused by numerical approximation of D
        if(probs[i] < 0) probs[i] = 0.0; 
    }

    // Sample next generation
    gsl_ran_multinomial(r, NUM_HAPS, N, probs, pop);
}


/* --- CORE SIMULATION LOGIC --- */

/**
 * @brief Runs a single simulation trajectory.
 */
void simulate_trajectory(SimParams p, 
                         gsl_rng *r, 
                         int *demo_sizes, 
                         int demo_count, 
                         unsigned int *final_pop_storage) {
    
    unsigned int pop[NUM_HAPS] = {p.N, 0, 0, 0}; // Start fixed for Ancestral
    int current_gen = 0;
    
    double mu = p.theta / (2.0 * p.N); 
    double r_prob = p.rho / (2.0 * p.N);

    /* --- PHASE 1: Burn-in / Mutation Origin --- */
    // This loop simulates until a specific time limit or fixation event.
    // Logic: It waits for a mutation to establish. If lost, it resets.
    
    int max_gens = p.run_mult * p.N;

    while (current_gen < max_gens) {
        
        // 1. Check for fixation of derived alleles (Loss of Ancestral)
        // If HAP_MUT_1 or HAP_MUT_2 fixed, rotate them to Ancestral slot and reset others
        // (Simulating a sweep, then waiting for next mutation)
        if (pop[HAP_MUT_1] + pop[HAP_DOUBLE] == p.N) {
            pop[HAP_ANC] = p.N - pop[HAP_DOUBLE];
            pop[HAP_MUT_1] = 0; 
            pop[HAP_MUT_2] = pop[HAP_DOUBLE]; // Rotate double to single
            pop[HAP_DOUBLE] = 0;
        } else if (pop[HAP_MUT_2] + pop[HAP_DOUBLE] == p.N) {
            pop[HAP_ANC] = p.N - pop[HAP_DOUBLE];
            pop[HAP_MUT_1] = pop[HAP_DOUBLE]; // Rotate
            pop[HAP_MUT_2] = 0;
            pop[HAP_DOUBLE] = 0;
        }

        // 2. Injection of new mutations logic
        // If population is monomorphic ancestral (pop[0] == N)
        if (pop[HAP_ANC] == p.N) {
            // Introduce mutation at locus 1 or 2 (50/50 chance)
            if (gsl_rng_uniform(r) < 0.5) {
                pop[HAP_MUT_1] = 1;
                pop[HAP_ANC] = p.N - 1;
            } else {
                pop[HAP_MUT_2] = 1;
                pop[HAP_ANC] = p.N - 1;
            }

            // Sample time-to-mutation (Geometric distribution)
            // waiting time ~ Geometric(theta)
            int gens_to_mut = 1 + gsl_ran_geometric(r, p.theta);
            
            // If waiting time exceeds simulation length, we are done (nothing happened)
            if ((max_gens - current_gen) < gens_to_mut) {
                pop[HAP_ANC] = p.N;
                pop[HAP_MUT_1] = 0; pop[HAP_MUT_2] = 0; pop[HAP_DOUBLE] = 0;
                break; // End simulation
            }
            current_gen += gens_to_mut;
        } else {
            current_gen += 1;
        }

        evolve_generation(p.N, p.beta_1, p.beta_2, r_prob, mu, pop, p.polygenic, r);
    }

    /* --- PHASE 2: Demography (Optional) --- */
    if (p.demography == 1 && demo_sizes != NULL) {
        for (int i = 0; i < demo_count; i++) {
            // CAST FIX: Make diploid_pop_size unsigned to match pop[] array type
            unsigned int diploid_pop_size = (unsigned int)(2 * demo_sizes[i]);

            evolve_generation(diploid_pop_size, p.beta_1, p.beta_2, r_prob, mu, pop, p.polygenic, r);

            // Re-apply fixation check (Rotation logic) inside demography
            // Now comparing unsigned to unsigned
            if (pop[HAP_MUT_1] + pop[HAP_DOUBLE] == diploid_pop_size) {
                pop[HAP_ANC] = diploid_pop_size - pop[HAP_DOUBLE];
                pop[HAP_MUT_1] = 0;
                pop[HAP_MUT_2] = pop[HAP_DOUBLE];
                pop[HAP_DOUBLE] = 0;
            } else if (pop[HAP_MUT_2] + pop[HAP_DOUBLE] == diploid_pop_size) {
                pop[HAP_ANC] = diploid_pop_size - pop[HAP_DOUBLE];
                pop[HAP_MUT_1] = pop[HAP_DOUBLE];
                pop[HAP_MUT_2] = 0;
                pop[HAP_DOUBLE] = 0;
            }
        }
    }

    // Save results
    for(int i=0; i<NUM_HAPS; i++) final_pop_storage[i] = pop[i];
}


/* --- MAIN DRIVER --- */

int main(int argc, char *argv[]) {
    if (argc != 12) {
        fprintf(stderr, "Usage: %s N_sims N_popsize theta rho run_mult eff_size_1 eff_size_2 symm_param seed polygenic [demography_file OR 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse Arguments
    int N_sims = atoi(argv[1]);
    
    SimParams p;
    p.N = atoi(argv[2]);
    p.theta = atof(argv[3]);
    p.rho = atof(argv[4]);
    p.run_mult = atoi(argv[5]);
    
    float eff_size_float_1 = atof(argv[6]);
    float eff_size_float_2 = atof(argv[7]);
    double symm_param = atof(argv[8]);
    double slurm_seed = atof(argv[9]);
    p.polygenic = atoi(argv[10]);

    // Load Demography Logic
    // We check if the last arg is "0" (Off), "1" (Default file), or a custom filename.
    char *demo_arg = argv[11];
    char *demo_filename = "demo_Ns.txt"; // Default for backward compatibility

    if (strcmp(demo_arg, "0") == 0) {
        p.demography = 0;
    } else if (strcmp(demo_arg, "1") == 0) {
        p.demography = 1;
        // demo_filename remains "demo_Ns.txt"
    } else {
        p.demography = 1;
        demo_filename = demo_arg;
    }

    // Load Demography ONCE if needed
    int *demo_sizes = NULL;
    int demo_count = 0;
    if (p.demography == 1) {
        demo_sizes = load_demography_file(demo_filename, &demo_count);
        fprintf(stderr, "Loaded demography from '%s': %d generations.\n", demo_filename, demo_count);
    }

    // Setup GSL RNG
    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);
    
    // Seed: Combine system time with SLURM seed for uniqueness on cluster
    unsigned long mySeed = (unsigned long)time(NULL) + (unsigned long)slurm_seed;
    gsl_rng_set(r, mySeed);

    fprintf(stderr, "Starting %d simulations...\n", N_sims);
    clock_t start = clock();

    // Print Header
    printf("beta_1\tbeta_2\tN00\tN01\tN10\tN11\n");

    // Loop simulations
    // Note: We do NOT store all results in a massive array. We print as we go.
    // This prevents stack overflow for large N_sims.
    for (int i = 0; i < N_sims; ++i) {
        // Randomize betas for this specific run
        p.beta_1 = eff_size_float_1 * get_random_direction(r, symm_param);
        p.beta_2 = eff_size_float_2 * get_random_direction(r, symm_param);

        unsigned int result_pop[NUM_HAPS];
        
        simulate_trajectory(p, r, demo_sizes, demo_count, result_pop);

        printf("%.6f\t%.6f\t%d\t%d\t%d\t%d\n", 
               p.beta_1, p.beta_2, 
               result_pop[HAP_ANC], result_pop[HAP_MUT_1], 
               result_pop[HAP_MUT_2], result_pop[HAP_DOUBLE]);
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Execution time: %.2f seconds\n", time_taken);

    // Cleanup
    if (demo_sizes) free(demo_sizes);
    gsl_rng_free(r);

    return EXIT_SUCCESS;
}