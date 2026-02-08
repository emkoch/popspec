/**
 * @file sims_pleiotropy.c
 * @brief Wright-Fisher simulation with pleiotropy support (2 phenotypes per locus).
 *
 * Similar structure to sim.c but each locus affects multiple phenotypes.
 * Fitness model: s = 1 - exp(-I2 * sum_phenotypes(effect_p^2))
 *
 * Compilation:
 * gcc -I/n/app/gsl/2.8-gcc-14.2.0/include -O3 -std=c99 -o sims_pleiotropy sims_pleiotropy.c -L/n/app/gsl/2.8-gcc-14.2.0/lib -lgsl -lgslcblas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* --- CONSTANTS --- */
enum Haplotype {
    HAP_ANC = 0,      // Neither mutation
    HAP_MUT_1 = 1,    // Mutation at locus 1 only
    HAP_MUT_2 = 2,    // Mutation at locus 2 only
    HAP_DOUBLE = 3,   // Both mutations
    NUM_HAPS = 4
};

#define N_PHENOTYPES 2  // Function and Abundance

/* --- STRUCTS --- */
typedef struct {
    unsigned int N;              // Population size
    double theta;                // Mutation rate param
    double rho;                  // Recombination rate param
    int run_mult;                // Generation multiplier
    double beta_1[N_PHENOTYPES]; // Effect sizes for locus 1 [func, abun]
    double beta_2[N_PHENOTYPES]; // Effect sizes for locus 2 [func, abun]
    double I2;                   // Intensity parameter
} SimParams;

/* --- FITNESS CALCULATION --- */

/**
 * @brief Calculate fitness using pleiotropy model.
 * Fitness = 1 - s, where s = 1 - exp(-I2 * sum_phenotypes(effect_p^2))
 */
double calculate_fitness(int haplotype, SimParams *p) {
    double sum_sq = 0.0;

    // For each phenotype
    for (int pheno = 0; pheno < N_PHENOTYPES; pheno++) {
        double effect = 0.0;

        // Add effects based on which mutations are present
        if (haplotype == HAP_MUT_1 || haplotype == HAP_DOUBLE) {
            effect += p->beta_1[pheno];
        }
        if (haplotype == HAP_MUT_2 || haplotype == HAP_DOUBLE) {
            effect += p->beta_2[pheno];
        }

        sum_sq += effect * effect;
    }

    // s = 1 - exp(-I2 * sum_sq)
    double s = 1.0 - exp(-p->I2 * sum_sq);
    return 1.0 - s;  // Fitness = 1 - s
}

/* --- EVOLUTION --- */

/**
 * @brief Evolve population one generation with selection and recombination.
 */
void evolve_generation(unsigned int N, double rho, unsigned int *pop, SimParams *p, gsl_rng *r) {
    // Calculate fitness for each haplotype
    double w[NUM_HAPS];
    double w_bar = 0.0;

    for (int h = 0; h < NUM_HAPS; h++) {
        w[h] = calculate_fitness(h, p);
        w_bar += pop[h] * w[h];
    }
    w_bar /= N;

    // Selection: multinomial sampling proportional to fitness
    double probs[NUM_HAPS];
    for (int h = 0; h < NUM_HAPS; h++) {
        probs[h] = (pop[h] * w[h]) / (N * w_bar);
    }

    unsigned int new_pop[NUM_HAPS];
    gsl_ran_multinomial(r, NUM_HAPS, N, probs, new_pop);

    // Recombination: break up double mutants
    if (rho > 0.0 && new_pop[HAP_DOUBLE] > 1) {
        unsigned int n_recomb = gsl_ran_binomial(r, rho / 2.0, new_pop[HAP_DOUBLE]);
        new_pop[HAP_DOUBLE] -= n_recomb;
        new_pop[HAP_MUT_1] += n_recomb / 2;
        new_pop[HAP_MUT_2] += (n_recomb + 1) / 2;  // Handle odd numbers
    }

    // Update population
    for (int h = 0; h < NUM_HAPS; h++) {
        pop[h] = new_pop[h];
    }
}

/**
 * @brief Run a single simulation trajectory.
 */
void simulate_trajectory(SimParams p, gsl_rng *r, unsigned int *final_pop) {
    unsigned int pop[NUM_HAPS] = {p.N, 0, 0, 0};  // Start with all ancestral

    int max_gens = p.run_mult * p.N;
    int current_gen = 0;

    double mu = p.theta / (2.0 * p.N);  // Per-locus mutation rate
    double r_prob = p.rho / (2.0 * p.N);  // Per-locus recombination rate

    /* --- PHASE 1: Standard Evolution --- */
    while (current_gen < max_gens) {

        // Fixation check: rotate if single mutation fixes
        if (pop[HAP_MUT_1] + pop[HAP_DOUBLE] == p.N) {
            pop[HAP_ANC] = p.N - pop[HAP_DOUBLE];
            pop[HAP_MUT_1] = 0;
            pop[HAP_MUT_2] = pop[HAP_DOUBLE];
            pop[HAP_DOUBLE] = 0;
        } else if (pop[HAP_MUT_2] + pop[HAP_DOUBLE] == p.N) {
            pop[HAP_ANC] = p.N - pop[HAP_DOUBLE];
            pop[HAP_MUT_1] = pop[HAP_DOUBLE];
            pop[HAP_MUT_2] = 0;
            pop[HAP_DOUBLE] = 0;
        }

        // Inject new mutation if monomorphic
        if (pop[HAP_ANC] == p.N) {
            // Introduce mutation at locus 1 or 2 (50/50)
            if (gsl_rng_uniform(r) < 0.5) {
                pop[HAP_MUT_1] = 1;
                pop[HAP_ANC] = p.N - 1;
            } else {
                pop[HAP_MUT_2] = 1;
                pop[HAP_ANC] = p.N - 1;
            }

            // Sample waiting time
            int gens_to_mut = 1 + gsl_ran_geometric(r, p.theta);

            if ((max_gens - current_gen) < gens_to_mut) {
                pop[HAP_ANC] = p.N;
                pop[HAP_MUT_1] = 0;
                pop[HAP_MUT_2] = 0;
                pop[HAP_DOUBLE] = 0;
                break;
            }
            current_gen += gens_to_mut;
        } else {
            current_gen += 1;
        }

        evolve_generation(p.N, r_prob, pop, &p, r);
    }

    // Save results
    for (int h = 0; h < NUM_HAPS; h++) {
        final_pop[h] = pop[h];
    }
}

/* --- MAIN --- */

int main(int argc, char *argv[]) {
    if (argc != 12) {
        fprintf(stderr, "Usage: %s N_sims N_popsize theta rho run_mult ", argv[0]);
        fprintf(stderr, "beta1_func beta1_abun beta2_func beta2_abun symm_param seed [I2]\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s 1000 20000 0.1 0.0 15 -0.02 -0.01 -0.03 -0.015 0.5 1 1.0\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse arguments
    int N_sims = atoi(argv[1]);

    SimParams p;
    p.N = atoi(argv[2]);
    p.theta = atof(argv[3]);
    p.rho = atof(argv[4]);
    p.run_mult = atoi(argv[5]);

    // Effect sizes
    float beta1_func = atof(argv[6]);
    float beta1_abun = atof(argv[7]);
    float beta2_func = atof(argv[8]);
    float beta2_abun = atof(argv[9]);

    double symm_param = atof(argv[10]);
    double slurm_seed = atof(argv[11]);
    p.I2 = (argc > 12) ? atof(argv[12]) : 1.0;

    // Setup RNG
    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);

    unsigned long mySeed = (unsigned long)time(NULL) + (unsigned long)slurm_seed;
    gsl_rng_set(r, mySeed);

    fprintf(stderr, "Starting %d pleiotropy simulations...\n", N_sims);
    fprintf(stderr, "I2=%.3f, N=%u, theta=%.6f, rho=%.6f\n", p.I2, p.N, p.theta, p.rho);
    clock_t start = clock();

    // Print header (matching sim.c format)
    printf("beta1_func\tbeta1_abun\tbeta2_func\tbeta2_abun\tN00\tN01\tN10\tN11\n");

    // Run simulations
    for (int i = 0; i < N_sims; i++) {
        // Randomize beta signs (symmetric around 0 or all negative)
        int dir1 = (gsl_rng_uniform(r) < symm_param) ? 1 : -1;
        int dir2 = (gsl_rng_uniform(r) < symm_param) ? 1 : -1;

        p.beta_1[0] = beta1_func * dir1;  // Function phenotype, locus 1
        p.beta_1[1] = beta1_abun * dir1;  // Abundance phenotype, locus 1
        p.beta_2[0] = beta2_func * dir2;  // Function phenotype, locus 2
        p.beta_2[1] = beta2_abun * dir2;  // Abundance phenotype, locus 2

        unsigned int result_pop[NUM_HAPS];
        simulate_trajectory(p, r, result_pop);

        printf("%.6f\t%.6f\t%.6f\t%.6f\t%u\t%u\t%u\t%u\n",
               p.beta_1[0], p.beta_1[1], p.beta_2[0], p.beta_2[1],
               result_pop[HAP_ANC], result_pop[HAP_MUT_2],
               result_pop[HAP_MUT_1], result_pop[HAP_DOUBLE]);
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Execution time: %.2f seconds\n", time_taken);

    gsl_rng_free(r);
    return EXIT_SUCCESS;
}
