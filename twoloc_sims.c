// Run: ./sims 100000 20000 0.01 0.0 5 0.0 0.5

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//#define N_MAX 20000

// Function to simulate a random choice based on probability
int random_choice(double probability) {
//    printf("random choice");
    return ((double)rand() / RAND_MAX) < probability ? 1 : -1;
}


// Function to simulate the population
void simulate_population(unsigned int N, 
                         double theta, 
                         double rho, 
                         int run_mult, 
                         double beta_1, 
                         double beta_2, 
                         unsigned int *pop) {

    int run = 0;
    double w_sing = exp(-pow(beta_1, 2));
    double w_doub = exp(-pow(beta_1 + beta_2, 2));
    double ww[4] = {0.0, 0.0, 0.0, 0.0};    // Fitnesses of haplotypes
    double probs[4] = {1.0, 0.0, 0.0, 0.0}; // Probabilities of haplotypes in next generation
    double deviation = 0.0;
    double total = 0.0;
    double mu = theta / (2 * N); // theta = 2*Ne*mu in a haploid population
    // double mu = theta / N;

    // Set up the RNG
    // TODO: give these better names
    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    while (run < run_mult * N) {
        // HAPLOTYPE VECTOR:
        // pop[0]: 0,0
        // pop[1]: 1,0
        // pop[2]: 0,1
        // pop[3]: 1,1

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
            int gens_to_mut = gsl_ran_geometric(r, theta); // Why did we have 1 + gsl_ran... here?
            if ((run_mult * N - run) < gens_to_mut) {      // geometric dist in gsl starts support at k=1
                pop[0] = N;
                pop[1] = 0;
                pop[2] = 0;
                pop[3] = 0;
                break;
                // end and output a monomorphic population if we wait too many generations for a mutation
            }
            run += gens_to_mut;
        } else {
            run += 1;
        }

        deviation = (beta_1 * (pop[1] + pop[3]) + beta_2 * (pop[2] + pop[3])) / N;
        ww[0] = 1 - pow(deviation, 2);
        ww[1] = 1 - pow(beta_1 - deviation, 2);
        ww[2] = 1 - pow(beta_2 - deviation, 2);
        ww[3] = 1 - pow(beta_1 + beta_2 - deviation, 2);

        probs[0] = ww[0] * (pop[0] - mu * pop[0] + rho * (pop[1] * pop[2] - pop[0] * pop[3]));
        probs[1] = ww[1] * (pop[1] + mu * pop[0] + rho * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[2] = ww[2] * (pop[2] + mu * pop[0] + rho * (pop[0] * pop[3] - pop[1] * pop[2]));
        probs[3] = ww[3] * (pop[3] + mu * (pop[1] + pop[2]) + rho * (pop[1] * pop[2] - pop[0] * pop[3]));

        // Normalize probabilities
        total = probs[0] + probs[1] + probs[2] + probs[3];
        probs[0] /= total;
        probs[1] /= total;
        probs[2] /= total;
        probs[3] /= total;
        
        // Sample from multinomial distribution
        // printf("probs: %d, %d, %d, %d, %d\n", pop[0], pop[1], pop[2], pop[3], N);
        gsl_ran_multinomial(r, 4, N, probs, pop);
    }

    gsl_rng_free(r);

}

// Function to simulate batch of populations
void sim_batch(int N_sims, 
               unsigned int N, 
               double theta, 
               double rho, 
               int run_mult, 
               float eff_size_float, 
               double symm_param) {

    double beta_1, beta_2;
    double beta_arr[N_sims][2];
    int pop_arr[N_sims][4];
    rho = rho / (N*N);

    srand(time(NULL));

    for (int i = 0; i < N_sims; ++i) {        
        // Randomly choose beta values
        beta_1 = eff_size_float * random_choice(symm_param);
        beta_2 = eff_size_float * random_choice(symm_param);
        beta_arr[i][0] = beta_1;
        beta_arr[i][1] = beta_2;
        
        // Simulate population
        unsigned int pop[4] = {N, 0, 0, 0};
        
        simulate_population(N, theta, rho, run_mult, beta_1, beta_2, pop);

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
    if (argc != 8) {
        fprintf(stderr, "Usage: %s N_sims N_popsize rho run_mult eff_size symm_param\n", argv[0]);
        return 1;
    }

    int N_sims = atoi(argv[1]);
    unsigned int N = atoi(argv[2]);
    double theta = atof(argv[3]);
    double rho = atof(argv[4]);
    int run_mult = atoi(argv[5]);
    float eff_size_float = atof(argv[6]);
    double symm_param = atof(argv[7]);

    fprintf(stderr, "starting\n");

    clock_t start = clock(); // Start the timer

    // 10000 20000 0.01 0.0 5 0.0 0.5
    sim_batch(N_sims, N, theta, rho, run_mult, eff_size_float, symm_param);

    clock_t end = clock(); // Stop the timer

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC; // Calculate the elapsed time

    fprintf(stderr, "Execution time: %.2f seconds\n", time_taken);

    return 0;
}