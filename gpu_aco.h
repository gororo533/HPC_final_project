/*
 * gpu_aco.h — Host API for GPU ant construction and 2-opt local search
 *
 * Stages supported:
 *   Stage 1: gpu_construct_solutions()  — parallel ant construction on GPU
 *   Stage 2: gpu_local_search_2opt()   — GPU 2-opt with nn_list restriction
 */

#ifndef GPU_ACO_H
#define GPU_ACO_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize GPU memory and upload static data.
 * Call once per run after compute_nn_lists() and compute_distances().
 *
 *   n          : number of cities
 *   n_ants     : colony size
 *   nn_ants    : nearest-neighbour depth for construction
 *   nn_ls      : nearest-neighbour depth for 2-opt local search
 *   alpha/beta : pheromone / heuristic exponents
 *   q0         : ACS greedy probability (0.0 = pure roulette)
 *   dist_cpu   : n×n distance matrix (CPU side, long int**)
 *   nn_list_cpu: n×nn_ants nearest-neighbour list (CPU side, long int**)
 */
void gpu_init(int n, int n_ants, int nn_ants, int nn_ls,
              double alpha, double beta, double q0,
              long int **dist_cpu,
              long int **nn_list_cpu);

/*
 * Upload updated pheromone and total matrices to the GPU.
 * Call at the beginning of each iteration (after pheromone update).
 */
void gpu_upload_pheromone(double **pheromone_cpu, double **total_cpu, int n);

/*
 * Launch Stage 1: parallel ant construction.
 * Non-blocking — overlap with CPU work if desired.
 */
void gpu_construct_solutions(void);

/*
 * Launch Stage 2: GPU 2-opt local search (nn_list-restricted).
 * Blocks until all passes are complete for all ants.
 */
void gpu_local_search_2opt(void);

/*
 * Download all ant tours and tour lengths from GPU to CPU arrays.
 * Synchronizes the device before copying.
 *
 *   tour_out[k]   : pre-allocated long int array of length (n+1)
 *   length_out[k] : pre-allocated long int scalar
 */
void gpu_download_results(long int **tour_out, long int *length_out,
                          int n_ants, int n);

/*
 * Free all GPU resources.  Call once at program exit.
 */
void gpu_free(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ACO_H */
