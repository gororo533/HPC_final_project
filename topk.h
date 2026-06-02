/*
 * topk.h - Top-K sparse pheromone update for ACO/TSP HPC optimization
 *
 * New in this file:
 *   - top_k_flag  : global flag (0 = original AllReduce, 1 = sparse top-k update)
 *   - top_k       : how many best paths to synchronise per iteration
 *   - Performance counters for timing comparisons
 */

#ifndef TOPK_H
#define TOPK_H

#include <stdint.h>

/* ---- mode switch ---- */
extern long int top_k_flag;   /* 0: original blocking update  1: sparse top-k */
extern long int top_k;        /* number of best paths kept per iteration      */

/* ---- performance counters ---- */
typedef struct {
    double total_wall_time;        /* wall-clock seconds for the whole run     */
    double pheromone_update_time;  /* seconds spent in pheromone update phase  */
    double solution_construct_time;/* seconds in construct_solutions            */
    double local_search_time;      /* seconds in local_search                  */
    long int n_iterations_total;   /* total iterations across all tries        */
    long int n_pheromone_updates;  /* how many pheromone update calls          */
    long int n_topk_syncs;         /* top-k path syncs (top_k_flag=1 only)     */
} perf_counters_t;

extern perf_counters_t perf;

/* ---- API ---- */
/* Collect the top-k ants (by tour_length) and apply their pheromone update.
   Uses the same global_update_pheromone() as the original code, but only
   for the k best ants instead of all n_ants.  Falls back to the caller's
   normal update when top_k >= n_ants.                                        */
void topk_pheromone_update(void);

/* Print the performance summary to stdout (called at program end).           */
void print_performance_summary(void);

#endif /* TOPK_H */
