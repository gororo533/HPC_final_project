/*
 * topk.c - Sparse top-K pheromone update implementation
 *
 * Strategy
 * --------
 * Instead of AllReduce over all n_ants paths (original behaviour), we:
 *   1. Partial-sort the ant colony to find the top-k shortest tours.
 *   2. Apply global_update_pheromone() only to those k ants.
 *   3. Optionally weight by rank (rank 1 gets weight k, rank k gets weight 1)
 *      so the signal is qualitatively similar to RAS but cheaper.
 *
 * The function also records timing in the global perf_counters_t struct so
 * we can compare wall time against the original update at the end.
 *
 * Overlap note
 * ------------
 * Full MPI_Isend/Irecv overlap requires MPI (not present in this serial
 * baseline).  This file provides the algorithmic "sparse update" half of
 * the optimisation so you can measure solution quality and timing before
 * adding MPI.  The overlap skeleton is documented in comments below.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ants.h"
#include "utilities.h"
#include "InOut.h"
#include "TSP.h"
#include "timer.h"
#include "topk.h"

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

long int top_k_flag = 0;   /* default: original mode */
long int top_k      = 5;   /* default: sync 5 best paths */

perf_counters_t perf = {0.0, 0.0, 0.0, 0.0, 0, 0, 0};

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/* Comparison function for qsort: ascending tour_length */
static int cmp_ant_by_length(const void *a, const void *b)
{
    long int la = ((const ant_struct *)a)->tour_length;
    long int lb = ((const ant_struct *)b)->tour_length;
    if (la < lb) return -1;
    if (la > lb) return  1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void topk_pheromone_update(void)
/*
    FUNCTION:  Apply pheromone update using only the top-k ants.
    INPUT:     none (uses global ant[], n_ants, top_k)
    OUTPUT:    none
    EFFECTS:   pheromone matrix updated; perf counters incremented

    MPI overlap skeleton (for future extension)
    -------------------------------------------
    In a distributed setting each rank holds its own sub-colony.
    After local top-k selection, the k best (tour + length) can be
    exchanged with MPI_Isend/Irecv while the local pheromone evaporation
    runs concurrently:

        MPI_Isend(local_topk_tours, ..., dest, tag, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(remote_topk_tours, ..., source, tag, MPI_COMM_WORLD, &req_recv);

        evaporation();                    // overlaps with communication

        MPI_Wait(&req_send, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

        // merge remote top-k with local top-k, update pheromone
*/
{
    long int k_actual;
    long int i;
    /* index array for partial sort (avoid copying full ant structs) */
    long int *idx;
    double t_start;

    t_start = elapsed_time(REAL);

    k_actual = (top_k < n_ants) ? top_k : n_ants;

    /* Build an index array sorted by tour_length (partial sort for top-k) */
    idx = (long int *)malloc(sizeof(long int) * (size_t)n_ants);
    if (idx == NULL) {
        fprintf(stderr, "topk_pheromone_update: out of memory\n");
        exit(1);
    }
    for (i = 0; i < n_ants; i++)
        idx[i] = i;

    /* Simple selection for small k_actual; qsort for large k_actual.
       For the typical case (k_actual <= 20) selection is faster. */
    if (k_actual <= 20) {
        long int j, tmp, min_idx;
        for (i = 0; i < k_actual; i++) {
            min_idx = i;
            for (j = i + 1; j < n_ants; j++) {
                if (ant[idx[j]].tour_length < ant[idx[min_idx]].tour_length)
                    min_idx = j;
            }
            tmp = idx[i]; idx[i] = idx[min_idx]; idx[min_idx] = tmp;
        }
    } else {
        /* Full sort via qsort on idx array */
        /* We need a comparator with access to ant[] - use a static pointer trick */
        /* For simplicity, just sort with a local lambda-equivalent */
        long int j, tmp;
        /* insertion sort is fine for up to a few hundred ants */
        for (i = 1; i < n_ants; i++) {
            tmp = idx[i]; j = i;
            while (j > 0 && ant[idx[j-1]].tour_length > ant[tmp].tour_length) {
                idx[j] = idx[j-1];
                j--;
            }
            idx[j] = tmp;
        }
    }

    /* Apply weighted pheromone update: rank 1 (best) gets weight k_actual,
       rank k_actual (worst of selected) gets weight 1.
       This mirrors rank-based AS (RAS) semantics. */
    for (i = 0; i < k_actual; i++) {
        long int weight = k_actual - i;   /* rank 1 → weight k, rank k → weight 1 */
        global_update_pheromone_weighted(&ant[idx[i]], weight);
    }

    free(idx);

    perf.pheromone_update_time += elapsed_time(REAL) - t_start;
    perf.n_pheromone_updates++;
    perf.n_topk_syncs += k_actual;
}


void print_performance_summary(void)
/*
    FUNCTION:  Print a structured timing/performance report to stdout.
    INPUT:     none
    OUTPUT:    none
*/
{
    printf("\n");
    printf("============================================================\n");
    printf("  PERFORMANCE SUMMARY\n");
    printf("  Mode: %s", top_k_flag ? "SPARSE TOP-K UPDATE" : "ORIGINAL BLOCKING UPDATE");
    if (top_k_flag)
        printf(" (k=%ld)", top_k);
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("  Total wall time          : %.4f s\n",  perf.total_wall_time);
    printf("  Solution construction    : %.4f s\n",  perf.solution_construct_time);
    printf("  Local search             : %.4f s\n",  perf.local_search_time);
    printf("  Pheromone update         : %.4f s\n",  perf.pheromone_update_time);
    printf("  Other (stats/overhead)   : %.4f s\n",
           perf.total_wall_time
           - perf.solution_construct_time
           - perf.local_search_time
           - perf.pheromone_update_time);
    printf("------------------------------------------------------------\n");
    printf("  Total iterations         : %ld\n",     perf.n_iterations_total);
    printf("  Pheromone update calls   : %ld\n",     perf.n_pheromone_updates);
    if (top_k_flag)
        printf("  Top-k ant syncs total    : %ld\n", perf.n_topk_syncs);
    printf("============================================================\n\n");
}
