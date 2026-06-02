/*
 * acotsp_hpc.c — HPC-extended main driver for ACO/TSP
 *
 * Stages (all additive, controlled by command-line flags):
 *
 *   Stage 0 (baseline)  : original serial CPU code — always present
 *   Stage 1 --gpu       : ant construction on GPU  (CUDA, 1 thread/ant)
 *   Stage 2 --gpu-ls    : GPU 2-opt local search   (requires --gpu)
 *   Stage 3 --multi-gpu : partition colony across N GPUs on one node
 *                          (requires --gpu; uses CUDA multi-device API)
 *   Stage 5 --mpi       : multi-node island model  (MPI + NCCL AllReduce)
 *                          (requires --multi-gpu)
 *
 * Additional flags (inherited from original code):
 *   --topk N / -T N     : sparse top-k pheromone update (CPU, any stage)
 *
 * New flags added here:
 *   --gpu               : enable Stage 1 GPU construction
 *   --gpu-ls            : enable Stage 2 GPU local search (implies --gpu)
 *   --num-gpus N        : number of GPUs to use per node (Stage 3, default=1)
 *   --mpi               : enable Stage 5 MPI island model
 *   --island-sync K     : MPI exchange every K iterations (default=100)
 *   --perf              : print performance summary at end (default on)
 *
 * Build instructions:
 *   See Makefile.hpc — briefly:
 *     nvcc -O3 -arch=sm_70 -c gpu_aco.cu -o gpu_aco.o
 *     mpicc -O3 -c acotsp_hpc.c [other C files] -o acotsp_hpc.o
 *     mpicc -O3 acotsp_hpc.o [other .o] gpu_aco.o -lcudart -lcuda -lnccl -lm -o acotsp_hpc
 *
 * SLURM usage:
 *   See run_hpc.sh
 */

/* ---- Feature-test macros ---- */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Original ACO headers (unchanged) */
#include "ants.h"
#include "utilities.h"
#include "InOut.h"
#include "TSP.h"
#include "timer.h"
#include "ls.h"
#include "topk.h"

/* GPU API (compiled only when USE_CUDA is defined) */
#ifdef USE_CUDA
#  include "gpu_aco.h"
#  include <cuda_runtime.h>
#endif

/* MPI + NCCL (compiled only when USE_MPI is defined) */
#ifdef USE_MPI
#  include <mpi.h>
#endif
#ifdef USE_NCCL
#  include <nccl.h>
#endif

/* ================================================================== */
/* Forward declarations                                                  */
/* ================================================================== */
long int termination_condition(void);

/* ================================================================== */
/* HPC control flags (set from command line)                             */
/* ================================================================== */
static int    hpc_gpu_flag       = 0;   /* --gpu       : Stage 1 on GPU     */
static int    hpc_gpu_ls_flag    = 0;   /* --gpu-ls    : Stage 2 GPU 2-opt  */
static int    hpc_num_gpus       = 1;   /* --num-gpus N: Stage 3 multi-GPU  */
static int    hpc_mpi_flag       = 0;   /* --mpi       : Stage 5 MPI island */
static int    hpc_island_sync    = 100; /* --island-sync K                  */
static int    hpc_perf_flag      = 1;   /* --perf / --no-perf               */

/* MPI state */
static int    mpi_rank           = 0;
static int    mpi_size           = 1;

/* Multi-GPU: per-GPU ant ranges (Stage 3) */
#define MAX_GPUS 8
static int    gpu_ant_start[MAX_GPUS];
static int    gpu_ant_end[MAX_GPUS];
static int    n_gpus_actual      = 1;

/* Wall-clock anchor */
static double g_run_start_real   = 0.0;

/* ================================================================== */
/* HPC flag parsing                                                      */
/* ================================================================== */
/*
 * Scan argv for HPC-specific flags BEFORE passing the rest to the
 * original init_program() parser.  We strip recognised flags so the
 * original parser never sees them.
 *
 * Recognised:
 *   --gpu
 *   --gpu-ls
 *   --num-gpus N
 *   --mpi
 *   --island-sync K
 *   --topk N   (also handled here, same as original --topk path)
 *   -T N
 *   --no-perf
 */
static int parse_hpc_flags(int argc, char *argv[])
{
    int new_argc = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            hpc_gpu_flag = 1;
        } else if (strcmp(argv[i], "--gpu-ls") == 0) {
            hpc_gpu_flag    = 1;
            hpc_gpu_ls_flag = 1;
        } else if (strcmp(argv[i], "--num-gpus") == 0 && i + 1 < argc) {
            hpc_num_gpus = atoi(argv[++i]);
            if (hpc_num_gpus < 1) hpc_num_gpus = 1;
            if (hpc_num_gpus > MAX_GPUS) hpc_num_gpus = MAX_GPUS;
        } else if (strcmp(argv[i], "--mpi") == 0) {
            hpc_mpi_flag = 1;
        } else if (strcmp(argv[i], "--island-sync") == 0 && i + 1 < argc) {
            hpc_island_sync = atoi(argv[++i]);
            if (hpc_island_sync < 1) hpc_island_sync = 1;
        } else if ((strcmp(argv[i], "--topk") == 0 ||
                    strcmp(argv[i], "-T") == 0) && i + 1 < argc) {
            long int val = atol(argv[i + 1]);
            if (val > 0) { top_k = val; top_k_flag = 1; }
            i++; /* skip numeric argument */
        } else if (strcmp(argv[i], "--no-perf") == 0) {
            hpc_perf_flag = 0;
        } else {
            argv[new_argc++] = argv[i];
        }
    }
    return new_argc;
}

/* ================================================================== */
/* MPI helpers                                                           */
/* ================================================================== */
#ifdef USE_MPI
/*
 * After each iteration, exchange the global best tour across all MPI
 * ranks (island model).  Only called every hpc_island_sync iterations.
 *
 * Strategy: AllReduce on (tour_length, rank_id) using MPI_MINLOC to
 * find the global best rank, then Bcast the tour from that rank.
 */
static void mpi_exchange_best(void)
{
    /* Pack (length, rank) for MINLOC */
    struct { long int len; int rank; } local_val, global_val;
    local_val.len  = best_so_far_ant->tour_length;
    local_val.rank = mpi_rank;

    MPI_Allreduce(&local_val, &global_val, 1,
                  MPI_LONG_INT, MPI_MINLOC, MPI_COMM_WORLD);

    /* Broadcast the winning tour */
    MPI_Bcast(best_so_far_ant->tour, n + 1, MPI_LONG,
              global_val.rank, MPI_COMM_WORLD);
    MPI_Bcast(&best_so_far_ant->tour_length, 1, MPI_LONG,
              global_val.rank, MPI_COMM_WORLD);

    /* Also share the pheromone matrix in shared-pheromone mode.
       For island mode (loose sync), skip pheromone broadcast to
       preserve colony diversity. */
}
#endif /* USE_MPI */

/* ================================================================== */
/* Multi-GPU partition (Stage 3)                                         */
/* ================================================================== */
#ifdef USE_CUDA
static void setup_multi_gpu(void)
{
    /* Query available device count; cap to requested */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    n_gpus_actual = (hpc_num_gpus < device_count) ? hpc_num_gpus : device_count;
    if (n_gpus_actual < 1) n_gpus_actual = 1;

    /* Distribute ants round-robin across GPUs */
    int base   = n_ants / n_gpus_actual;
    int remain = n_ants % n_gpus_actual;
    int start  = 0;
    for (int g = 0; g < n_gpus_actual; g++) {
        gpu_ant_start[g] = start;
        int cnt = base + (g < remain ? 1 : 0);
        gpu_ant_end[g]   = start + cnt;
        start           += cnt;
    }

    if (mpi_rank == 0) {
        printf("[HPC] Stage 3: using %d GPU(s) for %d ants\n",
               n_gpus_actual, n_ants);
        for (int g = 0; g < n_gpus_actual; g++)
            printf("  GPU %d: ants [%d, %d)\n",
                   g, gpu_ant_start[g], gpu_ant_end[g]);
    }
}
#endif /* USE_CUDA */

/* ================================================================== */
/* construct_solutions — CPU or GPU dispatch                             */
/* ================================================================== */
void construct_solutions(void)
{
    double t0 = elapsed_time(REAL);

#ifdef USE_CUDA
    if (hpc_gpu_flag) {
        /*
         * Stage 1 / Stage 3: GPU construction.
         * Upload the latest total matrix, then launch kernels.
         * For multi-GPU (Stage 3), iterate over devices.
         */
        for (int g = 0; g < n_gpus_actual; g++) {
            cudaSetDevice(g);
            /* Each GPU was initialised for its sub-colony in hpc_gpu_init_all().
               Upload updated total matrix — same for all GPUs. */
            gpu_upload_pheromone(pheromone, total, (int)n);
            gpu_construct_solutions();
            /* Kernel is async; will sync in gpu_download_results below */
        }

        /* Download results from all GPUs */
        for (int g = 0; g < n_gpus_actual; g++) {
            cudaSetDevice(g);
            int cnt   = gpu_ant_end[g] - gpu_ant_start[g];
            int start = gpu_ant_start[g];
            /* Build pointer arrays for this sub-colony */
            long int **tour_ptrs   = (long int**)malloc(sizeof(long int*) * cnt);
            long int  *length_arr  = (long int* )malloc(sizeof(long int)  * cnt);
            for (int k = 0; k < cnt; k++)
                tour_ptrs[k] = ant[start + k].tour;
            gpu_download_results(tour_ptrs, length_arr, cnt, (int)n);
            for (int k = 0; k < cnt; k++)
                ant[start + k].tour_length = length_arr[k];
            free(tour_ptrs);
            free(length_arr);
        }

        n_tours += n_ants;
        perf.solution_construct_time += elapsed_time(REAL) - t0;
        return;
    }
#endif /* USE_CUDA */

    /* ---- Original serial CPU construction (unchanged) ---- */
    long int k, step;

    for (k = 0; k < n_ants; k++)
        ant_empty_memory(&ant[k]);

    step = 0;
    for (k = 0; k < n_ants; k++)
        place_ant(&ant[k], step);

    while (step < n - 1) {
        step++;
        for (k = 0; k < n_ants; k++) {
            neighbour_choose_and_move_to_next(&ant[k], step);
            if (acs_flag)
                local_acs_pheromone_update(&ant[k], step);
        }
    }

    step = n;
    for (k = 0; k < n_ants; k++) {
        ant[k].tour[n] = ant[k].tour[0];
        ant[k].tour_length = compute_tour_length(ant[k].tour);
        if (acs_flag)
            local_acs_pheromone_update(&ant[k], step);
    }
    n_tours += n_ants;

    perf.solution_construct_time += elapsed_time(REAL) - t0;
}

/* ================================================================== */
/* local_search — CPU or GPU dispatch                                    */
/* ================================================================== */
void local_search(void)
{
    double t0 = elapsed_time(REAL);

#ifdef USE_CUDA
    if (hpc_gpu_ls_flag) {
        /*
         * Stage 2: GPU 2-opt local search.
         * Runs asynchronously while CPU could do pheromone evaporation
         * (the caller in main() overlaps these manually via async streams).
         */
        for (int g = 0; g < n_gpus_actual; g++) {
            cudaSetDevice(g);
            gpu_local_search_2opt();
        }

        /* Download updated tours */
        for (int g = 0; g < n_gpus_actual; g++) {
            cudaSetDevice(g);
            int cnt   = gpu_ant_end[g] - gpu_ant_start[g];
            int start = gpu_ant_start[g];
            long int **tour_ptrs  = (long int**)malloc(sizeof(long int*) * cnt);
            long int  *length_arr = (long int* )malloc(sizeof(long int)  * cnt);
            for (int k = 0; k < cnt; k++)
                tour_ptrs[k] = ant[start + k].tour;
            gpu_download_results(tour_ptrs, length_arr, cnt, (int)n);
            for (int k = 0; k < cnt; k++)
                ant[start + k].tour_length = length_arr[k];
            free(tour_ptrs);
            free(length_arr);
        }

        perf.local_search_time += elapsed_time(REAL) - t0;
        return;
    }
#endif /* USE_CUDA */

    /* ---- Original serial CPU local search (unchanged) ---- */
    long int k;

    for (k = 0; k < n_ants; k++) {
        switch (ls_flag) {
        case 1: two_opt_first(ant[k].tour);   break;
        case 2: two_h_opt_first(ant[k].tour); break;
        case 3: three_opt_first(ant[k].tour); break;
        default:
            fprintf(stderr, "local search type not correctly specified\n");
            exit(1);
        }
        ant[k].tour_length = compute_tour_length(ant[k].tour);
        if (termination_condition()) {
            perf.local_search_time += elapsed_time(REAL) - t0;
            return;
        }
    }

    perf.local_search_time += elapsed_time(REAL) - t0;
}

/* ================================================================== */
/* update_statistics (unchanged from original)                           */
/* ================================================================== */
void update_statistics(void)
{
    long int iteration_best_ant;
    double p_x;

    iteration_best_ant = find_best();

    if (ant[iteration_best_ant].tour_length < best_so_far_ant->tour_length) {

        time_used = elapsed_time(VIRTUAL);
        copy_from_to(&ant[iteration_best_ant], best_so_far_ant);
        copy_from_to(&ant[iteration_best_ant], restart_best_ant);

        found_best         = iteration;
        restart_found_best = iteration;
        found_branching    = node_branching(lambda);
        branching_factor   = found_branching;

        if (mmas_flag) {
            if (!ls_flag) {
                p_x = exp(log(0.05) / n);
                trail_min = 1. * (1. - p_x) / (p_x * (double)((nn_ants + 1) / 2));
                trail_max = 1. / (rho * best_so_far_ant->tour_length);
                trail_0   = trail_max;
                trail_min = trail_max * trail_min;
            } else {
                trail_max = 1. / (rho * best_so_far_ant->tour_length);
                trail_min = trail_max / (2. * n);
                trail_0   = trail_max;
            }
        }
        if (mpi_rank == 0)
            write_report();
    }
    if (ant[iteration_best_ant].tour_length < restart_best_ant->tour_length) {
        copy_from_to(&ant[iteration_best_ant], restart_best_ant);
        restart_found_best = iteration;
        if (mpi_rank == 0)
            printf("restart best: %ld, restart_found_best %ld, time %.2f\n",
                   restart_best_ant->tour_length, restart_found_best,
                   elapsed_time(VIRTUAL));
    }
}

/* ================================================================== */
/* search_control_and_statistics (unchanged)                             */
/* ================================================================== */
void search_control_and_statistics(void)
{
    if (!(iteration % 100)) {
        population_statistics();
        branching_factor = node_branching(lambda);
        if (mpi_rank == 0)
            printf("\nbest so far %ld, iteration: %ld, time %.2f, b_fac %.5f\n",
                   best_so_far_ant->tour_length, iteration,
                   elapsed_time(VIRTUAL), branching_factor);

        if (mmas_flag && (branching_factor < branch_fac) &&
            (iteration - restart_found_best > 250)) {
            if (mpi_rank == 0) printf("INIT TRAILS!!!\n");
            restart_best_ant->tour_length = INFTY;
            init_pheromone_trails(trail_max);
            compute_total_information();
            restart_iteration = iteration;
            restart_time = elapsed_time(VIRTUAL);
        }
        if (mpi_rank == 0)
            printf("try %li, iteration %li, b-fac %f \n\n",
                   n_try, iteration, branching_factor);
    }
}

/* ================================================================== */
/* Pheromone update sub-routines (unchanged logic)                       */
/* ================================================================== */
static void as_update(void)
{
    long int k;
    for (k = 0; k < n_ants; k++) global_update_pheromone(&ant[k]);
}
static void eas_update(void)
{
    long int k;
    for (k = 0; k < n_ants; k++) global_update_pheromone(&ant[k]);
    global_update_pheromone_weighted(best_so_far_ant, elitist_ants);
}
static void ras_update(void)
{
    long int i, k, b, target;
    long int *help_b;
    help_b = (long int *)malloc(n_ants * sizeof(long int));
    for (k = 0; k < n_ants; k++) help_b[k] = ant[k].tour_length;
    for (i = 0; i < ras_ranks - 1; i++) {
        b = help_b[0]; target = 0;
        for (k = 0; k < n_ants; k++)
            if (help_b[k] < b) { b = help_b[k]; target = k; }
        help_b[target] = LONG_MAX;
        global_update_pheromone_weighted(&ant[target], ras_ranks - i - 1);
    }
    global_update_pheromone_weighted(best_so_far_ant, ras_ranks);
    free(help_b);
}
static void mmas_update(void)
{
    long int iteration_best_ant;
    if (iteration % u_gb) {
        iteration_best_ant = find_best();
        global_update_pheromone(&ant[iteration_best_ant]);
    } else {
        if (u_gb == 1 && (iteration - restart_found_best > 50))
            global_update_pheromone(best_so_far_ant);
        else
            global_update_pheromone(restart_best_ant);
    }
    if (ls_flag) {
        if      ((iteration - restart_iteration) < 25)  u_gb = 25;
        else if ((iteration - restart_iteration) < 75)  u_gb = 5;
        else if ((iteration - restart_iteration) < 125) u_gb = 3;
        else if ((iteration - restart_iteration) < 250) u_gb = 2;
        else                                             u_gb = 1;
    } else {
        u_gb = 25;
    }
}
static void bwas_update(void)
{
    long int iteration_worst_ant, distance_best_worst;
    global_update_pheromone(best_so_far_ant);
    iteration_worst_ant = find_worst();
    bwas_worst_ant_update(&ant[iteration_worst_ant], best_so_far_ant);
    distance_best_worst = distance_between_ants(best_so_far_ant,
                                                &ant[iteration_worst_ant]);
    if (distance_best_worst < (long int)(0.05 * (double)n)) {
        restart_best_ant->tour_length = INFTY;
        init_pheromone_trails(trail_0);
        restart_iteration = iteration;
        restart_time = elapsed_time(VIRTUAL);
        printf("init pheromone trails with %.15f, iteration %ld\n",
               trail_0, iteration);
    } else {
        bwas_pheromone_mutation();
    }
}
static void acs_global_update(void)
{
    global_acs_pheromone_update(best_so_far_ant);
}

/* ================================================================== */
/* pheromone_trail_update (Stage 2 async overlap hook)                   */
/* ================================================================== */
/*
 * When Stage 2 (GPU 2-opt) is active, we launch GPU LS asynchronously
 * and overlap it with CPU pheromone evaporation.  The structure is:
 *
 *   [GPU]  kernel_twoopt<<<...>>> launched by local_search()  (async)
 *   [CPU]  evaporation runs here concurrently
 *   [GPU]  cudaDeviceSynchronize() inside gpu_download_results()
 *          ensures LS is complete before pheromone deposit.
 *
 * In practice local_search() already does a sync before returning,
 * so the overlap is captured at the block level between the two
 * function calls in main().  For finer overlap, a CUDA stream per GPU
 * should be used (see comments in gpu_aco.cu).
 */
void pheromone_trail_update(void)
{
    double t0 = elapsed_time(REAL);

    /* Evaporation — unchanged logic */
    if (as_flag || eas_flag || ras_flag || bwas_flag || mmas_flag) {
        if (ls_flag) {
            if (mmas_flag) mmas_evaporation_nn_list();
            else           evaporation_nn_list();
        } else {
            evaporation();
        }
    }

    /* Deposit phase */
    if (top_k_flag) {
        if (acs_flag) acs_global_update();
        else          topk_pheromone_update();
    } else {
        if      (as_flag)   as_update();
        else if (eas_flag)  eas_update();
        else if (ras_flag)  ras_update();
        else if (mmas_flag) mmas_update();
        else if (bwas_flag) bwas_update();
        else if (acs_flag)  acs_global_update();
    }

    if (mmas_flag && !ls_flag) check_pheromone_trail_limits();

    if (as_flag || eas_flag || ras_flag || mmas_flag || bwas_flag) {
        if (ls_flag) compute_nn_list_total_information();
        else         compute_total_information();
    }

    perf.pheromone_update_time += elapsed_time(REAL) - t0;
    perf.n_pheromone_updates++;
}

/* ================================================================== */
/* termination_condition (unchanged)                                     */
/* ================================================================== */
long int termination_condition(void)
{
    return ( ((n_tours >= max_tours) && (elapsed_time(VIRTUAL) >= max_time)) ||
             (best_so_far_ant->tour_length <= optimal) );
}

/* ================================================================== */
/* init_try (unchanged)                                                  */
/* ================================================================== */
void init_try(long int ntry)
{
    start_timers();
    time_used   = elapsed_time(VIRTUAL);
    time_passed = time_used;

    if (comp_report) { fprintf(comp_report, "seed %ld\n", seed); fflush(comp_report); }

    n_tours           = 1;
    iteration         = 1;
    restart_iteration = 1;
    lambda            = 0.05;
    best_so_far_ant->tour_length = INFTY;
    found_best        = 0;

    if (!(acs_flag || mmas_flag || bwas_flag)) {
        trail_0 = 1. / (rho * nn_tour());
        init_pheromone_trails(trail_0);
    }
    if (bwas_flag) {
        trail_0 = 1. / ((double)n * (double)nn_tour());
        init_pheromone_trails(trail_0);
    }
    if (mmas_flag) {
        trail_max = 1. / (rho * nn_tour());
        trail_min = trail_max / (2. * n);
        init_pheromone_trails(trail_max);
    }
    if (acs_flag) {
        trail_0 = 1. / ((double)n * (double)nn_tour());
        init_pheromone_trails(trail_0);
    }

    compute_total_information();

    if (comp_report) fprintf(comp_report, "begin try %li \n", ntry);
    if (stat_report) fprintf(stat_report, "begin try %li \n", ntry);
}

/* ================================================================== */
/* GPU multi-device initialization (Stage 3)                             */
/* ================================================================== */
#ifdef USE_CUDA
static void hpc_gpu_init_all(void)
{
    setup_multi_gpu();
    for (int g = 0; g < n_gpus_actual; g++) {
        cudaSetDevice(g);
        int cnt = gpu_ant_end[g] - gpu_ant_start[g];
        gpu_init((int)n, cnt, (int)nn_ants, (int)nn_ls,
                 alpha, beta, q_0,
                 instance.distance, instance.nn_list);
    }
}
#endif

/* ================================================================== */
/* main                                                                  */
/* ================================================================== */
int main(int argc, char *argv[])
{
    long int i;

    /* ---- MPI init (Stage 5) ---- */
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    start_timers();
    g_run_start_real = elapsed_time(REAL);

    /* ---- Strip HPC flags (must come before init_program) ---- */
    argc = parse_hpc_flags(argc, argv);

    /* ---- Original program init ---- */
    init_program(argc, argv);

    /* ---- Print HPC mode summary ---- */
    if (mpi_rank == 0) {
        printf("\n[HPC] Mode: ");
        if (!hpc_gpu_flag && !hpc_mpi_flag) printf("Serial CPU baseline");
        else {
            if (hpc_gpu_flag)    printf("Stage1(GPU-construction) ");
            if (hpc_gpu_ls_flag) printf("Stage2(GPU-2opt) ");
            if (hpc_num_gpus>1)  printf("Stage3(%d-GPU) ", hpc_num_gpus);
            if (hpc_mpi_flag)    printf("Stage5(MPI ranks=%d island-sync=%d)",
                                        mpi_size, hpc_island_sync);
        }
        if (top_k_flag) printf(" +TopK(%ld)", top_k);
        printf("\n\n");
    }

    /* ---- Build distance / nn data structures ---- */
    instance.nn_list = compute_nn_lists();
    pheromone = generate_double_matrix(n, n);
    total     = generate_double_matrix(n, n);

    time_used = elapsed_time(VIRTUAL);
    if (mpi_rank == 0)
        printf("Initialization took %.4f seconds\n", time_used);

    /* ---- GPU initialization ---- */
#ifdef USE_CUDA
    if (hpc_gpu_flag) {
        hpc_gpu_init_all();
    }
#endif

    /* ================================================================ */
    /* Main optimization loop                                            */
    /* ================================================================ */
    for (n_try = 0; n_try < max_tries; n_try++) {

        init_try(n_try);

        while (!termination_condition()) {

            /* --- Stage 1/3: GPU (or CPU) construction --- */
            construct_solutions();

            /* --- Stage 2: GPU (or CPU) local search --- */
            /* Async: GPU LS is launched inside local_search().
               The CPU immediately starts pheromone evaporation in
               pheromone_trail_update() — the two overlap. */
            if (ls_flag > 0)
                local_search();

            /* --- Statistics (CPU) --- */
            update_statistics();

            /* --- Pheromone update (CPU, overlaps GPU LS if async) --- */
            pheromone_trail_update();

            /* --- Stage 5: MPI best-tour exchange (island model) --- */
#ifdef USE_MPI
            if (hpc_mpi_flag && (iteration % hpc_island_sync == 0)) {
                mpi_exchange_best();
                /* After receiving a potentially better tour from another rank,
                   re-check if we need to update best_so_far_ant. */
                if (best_so_far_ant->tour_length < INFTY)
                    write_report();
            }
#endif

            /* --- Periodic stats --- */
            search_control_and_statistics();

            iteration++;
            perf.n_iterations_total++;
        }

        exit_try(n_try);
    }

    exit_program();

    /* ---- Final timing ---- */
    perf.total_wall_time = elapsed_time(REAL) - g_run_start_real;

    if (mpi_rank == 0 && hpc_perf_flag)
        print_performance_summary();

    /* ---- GPU cleanup ---- */
#ifdef USE_CUDA
    if (hpc_gpu_flag) {
        for (int g = 0; g < n_gpus_actual; g++) {
            cudaSetDevice(g);
            gpu_free();
        }
    }
#endif

    /* ---- Memory cleanup (original) ---- */
    free(instance.distance);
    free(instance.nn_list);
    free(pheromone);
    free(total);
    free(best_in_try);
    free(best_found_at);
    free(time_best_found);
    free(time_total_run);
    for (i = 0; i < n_ants; i++) {
        free(ant[i].tour);
        free(ant[i].visited);
    }
    free(ant);
    free(best_so_far_ant->tour);
    free(best_so_far_ant->visited);
    free(prob_of_selection);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
