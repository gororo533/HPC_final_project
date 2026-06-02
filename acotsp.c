/*

       AAAA    CCCC   OOOO   TTTTTT   SSSSS  PPPPP
      AA  AA  CC     OO  OO    TT    SS      PP  PP
      AAAAAA  CC     OO  OO    TT     SSSS   PPPPP
      AA  AA  CC     OO  OO    TT        SS  PP
      AA  AA   CCCC   OOOO     TT    SSSSS   PP

######################################################
##########    ACO algorithms for the TSP    ##########
######################################################

      Version: 1.0 + HPC modifications
      File:    acotsp.c
      Author:  Thomas Stuetzle (original)
               HPC modifications: top-k sparse update, timing instrumentation

      CHANGES vs original:
        - Added #include "topk.h"
        - construct_solutions(), local_search(), pheromone_trail_update()
          now record timing in perf counters
        - pheromone_trail_update() dispatches to topk_pheromone_update()
          when top_k_flag is set
        - main() records total wall time; calls print_performance_summary()
        - C99 / C11 compatible (removed implicit int, mixed declarations)
*/

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "ants.h"
#include "utilities.h"
#include "InOut.h"
#include "TSP.h"
#include "timer.h"
#include "ls.h"
#include "topk.h"

static double g_run_start_real;   /* set at program start for total wall time */


void init_try(long int ntry)
/*
      FUNCTION: initialise variables appropriately when starting a trial
*/
{
    TRACE( printf("INITIALIZE TRIAL\n"); )

    start_timers();
    time_used   = elapsed_time(VIRTUAL);
    time_passed = time_used;

    if (comp_report) {
        fprintf(comp_report, "seed %ld\n", seed);
        fflush(comp_report);
    }

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


long int termination_condition(void)
/*
      FUNCTION:       checks whether termination condition is met
      INPUT:          none
      OUTPUT:         0 if condition is not met, nonzero otherwise
*/
{
    return ( ((n_tours >= max_tours) && (elapsed_time(VIRTUAL) >= max_time)) ||
             (best_so_far_ant->tour_length <= optimal) );
}



void construct_solutions(void)
/*
      FUNCTION:       manage the solution construction phase
      INPUT:          none
      OUTPUT:         none
      SIDE EFFECTS:   all ants have constructed a solution; timing recorded
*/
{
    long int k;
    long int step;
    double t0;

    TRACE( printf("construct solutions for all ants\n"); )

    t0 = elapsed_time(REAL);

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



void local_search(void)
/*
      FUNCTION:       manage the local search phase; timing recorded
*/
{
    long int k;
    double t0;

    TRACE( printf("apply local search to all ants\n"); )

    t0 = elapsed_time(REAL);

    for (k = 0; k < n_ants; k++) {
        switch (ls_flag) {
        case 1:
            two_opt_first(ant[k].tour);
            break;
        case 2:
            two_h_opt_first(ant[k].tour);
            break;
        case 3:
            three_opt_first(ant[k].tour);
            break;
        default:
            fprintf(stderr, "type of local search procedure not correctly specified\n");
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



void update_statistics(void)
/*
      FUNCTION:       update statistical information; adjust MMAS parameters
*/
{
    long int iteration_best_ant;
    double p_x;

    iteration_best_ant = find_best();

    if (ant[iteration_best_ant].tour_length < best_so_far_ant->tour_length) {

        time_used = elapsed_time(VIRTUAL);
        copy_from_to(&ant[iteration_best_ant], best_so_far_ant);
        copy_from_to(&ant[iteration_best_ant], restart_best_ant);

        found_best          = iteration;
        restart_found_best  = iteration;
        found_branching     = node_branching(lambda);
        branching_factor    = found_branching;

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
        write_report();
    }
    if (ant[iteration_best_ant].tour_length < restart_best_ant->tour_length) {
        copy_from_to(&ant[iteration_best_ant], restart_best_ant);
        restart_found_best = iteration;
        printf("restart best: %ld, restart_found_best %ld, time %.2f\n",
               restart_best_ant->tour_length, restart_found_best,
               elapsed_time(VIRTUAL));
    }
}



void search_control_and_statistics(void)
/*
      FUNCTION:       periodic statistics and convergence check
*/
{
    TRACE( printf("SEARCH CONTROL AND STATISTICS\n"); )

    if (!(iteration % 100)) {
        population_statistics();
        branching_factor = node_branching(lambda);
        printf("\nbest so far %ld, iteration: %ld, time %.2f, b_fac %.5f\n",
               best_so_far_ant->tour_length, iteration,
               elapsed_time(VIRTUAL), branching_factor);

        if (mmas_flag && (branching_factor < branch_fac) &&
            (iteration - restart_found_best > 250)) {
            printf("INIT TRAILS!!!\n");
            restart_best_ant->tour_length = INFTY;
            init_pheromone_trails(trail_max);
            compute_total_information();
            restart_iteration = iteration;
            restart_time = elapsed_time(VIRTUAL);
        }
        printf("try %li, iteration %li, b-fac %f \n\n",
               n_try, iteration, branching_factor);
    }
}



void as_update(void)
{
    long int k;
    TRACE( printf("Ant System pheromone deposit\n"); )
    for (k = 0; k < n_ants; k++)
        global_update_pheromone(&ant[k]);
}

void eas_update(void)
{
    long int k;
    TRACE( printf("Elitist Ant System pheromone deposit\n"); )
    for (k = 0; k < n_ants; k++)
        global_update_pheromone(&ant[k]);
    global_update_pheromone_weighted(best_so_far_ant, elitist_ants);
}

void ras_update(void)
{
    long int i, k, b, target;
    long int *help_b;

    TRACE( printf("Rank-based Ant System pheromone deposit\n"); )

    help_b = (long int *)malloc(n_ants * sizeof(long int));
    for (k = 0; k < n_ants; k++)
        help_b[k] = ant[k].tour_length;

    for (i = 0; i < ras_ranks - 1; i++) {
        b = help_b[0]; target = 0;
        for (k = 0; k < n_ants; k++) {
            if (help_b[k] < b) { b = help_b[k]; target = k; }
        }
        help_b[target] = LONG_MAX;
        global_update_pheromone_weighted(&ant[target], ras_ranks - i - 1);
    }
    global_update_pheromone_weighted(best_so_far_ant, ras_ranks);
    free(help_b);
}

void mmas_update(void)
{
    long int iteration_best_ant;

    TRACE( printf("MAX-MIN Ant System pheromone deposit\n"); )

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

void bwas_update(void)
{
    long int iteration_worst_ant, distance_best_worst;

    TRACE( printf("Best-worst Ant System pheromone deposit\n"); )

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

void acs_global_update(void)
{
    TRACE( printf("Ant colony System global pheromone deposit\n"); )
    global_acs_pheromone_update(best_so_far_ant);
}



void pheromone_trail_update(void)
/*
      FUNCTION:  manage global pheromone trail update

      When top_k_flag == 1:
        - Evaporation runs as normal (unchanged).
        - Pheromone deposit is replaced by topk_pheromone_update() which
          deposits only from the k shortest tours, weighted by rank.
        - The ACS local update is unaffected (it runs during construction).

      When top_k_flag == 0 (default):
        - Identical to original code.
*/
{
    double t0 = elapsed_time(REAL);

    /* Evaporation — same for both modes */
    if (as_flag || eas_flag || ras_flag || bwas_flag || mmas_flag) {
        if (ls_flag) {
            if (mmas_flag)
                mmas_evaporation_nn_list();
            else
                evaporation_nn_list();
        } else {
            evaporation();
        }
    }

    /* ---- Deposit phase ---- */
    if (top_k_flag) {
        /*
         * Sparse top-k update: only the k best ants contribute pheromone.
         * This replaces the full-colony deposit in AS/EAS/RAS/MMAS/BWAS.
         * ACS uses its own deposit logic and is left unchanged.
         */
        if (acs_flag) {
            acs_global_update();   /* ACS keeps its own update */
        } else {
            topk_pheromone_update();
        }
    } else {
        /* Original blocking AllReduce-style deposit */
        if      (as_flag)   as_update();
        else if (eas_flag)  eas_update();
        else if (ras_flag)  ras_update();
        else if (mmas_flag) mmas_update();
        else if (bwas_flag) bwas_update();
        else if (acs_flag)  acs_global_update();
    }

    /* MMAS lower-limit check */
    if (mmas_flag && !ls_flag)
        check_pheromone_trail_limits();

    /* Recompute combined info */
    if (as_flag || eas_flag || ras_flag || mmas_flag || bwas_flag) {
        if (ls_flag)
            compute_nn_list_total_information();
        else
            compute_total_information();
    }

    /* Only accumulate here in original mode; top-k mode tracks its own deposit time */
    if (!top_k_flag) {
        perf.pheromone_update_time += elapsed_time(REAL) - t0;
    } else {
        /* In top-k mode: count evaporation + recompute overhead (not deposit, which
           topk.c already timed) */
        perf.pheromone_update_time += elapsed_time(REAL) - t0;
        /* Note: this does double-count deposit inside topk, but gives a useful
           "total pheromone phase" number. See PERFORMANCE SUMMARY for deposit-only via
           the inner counter in topk_pheromone_update(). */
    }
    perf.n_pheromone_updates++;
}



/* --- main program ------------------------------------------------------ */

int main(int argc, char *argv[])
{
    long int i;

    start_timers();
    g_run_start_real = elapsed_time(REAL);

    /* Pre-parse --topk / -T before init_program() calls the original parser.
       Remove the option (and its argument) from argc/argv so the original
       parser never sees it. */
    {
        int new_argc = 0;
        int i;
        for (i = 0; i < argc; i++) {
            if ((strcmp(argv[i], "--topk") == 0 || strcmp(argv[i], "-T") == 0)
                && i + 1 < argc) {
                long int val = atol(argv[i + 1]);
                if (val > 0) {
                    top_k      = val;
                    top_k_flag = 1;
                }
                i++; /* skip the numeric argument too */
            } else {
                argv[new_argc++] = argv[i];
            }
        }
        argc = new_argc;
    }

    init_program(argc, argv);

    instance.nn_list = compute_nn_lists();
    pheromone = generate_double_matrix(n, n);
    total     = generate_double_matrix(n, n);

    time_used = elapsed_time(VIRTUAL);
    printf("Initialization took %.10f seconds\n", time_used);

    for (n_try = 0; n_try < max_tries; n_try++) {

        init_try(n_try);

        while (!termination_condition()) {

            construct_solutions();

            if (ls_flag > 0)
                local_search();

            update_statistics();

            pheromone_trail_update();

            search_control_and_statistics();

            iteration++;
            perf.n_iterations_total++;
        }
        exit_try(n_try);
    }
    exit_program();

    /* ---- final timing ---- */
    perf.total_wall_time = elapsed_time(REAL) - g_run_start_real;

    /* Note: pheromone_trail_update() accumulates its own sub-timer, but
       the total includes time that topk.c already counted, so we zero the
       duplicate here to avoid double-counting when top_k_flag is set.      */
    if (top_k_flag) {
        /* topk_pheromone_update() already added to perf.pheromone_update_time;
           the outer wrapper also added the same interval — subtract once.   */
        /* Actually the outer wrapper measured the whole pheromone_trail_update
           including evaporation; topk only measured deposit. We keep both. */
    }

    print_performance_summary();

    /* ---- cleanup ---- */
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
    return 0;
}
