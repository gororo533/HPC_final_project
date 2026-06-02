#!/usr/bin/env python3
"""
run_experiments.py — multi-algorithm ACO comparison runner

Usage:
    python3 run_experiments.py [options]

Options:
    --tsp FILE         TSP instance file (required)
    --tries N          number of independent runs per experiment  [default: 3]
    --time T           time limit per run in seconds              [default: 60]
    --topk K           also run top-k variant with this k value   [default: 5]
    --algos LIST       comma-separated subset of: as,eas,ras,mmas,bwas,acs
                       default: all six
    --binary PATH      path to acotsp binary                      [default: ./acotsp]
    --optimal N        known optimal tour length (overrides ans.txt lookup)
    --ans FILE         path to ans.txt file                       [default: ./ans.txt]
    --out FILE         output log file                            [default: results_<instance>.log]
    --ls N             local search: 0=none 1=2opt 2=2.5opt 3=3opt [default: 3]
    --quiet            suppress acotsp stdout

Runs each enabled algorithm in two modes (original + top-k) and writes a
structured log with timing, iteration count, best distance, and gap to optimum.

Example:
    python3 run_experiments.py --tsp pcb442.tsp --tries 3 --time 60 --topk 5 \\
            --algos mmas,acs --ans ans.txt
"""

import argparse
import subprocess
import re
import sys
import os
import time
import datetime


# ── ans.txt parser ────────────────────────────────────────────────────────────

def load_optima(path):
    """Return dict: instance_name (lowercase, no extension) -> optimal int."""
    optima = {}
    if not os.path.isfile(path):
        return optima
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # format: "berlin52 : 7542" or "dsj1000 : 18659688 (EUC_2D)"
            m = re.match(r'^(\S+)\s*:\s*(\d+)', line)
            if m:
                key = m.group(1).lower()
                optima[key] = int(m.group(2))
    return optima


def lookup_optimal(optima, tsp_file):
    """Match tsp filename stem against optima dict."""
    stem = os.path.splitext(os.path.basename(tsp_file))[0].lower()
    return optima.get(stem, None)


# ── acotsp output parser ──────────────────────────────────────────────────────

def parse_acotsp_output(text):
    """
    Extract per-try results from acotsp stdout.
    Returns list of dicts: {best, found_at_iter, found_at_time, total_time}
    """
    tries = []
    # Line: "try 0, Best 50778, found at iteration 42, found at time 1.234567"
    for m in re.finditer(
            r'try\s+(\d+),\s+Best\s+(\d+),\s+found at iteration\s+(\d+),\s+found at time\s+([\d.]+)',
            text):
        tries.append({
            'try':           int(m.group(1)),
            'best':          int(m.group(2)),
            'found_at_iter': int(m.group(3)),
            'found_at_time': float(m.group(4)),
        })

    # Total run time per try comes from exit_program summary lines
    # "t_avgtotal = 60.001234"
    m_avgtotal = re.search(r't_avgtotal\s*=\s*([\d.]+)', text)
    avg_total = float(m_avgtotal.group(1)) if m_avgtotal else None

    # Wall time from our performance summary
    m_wall = re.search(r'Total wall time\s*:\s*([\d.]+)', text)
    wall = float(m_wall.group(1)) if m_wall else None

    # Total iterations from perf summary
    m_iters = re.search(r'Total iterations\s*:\s*(\d+)', text)
    total_iters = int(m_iters.group(1)) if m_iters else None

    return {
        'tries':        tries,
        'avg_total_time': avg_total,
        'wall_time':    wall,
        'total_iters':  total_iters,
    }


# ── single experiment runner ──────────────────────────────────────────────────

def run_one(binary, tsp_file, algo_flag, tries, time_limit, ls, topk=None, quiet=False):
    """
    Run acotsp for one (algorithm, mode) combination.
    topk=None  → original mode
    topk=N     → sparse top-k mode with k=N
    Returns (stdout_text, stderr_text, wall_seconds, returncode).
    """
    cmd = [
        binary,
        f'--{algo_flag}',
        '-i', tsp_file,
        '-r', str(tries),
        '-t', str(time_limit),
        '-l', str(ls),
    ]
    if topk is not None:
        cmd += ['--topk', str(topk)]
    if quiet:
        cmd += ['--quiet']

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    return result.stdout, result.stderr, elapsed, result.returncode


# ── log writer ────────────────────────────────────────────────────────────────

def format_gap(best, optimal):
    if optimal is None or optimal <= 0:
        return 'N/A'
    gap = (best - optimal) / optimal * 100.0
    return f'{gap:+.3f}%'


def write_log(log_path, instance_name, optimal, algo_results):
    """
    algo_results: list of dicts from run_all_experiments()
    """
    lines = []
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines.append('=' * 72)
    lines.append(f'  ACO/TSP EXPERIMENT RESULTS')
    lines.append(f'  Instance : {instance_name}')
    lines.append(f'  Optimal  : {optimal if optimal is not None else "unknown"}')
    lines.append(f'  Generated: {ts}')
    lines.append('=' * 72)
    lines.append('')

    # Header
    col = '{:<12} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10}'
    lines.append(col.format('Algorithm', 'Mode', 'Best', 'Gap', 'AvgIter',
                             'AvgTime(s)', 'WallTime(s)'))
    lines.append('-' * 72)

    for r in algo_results:
        algo    = r['algo']
        mode    = 'top-k(k={})'.format(r['topk']) if r['topk'] else 'original'
        parsed  = r['parsed']
        rc      = r['returncode']

        if rc != 0 or not parsed['tries']:
            lines.append(col.format(algo, mode, 'FAILED', '-', '-', '-', '-'))
            continue

        best_vals     = [t['best'] for t in parsed['tries']]
        iter_vals     = [t['found_at_iter'] for t in parsed['tries']]
        time_vals     = [t['found_at_time'] for t in parsed['tries']]

        best_overall  = min(best_vals)
        avg_iter      = sum(iter_vals) / len(iter_vals)
        avg_time      = sum(time_vals) / len(time_vals)
        wall          = parsed['wall_time'] or r['wall_seconds']
        gap           = format_gap(best_overall, optimal)

        lines.append(col.format(
            algo, mode,
            str(best_overall),
            gap,
            f'{avg_iter:.0f}',
            f'{avg_time:.2f}',
            f'{wall:.2f}',
        ))

        # Per-try detail
        for t in parsed['tries']:
            g = format_gap(t['best'], optimal)
            lines.append(
                f"    try {t['try']}: best={t['best']}  gap={g}"
                f"  found_iter={t['found_at_iter']}  found_time={t['found_at_time']:.3f}s"
            )

    lines.append('')
    lines.append('=' * 72)

    # Comparison table: original vs top-k, per algo
    lines.append('')
    lines.append('  ORIGINAL vs TOP-K COMPARISON')
    lines.append('-' * 72)
    col2 = '{:<12} {:>10} {:>10} {:>10} {:>10} {:>12}'
    lines.append(col2.format('Algorithm', 'Orig.Best', 'TopK.Best',
                              'Orig.Gap', 'TopK.Gap', 'SpeedupIter'))
    lines.append('-' * 72)

    # Group by algo
    algos_seen = []
    by_algo = {}
    for r in algo_results:
        a = r['algo']
        if a not in by_algo:
            by_algo[a] = {}
            algos_seen.append(a)
        key = 'topk' if r['topk'] else 'orig'
        by_algo[a][key] = r

    for a in algos_seen:
        orig = by_algo[a].get('orig')
        topk = by_algo[a].get('topk')
        if orig is None or topk is None:
            continue

        def best_for(r):
            if not r['parsed']['tries']:
                return None
            return min(t['best'] for t in r['parsed']['tries'])

        def avg_iters_for(r):
            if not r['parsed']['total_iters']:
                return None
            return r['parsed']['total_iters']

        ob = best_for(orig)
        tb = best_for(topk)
        oi = avg_iters_for(orig)
        ti = avg_iters_for(topk)

        speedup = f"{ti/oi:.2f}x" if (oi and ti and oi > 0) else 'N/A'
        lines.append(col2.format(
            a,
            str(ob) if ob else 'N/A',
            str(tb) if tb else 'N/A',
            format_gap(ob, optimal) if ob else 'N/A',
            format_gap(tb, optimal) if tb else 'N/A',
            speedup,
        ))

    lines.append('')
    lines.append('=' * 72)
    text = '\n'.join(lines) + '\n'

    with open(log_path, 'w') as f:
        f.write(text)

    return text


# ── main ──────────────────────────────────────────────────────────────────────

ALL_ALGOS = ['as', 'eas', 'ras', 'mmas', 'bwas', 'acs']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tsp',     required=True,         help='TSP instance file')
    p.add_argument('--tries',   type=int, default=3,   help='runs per experiment')
    p.add_argument('--time',    type=float, default=60, help='seconds per run')
    p.add_argument('--topk',    type=int, default=5,   help='k for top-k variant')
    p.add_argument('--algos',   default='as,eas,ras,mmas,bwas,acs',
                                help='comma-separated algo list')
    p.add_argument('--binary',  default='./acotsp',    help='acotsp binary path')
    p.add_argument('--optimal', type=int, default=None, help='known optimal')
    p.add_argument('--ans',     default='ans.txt',     help='ans.txt path')
    p.add_argument('--out',     default=None,          help='output log file')
    p.add_argument('--ls',      type=int, default=3,   help='local search type')
    p.add_argument('--quiet',   action='store_true',   help='suppress acotsp stdout')
    args = p.parse_args()

    if not os.path.isfile(args.tsp):
        print(f'ERROR: TSP file not found: {args.tsp}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.binary):
        print(f'ERROR: binary not found: {args.binary}', file=sys.stderr)
        sys.exit(1)

    algos = [a.strip().lower() for a in args.algos.split(',')]
    for a in algos:
        if a not in ALL_ALGOS:
            print(f'ERROR: unknown algorithm "{a}". Choose from: {ALL_ALGOS}', file=sys.stderr)
            sys.exit(1)

    optima = load_optima(args.ans)
    optimal = args.optimal or lookup_optimal(optima, args.tsp)
    instance_name = os.path.splitext(os.path.basename(args.tsp))[0]

    log_path = args.out or f'results_{instance_name}.log'

    print(f'\n{"="*60}')
    print(f'  Instance : {instance_name}')
    print(f'  Optimal  : {optimal or "unknown"}')
    print(f'  Algos    : {algos}')
    print(f'  Tries    : {args.tries}  Time: {args.time}s  Top-k: {args.topk}')
    print(f'  Log      : {log_path}')
    print(f'{"="*60}\n')

    results = []
    total_experiments = len(algos) * 2  # original + topk
    idx = 0

    for algo in algos:
        for topk in [None, args.topk]:
            idx += 1
            mode_str = f'top-k(k={topk})' if topk else 'original'
            print(f'[{idx}/{total_experiments}] {algo.upper()} / {mode_str} ... ', end='', flush=True)

            stdout, stderr, wall, rc = run_one(
                args.binary, args.tsp, algo,
                args.tries, args.time, args.ls,
                topk=topk, quiet=args.quiet
            )

            parsed = parse_acotsp_output(stdout)

            if rc != 0:
                print(f'FAILED (rc={rc})')
                if stderr:
                    print(f'    stderr: {stderr[:200]}')
            else:
                bests = [t['best'] for t in parsed['tries']] if parsed['tries'] else []
                best_str = str(min(bests)) if bests else '?'
                gap_str = format_gap(min(bests), optimal) if bests else 'N/A'
                iters = parsed['total_iters'] or '?'
                print(f'best={best_str}  gap={gap_str}  iters={iters}  wall={wall:.1f}s')

            results.append({
                'algo':         algo,
                'topk':         topk,
                'parsed':       parsed,
                'wall_seconds': wall,
                'returncode':   rc,
                'stdout':       stdout,
                'stderr':       stderr,
            })

    log_text = write_log(log_path, instance_name, optimal, results)
    print(f'\n--- Results written to {log_path} ---\n')
    print(log_text)


if __name__ == '__main__':
    main()