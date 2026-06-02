# ACOTSP HPC — Design Document & Usage Guide

## File Overview

| File | Role |
|---|---|
| `acotsp_hpc.c` | Modified main driver — replaces `acotsp.c` |
| `gpu_aco.cu` | CUDA kernels: Stage 1 construction + Stage 2 2-opt |
| `gpu_aco.h` | Host API for the CUDA kernels |
| `Makefile.hpc` | Build targets for each stage |
| `run_hpc.sh` | SLURM job script (2 nodes × 4 GPU = final config) |
| `run_stage_test.sh` | Per-stage test job scripts for incremental validation |
| `ants.c/h` | Unchanged — pheromone logic, CPU fallbacks |
| `topk.c/h` | Unchanged — sparse top-k pheromone update |
| (all other .c/.h) | Completely unchanged from original |

---

## Architecture by Stage

### Stage 0 — Serial CPU (baseline, always present)

No changes to original logic.  Used for correctness validation.

Binary: `acotsp_serial`  
Build: `make -f Makefile.hpc serial`

### Stage 1 — GPU Ant Construction (`--gpu`)

**What moves to GPU:** `construct_solutions()` — the inner loop
`neighbour_choose_and_move_to_next()` for all ants simultaneously.

**Parallelism:** 1 CUDA thread = 1 ant.  With 256 ants, 256 threads
run concurrently.  The distance matrix, nn_list, and total matrix are
all in GPU global memory.  The PRNG is a per-thread xorshift64.

**CPU keeps:** pheromone update, statistics, MMAS trail-limit checks.

**Upload cost per iteration:** the `total` matrix (n×n doubles) is
re-uploaded each iteration because the CPU updates pheromone.
For n=1000 that is ~8 MB — negligible vs construction time.

Expected speedup over Stage 0: **4–8×** on construction.

### Stage 2 — GPU 2-opt (`--gpu-ls`, implies `--gpu`)

**What moves to GPU:** `local_search()` — 2-opt with nn_list restriction.

**Parallelism:** 1 CUDA block = 1 ant.  Threads within the block
cooperate to scan edge-swap candidates (one thread per city-i).
The tour and position-inverse are kept in shared memory for fast
random access.

**Async overlap:** GPU 2-opt is launched asynchronously; the CPU
immediately starts pheromone evaporation (`evaporation()` /
`evaporation_nn_list()`).  Download + device sync happen when the
CPU needs tour results back (before pheromone deposit).

**Limitation:** shared-memory 2-opt supports n ≤ 4096.  For larger
instances the kernel falls back to global-memory access (still
parallel, slightly slower).

Expected speedup vs CPU 2-opt: **3–6×** for large n.

### Stage 3 — Multi-GPU Single Node (`--num-gpus N`)

**What changes:** the colony is split evenly across N GPUs.  Each GPU
owns `n_ants/N` ants and runs its own construction + 2-opt kernels
independently.  After each GPU finishes, results are downloaded to
the unified CPU `ant[]` array.  The CPU then does a single
pheromone update over all ants (unchanged logic).

**No cross-GPU communication during construction** — ants are
independent.  The pheromone matrix is broadcast to all GPUs once per
iteration (upload to each device separately; with NVLink the copies
overlap).

Effective colony with 4 GPUs and 256 ants/GPU: **1024 ants** at the
same wall time as 256 ants on 1 GPU — better exploration.

### Stage 5 — Multi-Node MPI Island Model (`--mpi`)

**What changes:** each MPI rank runs a complete Stage 3 sub-colony
(4 GPUs, 1024 ants) independently.  Every `--island-sync K`
iterations the ranks exchange their best-so-far tour via
`MPI_Allreduce(MPI_MINLOC)` + `MPI_Bcast`.

The winning tour (shortest) is injected into all ranks as the new
best-so-far.  This allows MMAS trail limits and the restart
heuristic to react quickly to cross-node improvements.

**Pheromone:** in the current implementation pheromone is NOT shared
across nodes (loose island model).  This preserves colony diversity
and avoids expensive AllReduce over n×n doubles every iteration.
To enable tight pheromone sync add `MPI_Allreduce` over `pheromone[]`
flat array after each iteration (see commented code in `acotsp_hpc.c`).

**NCCL:** currently used for potential future all-reduce of the pheromone
matrix via `ncclAllReduce`.  Not active in the island-model default.

---

## Command-Line Flags

### Original flags (all still work)

```
-r, --tries N          number of independent trials
-s, --tours N          max tour constructions per trial
-t, --time T           max wall time per trial (seconds)
-i, --tsplibfile FILE  TSPLIB input file
-o, --optimum N        stop if tour ≤ N found
-m, --ants N           number of ants (colony size)
-g, --nnants N         nn_ants (construction candidate list depth)
-a, --alpha A          pheromone exponent
-b, --beta  B          heuristic exponent
-e, --rho   R          evaporation rate
-q, --q0    Q          ACS greedy probability
-l, --localsearch N    1=2-opt, 2=2.5-opt, 3=3-opt
    --mmas / --acs / --ras / --eas / --as / --bwas
```

### Top-K flag (from topk.c)

```
--topk N   (or -T N)   sparse pheromone update: only top-N ants deposit
```

### New HPC flags (added in acotsp_hpc.c)

```
--gpu               Stage 1: ant construction on GPU
--gpu-ls            Stage 2: 2-opt local search on GPU (implies --gpu)
--num-gpus N        Stage 3: use N GPUs per node (default 1, max 8)
--mpi               Stage 5: enable MPI island model
--island-sync K     MPI best-tour exchange every K iterations (default 100)
--no-perf           suppress performance summary at end
```

---

## Build Instructions (Taiwania)

```bash
# 1. Copy all source files to your scratch directory
cd $SCRATCH/acotsp_hpc

# 2. Load modules (adjust versions to what's installed)
module load cuda/11.8
module load openmpi/4.1.4
module load nccl/2.15.5
module load gcc/11.3.0

# 3. Build targets
make -f Makefile.hpc serial      # Stage 0 — serial baseline
make -f Makefile.hpc gpu         # Stage 1+2 — single GPU
make -f Makefile.hpc multi-gpu   # Stage 1+2+3 — multi-GPU node
make -f Makefile.hpc mpi         # Stage 1+2+3+5 — full MPI (default)
```

If NCCL or MPI is unavailable, edit `Makefile.hpc`:
- Remove `-DUSE_NCCL` and `$(NCCL_*)` for no-NCCL builds.
- Use `make gpu` target (no MPI) for single-node GPU runs.

---

## SLURM Submission

### Quick test (1 node, 1 GPU, Stage 1+2)
```bash
sbatch run_stage_test.sh 1 berlin52.tsp
```

### Full run (2 nodes × 4 GPU, Stage 5)
```bash
# Edit run_hpc.sh: set TSPFILE, OPTVAL, YOUR_PROJECT_ACCOUNT
sbatch run_hpc.sh
```

### Interactive debug (recommended first)
```bash
salloc --nodes=1 --gres=gpu:1 --cpus-per-task=4 --time=00:15:00 \
       --partition=gp4d
module load cuda/11.8 gcc/11.3.0
./acotsp_gpu -i berlin52.tsp --mmas -m 32 -g 20 -l 1 \
             -s 1000 -t 60 --gpu --gpu-ls
```

---

## Taiwania Node Spec Reminder

| Resource | Per node |
|---|---|
| CPU | 2× Intel Xeon Gold |
| GPU | 8× Nvidia Tesla V100 32 GB |
| RAM | 768 GB |
| Storage | 240 GB SSD + 4 TB NVMe |
| Network | InfiniBand EDR (100 Gb/s) |

Your allocation: 2 nodes, 1 hour.  Recommended config:

```
--nodes=2  --ntasks-per-node=1  --gres=gpu:4  --cpus-per-task=8
```

This uses 4 of 8 available GPUs per node (within the 1 CPU : 4 GPU
ratio requirement).  The other 4 GPUs remain idle — acceptable for
the project.  If you want all 8, change `--num-gpus 8` and
`--gres=gpu:8`, but verify your account has that entitlement.

---

## Expected Performance (rough estimates, berlin52 → lin318 scale)

| Stage | Config | Speedup vs serial |
|---|---|---|
| 0 serial | 1 CPU | 1× |
| 1 GPU construct | 1× V100 | 4–8× |
| 2 + GPU 2-opt | 1× V100 | 6–12× |
| 3 multi-GPU | 4× V100 | 20–40× |
| 5 MPI | 2 nodes × 4 V100 | 50–80× |

Actual numbers depend heavily on n (number of cities) and n_ants.
For small n (e.g., berlin52 with 52 cities) GPU overhead dominates;
use n ≥ 500 (e.g., rat783, pr1002) for meaningful GPU gains.

---

## Correctness Validation

Run Stage 0 and Stage 1 on the same seed and verify tour quality:

```bash
# Serial
./acotsp_serial -i berlin52.tsp --mmas -m 32 -g 20 -l 1 \
    -s 500 -t 60 -r 3 --seed 12345

# GPU Stage 1 (construction differs due to different PRNG, but
# solution quality should be statistically equivalent)
./acotsp_gpu -i berlin52.tsp --mmas -m 32 -g 20 -l 1 \
    -s 500 -t 60 -r 3 --gpu --seed 12345
```

Note: GPU tours use xorshift64 PRNG (not the CPU `ran01` LCG), so
individual ant paths will differ but the statistical distribution of
tour lengths and convergence behaviour should be equivalent.

---

## Known Limitations / Future Work

1. **GPU 2-opt apply**: the current `kernel_twoopt` uses thread 0 to
   apply improvements serially after threads detect them.  A fully
   parallel apply requires careful conflict resolution (e.g., Lin–
   Kernighan move locking).  This is left as a future enhancement.

2. **NCCL pheromone AllReduce**: skeleton is present but not wired up.
   To enable: add `ncclAllReduce(pheromone_flat, ...)` in
   `pheromone_trail_update()` before `compute_total_information()`.

3. **ACS local update on GPU**: `local_acs_pheromone_update()` runs
   during construction on CPU.  With `--acs` + `--gpu`, pheromone
   updates during construction are skipped (construction still runs
   on GPU, but the ACS local update is omitted).  Recommend using
   `--mmas` with GPU stages.

4. **Large n (> 4096)**: Stage 2 GPU 2-opt falls back gracefully to
   global-memory mode.  For n > 10000, consider using only Stage 1
   GPU construction + CPU 2-opt.
