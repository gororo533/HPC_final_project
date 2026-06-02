#!/bin/bash
#SBATCH --job-name=acotsp_hpc
#SBATCH --account=YOUR_PROJECT_ACCOUNT       # <-- 填你的 project account
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1                  # 1 MPI rank per node
#SBATCH --cpus-per-task=8                    # 8 CPU threads per rank
#SBATCH --gres=gpu:4                         # 4 V100s per node (Stage 1-3 use all 4)
#SBATCH --mem=128G
#SBATCH --time=01:00:00                      # 1 hour wall limit
#SBATCH --output=acotsp_%j.out
#SBATCH --error=acotsp_%j.err
#SBATCH --partition=gp4d                     # Taiwania GPU partition (adjust if needed)

# ================================================================
# Environment setup
# ================================================================
module purge
module load cuda/11.8          # or cuda/12.x — match your build
module load openmpi/4.1.4      # or your installed version
module load nccl/2.15.5        # match CUDA version
module load gcc/11.3.0

# Verify GPU visibility
echo "=== Node $SLURM_NODEID: GPU check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ================================================================
# Run configuration
# ================================================================

# --- TSP instance file (TSPLIB format) ---
TSPFILE="berlin52.tsp"          # replace with your instance
OPTVAL=7542                     # known optimum (use 0 if unknown)

# --- ACO algorithm flags (MMAS with 2-opt local search) ---
ACO_FLAGS="--mmas -m 256 -g 20 -a 1.0 -b 2.0 -e 0.02 \
           -s 100000 -t 3500 -r 1 \
           -o ${OPTVAL} -l 1 \
           -i ${TSPFILE}"

# --- HPC stage flags ---
# Stage 1+2+3+5: full GPU + MPI
HPC_FLAGS="--gpu --gpu-ls --num-gpus 4 --mpi --island-sync 50"

# --- Optional top-k sparse pheromone update ---
# Add:  --topk 5
# to activate; remove or set to 0 to disable.

# ================================================================
# Compilation (do this once before submitting, or keep here for safety)
# ================================================================
cd $SLURM_SUBMIT_DIR

# Uncomment to rebuild:
# make -f Makefile.hpc clean
# make -f Makefile.hpc mpi

if [ ! -f acotsp_hpc ]; then
    echo "ERROR: acotsp_hpc binary not found. Run: make -f Makefile.hpc mpi"
    exit 1
fi

# ================================================================
# Launch
# ================================================================
echo ""
echo "=== SLURM config ==="
echo "  Nodes       : $SLURM_NNODES"
echo "  Ranks/node  : $SLURM_NTASKS_PER_NODE"
echo "  CPUs/rank   : $SLURM_CPUS_PER_TASK"
echo "  GPUs/node   : 4"
echo "  TSP file    : $TSPFILE"
echo "  HPC flags   : $HPC_FLAGS"
echo "===================="
echo ""

# NVLink/InfiniBand tuning for NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0        # use InfiniBand if available
export NCCL_NET_GDR_LEVEL=2     # GPUDirect RDMA if InfiniBand supports it
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch 2 MPI ranks (1 per node), each using 4 GPUs
mpirun --np 2 \
       --npernode 1 \
       --bind-to none \
       ./acotsp_hpc ${ACO_FLAGS} ${HPC_FLAGS}

echo ""
echo "=== Job finished at $(date) ==="
