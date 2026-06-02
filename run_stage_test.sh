#!/bin/bash
# run_stage_test.sh — Individual stage test scripts for incremental validation
# Usage:  bash run_stage_test.sh <stage> <tspfile>
#   stage: 0 | 1 | 2 | 3 | 5
#   tspfile: path to TSPLIB .tsp file

STAGE=${1:-0}
TSPFILE=${2:-berlin52.tsp}
OPTVAL=7542

BASE_ACO="--mmas -m 128 -g 20 -a 1.0 -b 2.0 -e 0.02 \
          -s 50000 -t 1800 -r 1 -o ${OPTVAL} -l 1 -i ${TSPFILE}"

case $STAGE in
0)
    echo "=== Stage 0: Serial CPU baseline ==="
    sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=aco_s0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=stage0_%j.out
module load gcc/11.3.0
./acotsp_serial --mmas -m 128 -g 20 -a 1.0 -b 2.0 -e 0.02 \
    -s 50000 -t 1800 -r 1 -o 7542 -l 1 -i berlin52.tsp
EOF
    ;;

1)
    echo "=== Stage 1: GPU ant construction ==="
    sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=aco_s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=stage1_%j.out
#SBATCH --partition=gp4d
module load cuda/11.8 gcc/11.3.0
./acotsp_gpu --mmas -m 128 -g 20 -a 1.0 -b 2.0 -e 0.02 \
    -s 50000 -t 1800 -r 1 -o 7542 -l 1 -i berlin52.tsp \
    --gpu --num-gpus 1
EOF
    ;;

2)
    echo "=== Stage 2: GPU construction + GPU 2-opt ==="
    sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=aco_s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=stage2_%j.out
#SBATCH --partition=gp4d
module load cuda/11.8 gcc/11.3.0
./acotsp_gpu --mmas -m 128 -g 20 -a 1.0 -b 2.0 -e 0.02 \
    -s 50000 -t 1800 -r 1 -o 7542 -l 1 -i berlin52.tsp \
    --gpu --gpu-ls --num-gpus 1
EOF
    ;;

3)
    echo "=== Stage 3: 4-GPU single node ==="
    sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=aco_s3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=stage3_%j.out
#SBATCH --partition=gp4d
module load cuda/11.8 gcc/11.3.0
./acotsp_multigpu --mmas -m 512 -g 20 -a 1.0 -b 2.0 -e 0.02 \
    -s 200000 -t 3600 -r 1 -o 7542 -l 1 -i berlin52.tsp \
    --gpu --gpu-ls --num-gpus 4
EOF
    ;;

5)
    echo "=== Stage 5: 2-node MPI + 4 GPU per node ==="
    sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=aco_s5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=stage5_%j.out
#SBATCH --partition=gp4d
module load cuda/11.8 openmpi/4.1.4 nccl/2.15.5 gcc/11.3.0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
mpirun --np 2 --npernode 1 --bind-to none \
    ./acotsp_hpc --mmas -m 1024 -g 20 -a 1.0 -b 2.0 -e 0.02 \
    -s 400000 -t 3600 -r 1 -o 7542 -l 1 -i berlin52.tsp \
    --gpu --gpu-ls --num-gpus 4 --mpi --island-sync 50
EOF
    ;;

*)
    echo "Unknown stage: $STAGE  (use 0, 1, 2, 3, or 5)"
    exit 1
    ;;
esac
